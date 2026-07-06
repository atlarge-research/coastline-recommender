"""Tests for OpenDC energy prediction modules.

Tests topology building, MAPE calculation, and strict error handling.
OpenDC binary execution tests are skipped if the binary is not available.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from coastline.sdk.io.odc_runner.runner import OpenDCRunnerError
from coastline.sdk.predictors.energy.opendc.mape import MapeComparator
from coastline.sdk.predictors.energy.opendc.predictor import OpenDCEnergyPredictor

# NOTE: topology-builder and OpenDCRunner construction/parquet tests live in
# common/tests/test_odc_runner.py; this file keeps only the MAPE comparator and
# the strict-error-propagation tests that are unique here.


class TestMapeComparator:
    def _make_power_df(self, start, n_points, base_power, noise=0):
        timestamps = [start + timedelta(seconds=60 * i) for i in range(n_points)]
        power = [base_power + noise * np.sin(i) for i in range(n_points)]
        return pd.DataFrame({"timestamp": timestamps, "power_draw": power})

    def test_identical_series_gives_exactly_zero_mape(self):
        # Analytic reference point: sim == actual at every point => every relative
        # error |a-s|/a is exactly 0, so their mean (the MAPE) is exactly 0.0.
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        df = self._make_power_df(start, 20, 300.0)
        comp = MapeComparator(mape_window_minutes=30)
        result = comp.compare(df, df.copy(), start + timedelta(minutes=20))
        assert result["mape"] == pytest.approx(0.0, abs=1e-9)

    def test_mape_is_mean_of_per_point_relative_errors(self):
        # Four aligned points with DIFFERENT per-point errors, so this pins the
        # aggregation (mean over points) AND the denominator (actual, not sim):
        #   actual = [100, 100, 100, 200], sim = [110, 90, 100, 240]
        #   |a-s|/a * 100 = [10, 10, 0, 20]  -> MAPE = (10+10+0+20)/4 = 10.0
        # A bug that divided by sim (10/110, 10/90, 0, 40/240) or summed instead of
        # averaged would land far from 10.0.
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        stamps = [start + timedelta(seconds=60 * i) for i in range(4)]
        actual = pd.DataFrame({"timestamp": stamps, "power_draw": [100.0, 100.0, 100.0, 200.0]})
        sim = pd.DataFrame({"timestamp": stamps, "power_draw": [110.0, 90.0, 100.0, 240.0]})
        comp = MapeComparator(mape_window_minutes=30)
        result = comp.compare(sim, actual, start + timedelta(minutes=5))
        assert result["mape"] == pytest.approx(10.0)
        # Reported window means are plain arithmetic means, hand-derived:
        #   mean_actual    = (100+100+100+200)/4 = 125.0
        #   mean_simulated = (110+ 90+100+240)/4 = 135.0
        assert result["mean_actual"] == pytest.approx(125.0)
        assert result["mean_simulated"] == pytest.approx(135.0)

    def test_empty_dataframe_raises(self):
        comp = MapeComparator()
        with pytest.raises(ValueError, match="Empty"):
            comp.compare(
                pd.DataFrame({"timestamp": [], "power_draw": []}),
                pd.DataFrame({"timestamp": [], "power_draw": []}),
                datetime.now(timezone.utc),
            )

    def test_missing_column_raises(self):
        comp = MapeComparator()
        df = pd.DataFrame({"timestamp": [datetime.now(timezone.utc)], "watts": [100]})
        with pytest.raises(ValueError, match="power_draw"):
            comp.compare(df, df, datetime.now(timezone.utc))


class TestRuntimeReporting:
    """OpenDC simulates power, not job duration. When no job size (total_tokens/
    epochs) is supplied, Kavier's summary carries train_runtime == 0.0 — a sentinel
    meaning 'unknown', NOT a real zero-second runtime. The predictor must surface it
    as ``predicted_runtime_seconds is None`` rather than a misleading 0.0.
    """

    def _workload(self):
        from coastline.sdk.models.workload import WorkloadSpec

        return WorkloadSpec(
            llm_model="test-model",
            fine_tuning_method="lora",
            gpu_model="NVIDIA-A100-SXM4-80GB",
            tokens_per_sample=1024,
            batch_size=4,
            gpus_per_node=2,
            number_of_nodes=1,
        )

    def _context(self):
        from coastline.sdk.models.context import Constraints, SystemContext

        return SystemContext(
            available_gpu_models=["NVIDIA-A100-SXM4-80GB"],
            max_gpus=8,
            gpu_memory={"NVIDIA-A100-SXM4-80GB": 80},
            constraints=Constraints(max_gpus=8, gpus_per_node=8, max_nodes=1),
        )

    def _patched_predictor(self, summary):
        """Build an OpenDCEnergyPredictor whose Kavier export + OpenDC run are stubbed,
        returning the given `summary` and a fixed non-empty power trace."""
        from coastline.sdk.predictors.energy.opendc import predictor as predictor_mod

        power_df = pd.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1, tzinfo=timezone.utc)],
                "power_draw": [400.0],
            }
        )
        tasks_df = pd.DataFrame([{"gpu_capacity": 1.0, "cpu_capacity": 1.0, "gpu_count": 2}])
        frags_df = pd.DataFrame([{"gpu_usage": 0.0}])

        pred = OpenDCEnergyPredictor.__new__(OpenDCEnergyPredictor)
        pred._calibration_factor = 1.0
        pred._timeout = 5
        pred._bin_path = None
        pred._max_workers = 1
        pred._runner = MagicMock()
        pred._runner.run_simulation.return_value = power_df

        # build_training_opendc_frames / prepare_opendc_input are imported lazily inside
        # predict() (so a kavier rename can't break module import), so patch them at the
        # source; _fix_fragment_utilization is a local function, patched on the module.
        patches = [
            patch(
                "kavier.sdk.io.training_opendc.build_training_opendc_frames", return_value=(tasks_df, frags_df, summary)
            ),
            patch("kavier.sdk.io.opendc.adapter.prepare_opendc_input", return_value=None),
            patch.object(predictor_mod, "_fix_fragment_utilization", side_effect=lambda t, f: f),
        ]
        return pred, patches

    def test_zero_train_runtime_reported_as_none(self):
        # total_tokens=None path -> Kavier summary has train_runtime == 0.0.
        pred, patches = self._patched_predictor({"train_runtime": 0.0})
        for p in patches:
            p.start()
        try:
            result = pred.predict(self._workload(), self._context())
        finally:
            for p in patches:
                p.stop()
        assert result.predicted_runtime_seconds is None, (
            "train_runtime==0.0 (no job size) must surface as None, not a misleading 0.0"
        )

    def test_real_train_runtime_flows_through(self):
        # If a real runtime ever flows in, it must be preserved (not nulled).
        pred, patches = self._patched_predictor({"train_runtime": 1234.5})
        for p in patches:
            p.start()
        try:
            result = pred.predict(self._workload(), self._context())
        finally:
            for p in patches:
                p.stop()
        assert result.predicted_runtime_seconds == pytest.approx(1234.5)

    def test_total_datacenter_power_divided_to_per_gpu(self):
        # OpenDC's power_draw is the TOTAL datacenter draw. _patched_predictor feeds a
        # constant 400 W trace for a 2-GPU job (gpus_per_node=2, number_of_nodes=1 ->
        # total_gpus=2). The predictor must expose PER-GPU watts so it matches
        # KavierPowerPredictor's convention (else scoring double-counts ×total_gpus):
        #   per_gpu = mean_total_power / total_gpus = 400 / 2 = 200 W.
        pred, patches = self._patched_predictor({"train_runtime": 500.0})
        for p in patches:
            p.start()
        try:
            result = pred.predict(self._workload(), self._context())
        finally:
            for p in patches:
                p.stop()
        assert result.total_gpus == 2  # 2 gpus/node x 1 node
        assert result.predicted_power == pytest.approx(200.0)  # 400 total / 2 GPUs
        # metadata keeps both views: raw total draw and the per-GPU figure it exposes.
        assert result.metadata["mean_power_watts"] == pytest.approx(400.0)
        assert result.metadata["mean_power_watts_per_gpu"] == pytest.approx(200.0)


class TestStrictErrorHandling:
    """Verify strict errors for system-level failures (no silent fallbacks)."""

    def test_workflow_raises_on_empty_grid(self):
        """If all candidates fail prediction, raise RuntimeError."""
        from coastline.sdk.models.context import Constraints, SystemContext
        from coastline.sdk.models.workload import WorkloadSpec
        from coastline.sdk.pipeline.feasibility import NoOpFeasibilityChecker
        from coastline.sdk.pipeline.workflow import GridWorkflowPipeline

        mock_throughput = MagicMock()
        mock_throughput.get_name.return_value = "mock_throughput"
        mock_throughput.predict.return_value = None

        pipeline = GridWorkflowPipeline.from_config(
            config={"grid": {"batch_sizes": [1], "total_gpus": [1], "top_k": 1}},
            selection_policy="min_gpu",
            strategy_name="test",
            throughput_predictor=mock_throughput,
            power_predictor=MagicMock(),
            feasibility_checker=NoOpFeasibilityChecker(),
        )

        workload = WorkloadSpec(
            llm_model="test",
            fine_tuning_method="lora",
            gpu_model="NVIDIA-A100-SXM4-80GB",
            tokens_per_sample=1024,
            batch_size=1,
        )
        context = SystemContext(
            available_gpu_models=["NVIDIA-A100-SXM4-80GB"],
            max_gpus=8,
            gpu_memory={"NVIDIA-A100-SXM4-80GB": 80},
            constraints=Constraints(max_gpus=8, gpus_per_node=8, max_nodes=1),
        )

        with pytest.raises(RuntimeError, match="no feasible candidates"):
            pipeline.recommend(workload, context)

    def test_predictor_exception_propagates(self):
        """If a predictor raises (e.g. OpenDC fails), it propagates through the loop."""
        from coastline.sdk.models.context import Constraints, SystemContext
        from coastline.sdk.models.recommendation import Prediction
        from coastline.sdk.models.workload import WorkloadSpec
        from coastline.sdk.pipeline.feasibility import NoOpFeasibilityChecker
        from coastline.sdk.pipeline.workflow import GridWorkflowPipeline

        mock_throughput = MagicMock()
        mock_throughput.predict.return_value = Prediction(
            gpus_per_node=1,
            number_of_nodes=1,
            total_gpus=1,
            predicted_throughput=100.0,
            predicted_power=None,
        )
        mock_power = MagicMock()
        mock_power.predict.side_effect = OpenDCRunnerError("binary missing")

        pipeline = GridWorkflowPipeline.from_config(
            config={"grid": {"batch_sizes": [1], "total_gpus": [1], "top_k": 1}},
            selection_policy="min_gpu",
            strategy_name="test",
            throughput_predictor=mock_throughput,
            power_predictor=mock_power,
            feasibility_checker=NoOpFeasibilityChecker(),
        )

        workload = WorkloadSpec(
            llm_model="test",
            fine_tuning_method="lora",
            gpu_model="NVIDIA-A100-SXM4-80GB",
            tokens_per_sample=1024,
            batch_size=1,
        )
        context = SystemContext(
            available_gpu_models=["NVIDIA-A100-SXM4-80GB"],
            max_gpus=8,
            gpu_memory={"NVIDIA-A100-SXM4-80GB": 80},
            constraints=Constraints(max_gpus=8, gpus_per_node=8, max_nodes=1),
        )

        with pytest.raises(OpenDCRunnerError, match="binary missing"):
            pipeline.recommend(workload, context)

    def test_unknown_energy_predictor_raises(self):
        from coastline.sdk.pipeline.workflow import _create_power_predictor

        with pytest.raises(ValueError, match="Unknown energy predictor"):
            _create_power_predictor({"energy": "nonexistent"})

    def test_unknown_strategy_raises(self):
        from coastline.sdk.policies import PolicyFactory

        with pytest.raises(ValueError, match="Unknown strategy"):
            PolicyFactory.create_strategy(strategy_name="nonexistent")
