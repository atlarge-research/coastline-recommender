"""Unit tests for the energy predictors.

Covers the two energy predictors under
``recommender/predictors/energy/``:

* ``KavierPowerPredictor`` (``kavier_power_predictor.py``) — a thin wrapper that
  delegates to the analytical Kavier physics predictor and surfaces GPU power.
* ``OpenDCEnergyPredictor`` (``opendc/predictor.py``) — runs the Kavier export
  through the OpenDC datacenter simulator and reports mean power draw.

Design notes
------------
* No ML model artifacts are loaded. Kavier is purely analytical, so its tests
  run end-to-end and are deterministic.
* ``OpenDCEnergyPredictor`` needs (a) the importable Kavier OpenDC export
  (``kavier.sdk.io.opendc.adapter`` / ``kavier.sdk.io.training_opendc`` /
  ``kavier.sdk.training.core.engine``) and (b) the compiled OpenDC binary (a Java
  subprocess). Running the *actual* binary is slow and timing-dependent, so the
  predictor's pure-Python interface (frame build, utilisation fix, mean-power
  extraction, return shape, error guards) is exercised with
  ``OpenDCRunner.run_simulation`` mocked to return a synthetic power trace.
  Construction of the predictor builds a real ``OpenDCRunner``, which requires
  the binary on disk; tests that construct it are skipped when the binary /
  export are unavailable. The gate is scoped to the OpenDC classes only — the
  Kavier and pure-Python helper tests always run.

IMPORTANT: this file is test-only and must not import or mutate prod modules
beyond reading their public behaviour.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.recommendation import Prediction
from coastline.sdk.models.workload import WorkloadSpec

# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #

# A GPU/model pair that is calibrated in Kavier's library (see the project docs).
SUPPORTED_GPU = "NVIDIA-A100-80GB-PCIe"
SUPPORTED_MODEL = "mistral-7b-v0.1"

# NVIDIA A100 80GB PCIe datasheet: TDP = 300 W; idle draw is ~60 W. These are
# hardware facts independent of Kavier's power model — a valid physical envelope
# for any per-GPU power figure the analytical engine reports under training load.
A100_PCIE_TDP_W = 300.0
A100_PCIE_IDLE_W = 60.0


@pytest.fixture
def context():
    """System context advertising a Kavier-supported A100 GPU."""
    return SystemContext(
        available_gpu_models=[SUPPORTED_GPU],
        max_gpus=32,
        gpu_memory={SUPPORTED_GPU: 80},
        constraints=Constraints(max_gpus=32, gpus_per_node=8, max_nodes=4),
    )


def _workload(model=SUPPORTED_MODEL, gpu=SUPPORTED_GPU, gpus_per_node=1, number_of_nodes=1):
    """Build a WorkloadSpec.

    Per the Coastline/ado convention, ``gpus_per_node`` carries the job's TOTAL GPU
    count (it mirrors the trace's ``number_gpus``); the Kavier engine splits it
    across ``number_of_nodes`` internally.
    """
    return WorkloadSpec(
        llm_model=model,
        fine_tuning_method="lora",
        gpu_model=gpu,
        tokens_per_sample=2048,
        batch_size=16,
        gpus_per_node=gpus_per_node,
        number_of_nodes=number_of_nodes,
    )


# --------------------------------------------------------------------------- #
# KavierPowerPredictor
# --------------------------------------------------------------------------- #

from coastline.sdk.predictors.energy.kavier.kavier_power_predictor import KavierPowerPredictor  # noqa: E402


class TestKavierPowerPredictor:
    @pytest.mark.parametrize(
        "model, gpu",
        [
            ("totally-not-a-real-model-xyz", SUPPORTED_GPU),  # unsupported model key
            (SUPPORTED_MODEL, "FAKE-GPU-9000"),  # unsupported GPU key
        ],
    )
    def test_out_of_library_config_maps_to_none(self, context, model, gpu):
        """Contract: an out-of-library (model or GPU) config yields None, not a Prediction.

        Kavier raises KeyError for an unknown model/GPU and returns an error
        Prediction with ``predicted_power=None``; the power wrapper maps that
        None/<=0 power to None. Falsifies a body that returned any Prediction
        (or a positive-power fabrication) for an uncalibrated config. Both
        catalog dimensions are exercised because they hit distinct KeyErrors.
        """
        pred = KavierPowerPredictor().predict(_workload(model=model, gpu=gpu), context)
        assert pred is None

    def test_per_gpu_power_is_invariant_to_gpu_count_and_within_hardware_envelope(self, context):
        """Kavier's per-GPU watts is a PER-GPU figure, so it must not change with fleet size.

        Independent oracles (no pinned engine number):
        * INVARIANT — ``predicted_power`` is identical for 1/2/4/8 GPUs (per-GPU,
          not total). A bug returning TOTAL datacenter watts would make the 8-GPU
          figure ~8x the 1-GPU figure -> the set would have >1 element.
        * PHYSICAL BOUND — per-GPU watts lie in [idle, TDP] = [60, 300] W for the
          A100 80GB PCIe (datasheet). A "total power not divided by GPUs" bug
          would push the 8-GPU value to ~1276 W, well above TDP.
        * BOOKKEEPING — ``total_gpus`` tracks the requested count exactly.
        """
        predictor = KavierPowerPredictor()
        preds = {n: predictor.predict(_workload(gpus_per_node=n), context) for n in (1, 2, 4, 8)}

        for n, pred in preds.items():
            assert pred is not None, f"expected prediction for {n} GPUs"
            assert pred.total_gpus == n, "total_gpus must reflect the requested GPU count"
            assert A100_PCIE_IDLE_W <= pred.predicted_power <= A100_PCIE_TDP_W, (
                f"per-GPU power {pred.predicted_power}W outside hardware envelope "
                f"[{A100_PCIE_IDLE_W}, {A100_PCIE_TDP_W}] at {n} GPUs"
            )

        per_gpu = {n: pred.predicted_power for n, pred in preds.items()}
        assert len(set(per_gpu.values())) == 1, f"per-GPU power should be invariant to GPU count, got {per_gpu}"

    def test_total_throughput_scales_up_but_sublinearly_with_gpu_count(self, context):
        """Scaling law: more GPUs -> more total throughput, but < N-times (comm overhead).

        Oracle is a scaling law, not a magic value:
        * MONOTONE — throughput strictly increases from 1 -> 8 GPUs (adding compute
          must not lose throughput). Catches a regression where multi-GPU is slower.
        * SUB-LINEAR — 8-GPU total throughput < 8 x single-GPU throughput; perfect
          linear scaling is physically impossible once inter-GPU communication
          costs anything. Catches a bug that scaled throughput by GPU count with
          no communication penalty (would give >= 8x).
        """
        predictor = KavierPowerPredictor()
        thr = {n: predictor.predict(_workload(gpus_per_node=n), context).predicted_throughput for n in (1, 2, 4, 8)}

        assert thr[1] < thr[2] < thr[4] < thr[8], f"throughput must grow with GPUs, got {thr}"
        assert thr[8] < 8 * thr[1], f"8-GPU throughput {thr[8]} must be sub-linear vs 8 x single-GPU {8 * thr[1]}"
        assert thr[2] < 2 * thr[1], f"2-GPU throughput {thr[2]} must be sub-linear vs {2 * thr[1]}"


# --------------------------------------------------------------------------- #
# OpenDC energy predictor
# --------------------------------------------------------------------------- #
#
# The predictor MODULE imports without the Kavier OpenDC export (its kavier.sdk.io
# imports are lazy, inside methods), so we import it unconditionally and gate the
# tests that actually *run* a prediction. The stale ``opendc.adapter`` guard that
# previously lived here targeted an unrelated top-level package the code never
# imports and dead-skipped the whole file.

from coastline.sdk.io.odc_runner.runner import OpenDCRunner, OpenDCRunnerError  # noqa: E402
from coastline.sdk.predictors.energy.opendc.predictor import (  # noqa: E402
    OpenDCEnergyPredictor,
    _fix_fragment_utilization,
)


def _opendc_export_available() -> bool:
    """True when the Kavier OpenDC export path the predictor calls is importable."""
    try:
        import kavier.sdk.io.opendc.adapter  # noqa: F401
        import kavier.sdk.io.training_opendc  # noqa: F401
        import kavier.sdk.training.core.engine  # noqa: F401

        return True
    except Exception:
        return False


def _opendc_binary_available() -> bool:
    """True if a real OpenDC runner can be constructed (binary present).

    ``OpenDCEnergyPredictor.__init__`` builds an ``OpenDCRunner``, which raises
    ``FileNotFoundError`` when the compiled binary is missing. Tests that need to
    *construct* the predictor are gated on this.
    """
    try:
        OpenDCRunner()
        return True
    except FileNotFoundError:
        return False


# The predict tests build the Kavier export for real (only run_simulation is
# mocked) and construct the predictor, so they need both the export and binary.
_NEED_OPENDC = pytest.mark.skipif(
    not (_opendc_export_available() and _opendc_binary_available()),
    reason="OpenDC export and/or compiled binary not available",
)


def _synthetic_power_df(values):
    """A deterministic power timeseries standing in for OpenDC's parquet output."""
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(seconds=60 * i) for i in range(len(values))]
    return pd.DataFrame({"timestamp": timestamps, "power_draw": list(values)})


class TestOpenDCFragmentUtilization:
    """Pure-Python helper: no binary, no Kavier export — fully deterministic."""

    def test_rescales_cpu_usage_by_capacity_ratio_independent_of_gpu_count(self):
        """cpu_usage is rewritten to encode true GPU utilisation for the MSE power model.

        The impl computes ``util = gpu_usage / (gpu_cap*total_gpus)`` then
        ``cpu_usage = util * cpu_cap * total_gpus``. Algebraically ``total_gpus``
        CANCELS, leaving ``cpu_usage = gpu_usage * cpu_cap / gpu_cap`` — a
        different form used here as the independent oracle.

        Inputs (deliberately cpu_cap != gpu_cap, total_gpus != 1 so a term-drop
        shows): gpu_usage=1500, gpu_cap=2000, cpu_cap=1000, total_gpus=3.
        By hand: 1500 * 1000 / 2000 = 750. (Cross-check via impl form:
        util = 1500/(2000*3) = 0.25; 0.25 * 1000 * 3 = 750.)
        """
        tasks = pd.DataFrame({"gpu_capacity": [2000.0], "cpu_capacity": [1000.0], "gpu_count": [3]})
        fragments = pd.DataFrame({"gpu_usage": [1500.0], "cpu_usage": [999.0]})
        out = _fix_fragment_utilization(tasks, fragments)
        assert out["cpu_usage"].iloc[0] == pytest.approx(750.0)

    def test_noop_when_zero_gpu_capacity(self):
        """Guard: gpu_capacity <= 0 would divide-by-zero, so the frame is returned untouched.

        Oracle: the original cpu_usage (7.0) must survive verbatim. A body that
        dropped the guard would compute 1000/(0*2) -> inf/NaN, not 7.0.
        """
        tasks = pd.DataFrame({"gpu_capacity": [0.0], "cpu_capacity": [1000.0], "gpu_count": [2]})
        fragments = pd.DataFrame({"gpu_usage": [1000.0], "cpu_usage": [7.0]})
        out = _fix_fragment_utilization(tasks, fragments)
        assert out["cpu_usage"].iloc[0] == 7.0  # unchanged


@_NEED_OPENDC
class TestOpenDCEnergyPredictor:
    """Predictor interface tests with the OpenDC binary subprocess mocked out.

    The Kavier export (frame building) runs for real — it is analytical and
    fast — but ``OpenDCRunner.run_simulation`` is patched so no Java subprocess
    is launched, keeping the tests fast and deterministic.
    """

    def test_predict_reports_mean_total_power_divided_per_gpu(self, context):
        """predicted_power = mean(power_draw) / total_gpus; total preserved in metadata.

        Independent oracles from a hand-picked trace (200, 500, 200) W with
        total_gpus=2:
        * mean = (200+500+200)/3 = 300 W total (distinct from max=500, min=200,
          and the first sample=200 — so a "return max/first" bug is caught).
        * per-GPU = 300 / 2 = 150 W (the double-count fix: OpenDC reports TOTAL
          datacenter watts, must be divided by GPU count). A missing division
          would leave predicted_power at 300.
        * max/min metadata are the trace extremes 500 / 200.
        * OpenDC is an ENERGY predictor: throughput is None; with total_tokens
          unset Kavier's train_runtime is a 0.0 sentinel -> runtime surfaces None.
        """
        with patch.object(OpenDCRunner, "run_simulation", return_value=_synthetic_power_df((200.0, 500.0, 200.0))):
            pred = OpenDCEnergyPredictor(calibration_factor=1.0).predict(_workload(gpus_per_node=2), context)

        assert isinstance(pred, Prediction)
        assert pred.total_gpus == 2
        assert pred.predicted_power == pytest.approx(150.0)
        assert pred.metadata["mean_power_watts"] == pytest.approx(300.0)
        assert pred.metadata["mean_power_watts_per_gpu"] == pytest.approx(150.0)
        assert pred.metadata["max_power_watts"] == pytest.approx(500.0)
        assert pred.metadata["min_power_watts"] == pytest.approx(200.0)
        assert pred.predicted_throughput is None
        assert pred.predicted_runtime_seconds is None

    def test_per_gpu_division_scales_with_gpu_count(self, context):
        """Same TOTAL trace, more GPUs -> proportionally lower per-GPU watts.

        Cross-check/scaling oracle independent of the exact wattage: with an
        identical mocked trace (mean total = 300 W), 4 GPUs must report exactly
        half the per-GPU power of 2 GPUs, because per-GPU = total / count and the
        total is held fixed. 300/2 = 150 vs 300/4 = 75, ratio 2.0. Catches a bug
        that divided by a constant (or not at all) instead of total_gpus.
        """
        trace = _synthetic_power_df((200.0, 500.0, 200.0))  # mean total = 300 W
        with patch.object(OpenDCRunner, "run_simulation", return_value=trace):
            p2 = OpenDCEnergyPredictor().predict(_workload(gpus_per_node=2), context)
            p4 = OpenDCEnergyPredictor().predict(_workload(gpus_per_node=4), context)

        # Total draw is a property of the (fixed) trace, not the GPU count.
        assert p2.metadata["mean_power_watts"] == pytest.approx(p4.metadata["mean_power_watts"])
        assert p2.predicted_power == pytest.approx(2.0 * p4.predicted_power)
        assert p4.predicted_power == pytest.approx(75.0)  # 300 / 4 by hand

    def test_empty_power_trace_raises(self, context):
        """Contract: an empty trace is a hard error (strict predictor, no fallback)."""
        empty = pd.DataFrame({"timestamp": [], "power_draw": []})
        with patch.object(OpenDCRunner, "run_simulation", return_value=empty):
            with pytest.raises(OpenDCRunnerError, match="empty power trace"):
                OpenDCEnergyPredictor().predict(_workload(), context)

    def test_non_positive_mean_power_raises(self, context):
        """Contract: mean(0,0,0) = 0 <= 0 -> raises rather than emitting bogus 0 W power."""
        zeros = _synthetic_power_df((0.0, 0.0, 0.0))
        with patch.object(OpenDCRunner, "run_simulation", return_value=zeros):
            with pytest.raises(OpenDCRunnerError, match="non-positive mean power"):
                OpenDCEnergyPredictor().predict(_workload(), context)


class TestOpenDCConstructionWithoutBinary:
    """When the binary is missing, construction must fail loudly (no fallback)."""

    @pytest.mark.skipif(
        _opendc_binary_available(),
        reason="binary present; cannot exercise the missing-binary path here",
    )
    def test_construction_raises_without_binary(self):
        with pytest.raises(FileNotFoundError):
            OpenDCEnergyPredictor()

    def test_construction_raises_with_explicit_bad_path(self):
        """An explicit nonexistent binary path always raises, regardless of env."""
        with pytest.raises(FileNotFoundError, match="OpenDC binary not found"):
            OpenDCEnergyPredictor(opendc_bin_path=Path("/nonexistent/opendc/binary"))
