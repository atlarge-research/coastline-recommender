"""Unit tests for shared ML inference helpers in ``ml_common``.

Focus: ``finalize_ml_prediction`` is the guard that turns a raw model output into a
``Prediction``. Its contract (from the docstring):
  * throughput missing (None) or non-finite (NaN/inf) -> return None entirely
    (NOT clamped to 0 -- clamping would let garbage flow into downstream scoring)
  * negative-but-finite throughput -> clamped to 0.0
  * non-finite / negative runtime -> None / 0.0 respectively
  * total_gpus derived as gpus_per_node * number_of_nodes
Each assertion below carries an independent oracle (a hand value or the stated contract
branch) so it can only pass for the correct behavior.
"""

import pytest

from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.performance.data_driven.ml_common import finalize_ml_prediction


def _wl(gpus_per_node=4, number_of_nodes=2):
    return WorkloadSpec(
        llm_model="mistral-7b-v0.1",
        fine_tuning_method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=32,
        gpus_per_node=gpus_per_node,
        number_of_nodes=number_of_nodes,
    )


class TestFinalizeMlPrediction:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None])
    def test_missing_or_non_finite_throughput_returns_none(self, bad):
        # Contract: a missing (None) or non-finite (NaN/+-inf) throughput must abort the
        # whole Prediction. Falsification: if the code instead did throughput=max(bad,0)
        # it would emit a Prediction carrying NaN/inf (or 0 for None) into scoring -- this
        # asserts the None-return branch that prevents that corruption.
        assert finalize_ml_prediction(_wl(), throughput=bad, runtime_seconds=100.0, metadata={}) is None

    def test_negative_throughput_clamped_to_zero(self):
        # Oracle: clamp is max(throughput, 0.0); by hand max(-5.0, 0.0) == 0.0.
        # (A finite negative is a real value, so it survives as a Prediction -- unlike NaN.)
        p = finalize_ml_prediction(_wl(), throughput=-5.0, runtime_seconds=100.0, metadata={})
        assert p is not None
        assert p.predicted_throughput == 0.0

    def test_finite_positive_throughput_passes_through_unchanged(self):
        # Oracle: total_gpus = gpus_per_node * number_of_nodes = 4 * 2 = 8 (hand-derived,
        # independent of the passed-through throughput/runtime). A finite positive
        # throughput/runtime must be preserved verbatim (max(500,0)=500, not re-scaled),
        # and metadata handed straight through.
        p = finalize_ml_prediction(
            _wl(gpus_per_node=4, number_of_nodes=2),
            throughput=500.0,
            runtime_seconds=120.0,
            metadata={"predictor": "x"},
        )
        assert p is not None
        assert p.predicted_throughput == 500.0
        assert p.predicted_runtime_seconds == 120.0
        assert p.total_gpus == 8  # 4 * 2
        assert p.metadata["predictor"] == "x"

    def test_total_gpus_scales_with_node_count(self):
        # Scaling oracle: doubling number_of_nodes (2 -> 4) at fixed gpus_per_node=4 must
        # exactly double total_gpus (8 -> 16). Pins the multiply, not a snapshot: catches a
        # bug that used only gpus_per_node or added instead of multiplied.
        p2 = finalize_ml_prediction(
            _wl(gpus_per_node=4, number_of_nodes=2), throughput=1.0, runtime_seconds=1.0, metadata={}
        )
        p4 = finalize_ml_prediction(
            _wl(gpus_per_node=4, number_of_nodes=4), throughput=1.0, runtime_seconds=1.0, metadata={}
        )
        assert p2.total_gpus == 8  # 4 * 2
        assert p4.total_gpus == 16  # 4 * 4

    @pytest.mark.parametrize("bad_runtime", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_runtime_becomes_none_but_keeps_prediction(self, bad_runtime):
        # Contract: a non-finite runtime is dropped to None, yet a valid throughput still
        # yields a Prediction (runtime failure must not nuke the whole result). Oracle: the
        # `max(rt,0) if isfinite else None` branch -> None for NaN/+-inf.
        p = finalize_ml_prediction(_wl(), throughput=500.0, runtime_seconds=bad_runtime, metadata={})
        assert p is not None
        assert p.predicted_throughput == 500.0
        assert p.predicted_runtime_seconds is None

    def test_negative_runtime_clamped_to_zero(self):
        # Oracle: runtime clamp is max(runtime, 0.0); by hand max(-10.0, 0.0) == 0.0.
        # Distinct branch from the non-finite case (finite-but-negative stays a number).
        p = finalize_ml_prediction(_wl(), throughput=500.0, runtime_seconds=-10.0, metadata={})
        assert p is not None
        assert p.predicted_runtime_seconds == 0.0

    def test_runtime_none_stays_none(self):
        # Contract: runtime_seconds=None short-circuits before float() (distinct branch from
        # the non-finite path), so a None input yields predicted_runtime_seconds is None
        # while the (valid) throughput still produces a Prediction.
        p = finalize_ml_prediction(_wl(), throughput=500.0, runtime_seconds=None, metadata={})
        assert p is not None
        assert p.predicted_throughput == 500.0
        assert p.predicted_runtime_seconds is None
