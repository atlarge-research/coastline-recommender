"""Tests for ``finalize_ml_prediction`` in ``ml_common``, which turns a raw model output into a
``Prediction``:
  * a missing (None) or non-finite throughput gives None, so bad values do not reach scoring;
  * a negative finite throughput is clamped to 0.0;
  * a non-finite runtime becomes None and a negative one 0.0;
  * total_gpus is gpus_per_node x number_of_nodes.
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
        # A missing or non-finite (NaN, +-inf) throughput gives no Prediction at all.
        assert finalize_ml_prediction(_wl(), throughput=bad, runtime_seconds=100.0, metadata={}) is None

    def test_negative_throughput_clamped_to_zero(self):
        # max(-5.0, 0.0) == 0.0; a finite negative still gives a Prediction.
        p = finalize_ml_prediction(_wl(), throughput=-5.0, runtime_seconds=100.0, metadata={})
        assert p is not None
        assert p.predicted_throughput == 0.0

    def test_finite_positive_throughput_passes_through_unchanged(self):
        # Finite positive values and the metadata pass through unchanged; total_gpus = 4 x 2 = 8.
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
        # Doubling number_of_nodes from 2 to 4 at 4 GPUs per node doubles total_gpus from 8 to 16.
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
        # A non-finite runtime becomes None, and a valid throughput still gives a Prediction.
        p = finalize_ml_prediction(_wl(), throughput=500.0, runtime_seconds=bad_runtime, metadata={})
        assert p is not None
        assert p.predicted_throughput == 500.0
        assert p.predicted_runtime_seconds is None

    def test_negative_runtime_clamped_to_zero(self):
        # max(-10.0, 0.0) == 0.0; a finite negative runtime stays a number.
        p = finalize_ml_prediction(_wl(), throughput=500.0, runtime_seconds=-10.0, metadata={})
        assert p is not None
        assert p.predicted_runtime_seconds == 0.0

    def test_runtime_none_stays_none(self):
        # runtime_seconds=None stays None without reaching float(), and the Prediction is still made.
        p = finalize_ml_prediction(_wl(), throughput=500.0, runtime_seconds=None, metadata={})
        assert p is not None
        assert p.predicted_throughput == 500.0
        assert p.predicted_runtime_seconds is None
