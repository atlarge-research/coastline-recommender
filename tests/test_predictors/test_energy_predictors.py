"""Tests for ``KavierPowerPredictor`` (``sdk/predictors/energy/kavier/kavier_power_predictor.py``),
which wraps the Kavier physics predictor and reports GPU power.

No ML model artifacts are loaded; Kavier is analytical, so these tests are deterministic.
"""

import pytest

from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.energy.kavier.kavier_power_predictor import KavierPowerPredictor

# A GPU and model calibrated in Kavier's library.
SUPPORTED_GPU = "NVIDIA-A100-80GB-PCIe"
SUPPORTED_MODEL = "mistral-7b-v0.1"

# NVIDIA A100 80GB PCIe datasheet: TDP 300 W, idle draw about 60 W. Per-GPU power under training
# load must lie in this range.
A100_PCIE_TDP_W = 300.0
A100_PCIE_IDLE_W = 60.0


@pytest.fixture
def context():
    """System context with a Kavier-supported A100 GPU."""
    return SystemContext(
        available_gpu_models=[SUPPORTED_GPU],
        max_gpus=32,
        gpu_memory={SUPPORTED_GPU: 80},
        constraints=Constraints(max_gpus=32, gpus_per_node=8, max_nodes=4),
    )


def _workload(model=SUPPORTED_MODEL, gpu=SUPPORTED_GPU, gpus_per_node=1, number_of_nodes=1):
    """Build a LoRA WorkloadSpec. The tests here use one node, so ``gpus_per_node`` is the total."""
    return WorkloadSpec(
        llm_model=model,
        fine_tuning_method="lora",
        gpu_model=gpu,
        tokens_per_sample=2048,
        batch_size=16,
        gpus_per_node=gpus_per_node,
        number_of_nodes=number_of_nodes,
    )


class TestKavierPowerPredictor:
    @pytest.mark.parametrize(
        "model, gpu",
        [
            ("totally-not-a-real-model-xyz", SUPPORTED_GPU),  # unsupported model key
            (SUPPORTED_MODEL, "FAKE-GPU-9000"),  # unsupported GPU key
        ],
    )
    def test_out_of_library_config_gives_no_power_and_the_reason(self, context, model, gpu):
        """An unknown model or GPU gives no power and a reason in ``error_detail``.

        KavierPredictor returns an error Prediction with ``predicted_power=None`` and the power
        wrapper passes it on. Model and GPU lookups fail separately, so both are tested.
        """
        pred = KavierPowerPredictor().predict(_workload(model=model, gpu=gpu), context)
        assert pred is not None
        assert pred.predicted_power is None
        assert pred.metadata["error_detail"]

    def test_an_unknown_method_gives_no_power_and_kavier_reason(self, context):
        workload = _workload().model_copy(update={"fine_tuning_method": "lorra"})

        pred = KavierPowerPredictor().predict(workload, context)

        assert pred is not None and pred.predicted_power is None
        assert "unknown method 'lorra'" in pred.metadata["error_detail"]

    def test_per_gpu_power_is_invariant_to_gpu_count_and_within_hardware_envelope(self, context):
        """``predicted_power`` is per GPU: the same for 1, 2, 4 and 8 GPUs, and within the
        A100 80GB PCIe's [idle, TDP] range of [60, 300] W. ``total_gpus`` matches the request.
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
        """Total throughput rises with GPU count (1, 2, 4, 8) but less than linearly, because of
        communication cost: 2 GPUs give less than 2x and 8 GPUs less than 8x one GPU."""
        predictor = KavierPowerPredictor()
        thr = {n: predictor.predict(_workload(gpus_per_node=n), context).predicted_throughput for n in (1, 2, 4, 8)}

        assert thr[1] < thr[2] < thr[4] < thr[8], f"throughput must grow with GPUs, got {thr}"
        assert thr[8] < 8 * thr[1], f"8-GPU throughput {thr[8]} must be sub-linear vs 8 x single-GPU {8 * thr[1]}"
        assert thr[2] < 2 * thr[1], f"2-GPU throughput {thr[2]} must be sub-linear vs {2 * thr[1]}"
