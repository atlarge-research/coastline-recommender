"""Unit tests for the energy predictor.

Covers the energy predictor under ``recommender/predictors/energy/``:

* ``KavierPowerPredictor`` (``kavier_power_predictor.py``) — a thin wrapper that
  delegates to the analytical Kavier physics predictor and surfaces GPU power.

Design notes
------------
* No ML model artifacts are loaded. Kavier is purely analytical, so its tests
  run end-to-end and are deterministic.

IMPORTANT: this file is test-only and must not import or mutate prod modules
beyond reading their public behaviour.
"""

import pytest

from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.energy.kavier.kavier_power_predictor import KavierPowerPredictor

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
