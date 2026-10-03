"""Tests for the Kavier physics predictor.

The tests check properties that do not depend on Kavier's exact numbers: per-GPU power against the
published TDP, total_gpus = gpus_per_node x number_of_nodes, the error Prediction for unsupported
configs, the step-time-only runtime, and sub-linear throughput growth with GPU count.
"""

import pytest

from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.workload import WorkloadSpec


@pytest.fixture
def test_context():
    """Standard test context (A100-80GB, up to 4 nodes x 8 GPUs)."""
    return SystemContext(
        available_gpu_models=["NVIDIA-A100-80GB-PCIe"],
        max_gpus=32,
        gpu_memory={"NVIDIA-A100-80GB-PCIe": 80},
        constraints=Constraints(max_gpus=32, gpus_per_node=8, max_nodes=4),
    )


def _workload(**overrides):
    """A Kavier-supported LoRA workload with optional overrides."""
    base = dict(
        llm_model="mistral-7b-v0.1",
        fine_tuning_method="lora",
        gpu_model="NVIDIA-A100-80GB-PCIe",
        tokens_per_sample=2048,
        batch_size=16,
        gpus_per_node=8,
        number_of_nodes=1,
    )
    base.update(overrides)
    return WorkloadSpec(**base)


# Published thermal design power (watts) per GPU. Per-GPU draw must lie in (0, TDP].
# The (model, gpu, gpu_tdp_watts) rows cover several supported models and GPUs.
_SUPPORTED_CATALOG = [
    ("mistral-7b-v0.1", "NVIDIA-A100-80GB-PCIe", 300),
    ("mistral-7b-v0.1", "NVIDIA-A100-SXM4-80GB", 400),
    ("mistral-7b-v0.1", "NVIDIA-H100-PCIe", 350),
    ("granite-3-8b", "NVIDIA-A100-80GB-PCIe", 300),
    ("granite-3.3-8b", "NVIDIA-H100-PCIe", 350),
    ("llama3.2-3b", "L40S", 350),
]


@pytest.mark.parametrize("model,gpu,gpu_tdp_watts", _SUPPORTED_CATALOG)
def test_kavier_supported_config_yields_physically_valid_prediction(model, gpu, gpu_tdp_watts):
    """A supported (model, GPU, LoRA) config gives a finite throughput and power in (0, TDP]."""
    from coastline.sdk.predictors.performance.physics.kavier_predictor import KavierPredictor

    context = SystemContext(
        available_gpu_models=[gpu],
        max_gpus=32,
        gpu_memory={gpu: 80},
        constraints=Constraints(max_gpus=32, gpus_per_node=8, max_nodes=4),
    )
    workload = _workload(llm_model=model, gpu_model=gpu)

    prediction = predictor_predict(KavierPredictor(), workload, context)

    # Throughput is finite and positive.
    thr = prediction.predicted_throughput
    assert thr is not None and thr > 0 and thr < float("inf"), f"throughput not finite-positive: {thr}"
    # Per-GPU power is above zero and at most the TDP; the total for 8 GPUs would exceed it.
    power = prediction.predicted_power
    assert power is not None and 0 < power <= gpu_tdp_watts, (
        f"per-GPU power {power}W outside (0, {gpu_tdp_watts}]W envelope for {gpu}"
    )
    # 8 GPUs per node x 1 node = 8.
    assert prediction.total_gpus == 8
    assert prediction.metadata["predictor"] == "kavier"
    assert prediction.metadata["model_used"] == "physics_based"


def test_kavier_node_layout_propagates_to_prediction(test_context):
    """Multi-node request: total_gpus = gpus_per_node x number_of_nodes = 8 x 2 = 16."""
    from coastline.sdk.predictors.performance.physics.kavier_predictor import KavierPredictor

    workload = _workload(gpus_per_node=8, number_of_nodes=2)

    prediction = predictor_predict(KavierPredictor(), workload, test_context)

    # The Prediction model also checks total == gpus_per_node x number_of_nodes when it is built.
    assert prediction.gpus_per_node == 8
    assert prediction.number_of_nodes == 2
    assert prediction.total_gpus == 16
    # A supported multi-node config gives a positive throughput.
    assert prediction.predicted_throughput > 0


def test_kavier_runtime_is_step_time_only_not_job_runtime(test_context):
    """Kavier reports per-step timing, so the job runtime is None."""
    from coastline.sdk.predictors.performance.physics.kavier_predictor import KavierPredictor

    prediction = predictor_predict(KavierPredictor(), _workload(), test_context)

    # KavierPredictor documents predicted_runtime_seconds as always None.
    assert prediction.predicted_runtime_seconds is None
    assert prediction.metadata["runtime_semantics"] == "step_time_only"


# Kavier rejects an unknown model and an unknown GPU separately; both give the same error.
@pytest.mark.parametrize(
    "bad_field,bad_value",
    [
        ("llm_model", "unsupported-model-xyz-12345"),
        ("gpu_model", "FAKE-GPU-999"),
    ],
)
def test_kavier_unsupported_config_returns_error_prediction(test_context, bad_field, bad_value):
    """An unsupported model or GPU gives a Prediction with no numbers and unsupported_config
    metadata."""
    from coastline.sdk.predictors.performance.physics.kavier_predictor import KavierPredictor

    # WorkloadSpec canonicalizes llm_model (lowercase, no org prefix); these values are already
    # canonical, so Kavier gets them unchanged.
    workload = _workload(**{bad_field: bad_value})

    prediction = predictor_predict(KavierPredictor(), workload, test_context)

    # A Prediction with no numeric outputs and an error tag.
    assert prediction is not None
    assert prediction.predicted_throughput is None
    assert prediction.predicted_power is None
    assert prediction.metadata["error"] == "unsupported_config"
    # The rejected name is in the error detail.
    assert bad_value in prediction.metadata["error_detail"]
    # The layout is kept on the error path: 8 x 1 = 8.
    assert prediction.total_gpus == 8


def test_kavier_throughput_scales_sublinearly_with_gpu_count(test_context):
    """More GPUs raise throughput sub-linearly (communication overhead): throughput rises from 1 to
    4 to 8 GPUs, going from 4 to 8 gives less than 2x, and 8 GPUs give less than 8x one GPU."""
    from coastline.sdk.predictors.performance.physics.kavier_predictor import KavierPredictor

    predictor = KavierPredictor()
    thr_1 = predictor_predict(predictor, _workload(gpus_per_node=1), test_context).predicted_throughput
    thr_4 = predictor_predict(predictor, _workload(gpus_per_node=4), test_context).predicted_throughput
    thr_8 = predictor_predict(predictor, _workload(gpus_per_node=8), test_context).predicted_throughput

    # Monotone in GPU count.
    assert thr_1 < thr_4 < thr_8, f"throughput not monotone: {thr_1} {thr_4} {thr_8}"
    # Doubling from 4 to 8 GPUs gives less than 2x (perfect scaling would give 2x).
    assert thr_8 / thr_4 < 2.0, f"4->8 speedup {thr_8 / thr_4:.3f} not sub-linear"
    # It still gives more than 1.5x.
    assert thr_8 / thr_4 > 1.5, f"4->8 speedup {thr_8 / thr_4:.3f} unreasonably low"
    # 8 GPUs give less than 8x one GPU.
    assert thr_8 < 8 * thr_1, f"8-GPU throughput {thr_8} >= 8x single-GPU {thr_1} (super-linear)"


def predictor_predict(predictor, workload, context):
    """Run predict and assert a Prediction came back (Kavier supports these configs)."""
    prediction = predictor.predict(workload, context)
    assert prediction is not None, "Kavier returned None for a supported config"
    return prediction


def test_kavier_ground_truth_validation():
    """Compares Kavier with measured tokens/s from profiling runs: median error under 20% (the
    Kavier paper's figure) and mean error under 30%. Skips when the profiling trace is absent."""
    import pandas as pd

    from coastline.sdk.predictors.performance.physics.kavier_predictor import KavierPredictor

    data_path = "trace-archive/profiling-dataset/raw_trace.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        pytest.skip(f"Ground truth data not found: {data_path}")

    test_cases = df[
        (df["model_name"] == "mistral-7b-v0.1")
        & (df["method"] == "lora")
        & (df["gpu_model"] == "NVIDIA-A100-80GB-PCIe")
        & (df["dataset_tokens_per_second"] > 0)
        & (df["train_runtime"] > 0)
    ].head(10)

    if len(test_cases) == 0:
        pytest.skip("No matching ground truth data for Kavier validation")

    predictor = KavierPredictor()
    errors = []

    for _, row in test_cases.iterrows():
        workload = WorkloadSpec(
            llm_model=row["model_name"],
            fine_tuning_method=row["method"],
            gpu_model=row["gpu_model"],
            tokens_per_sample=int(row["tokens_per_sample"]),
            batch_size=int(row["batch_size"]),
            gpus_per_node=int(row["number_gpus"]),
            number_of_nodes=int(row["number_nodes"]),
        )

        context = SystemContext(
            available_gpu_models=["NVIDIA-A100-80GB-PCIe"],
            max_gpus=32,
            gpu_memory={"NVIDIA-A100-80GB-PCIe": 80},
            constraints=Constraints(max_gpus=32, gpus_per_node=8, max_nodes=4),
        )

        prediction = predictor.predict(workload, context)

        if prediction is not None and prediction.predicted_throughput is not None:
            throughput_col = "dataset_tokens_per_second"
            if throughput_col not in row.index:
                throughput_col = "train_tokens_per_second"
            actual_throughput = row[throughput_col]
            predicted_throughput = prediction.predicted_throughput
            error = abs(predicted_throughput - actual_throughput) / actual_throughput
            errors.append(error)

    assert len(errors) > 0, "no predictions produced for ground-truth rows"
    median_error = sorted(errors)[len(errors) // 2]
    mean_error = sum(errors) / len(errors)

    # The Kavier paper reports a median error under 20%; the mean is more sensitive to the tail.
    assert median_error < 0.20, f"Kavier median error {median_error:.1%} exceeds 20% threshold"
    assert mean_error < 0.30, f"Kavier mean error {mean_error:.1%} exceeds 30% threshold"
