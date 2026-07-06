"""Kavier physics predictor: contract + invariant + scaling tests.

Kavier is an analytical (black-box) engine, so we do NOT pin its raw throughput
numbers. Instead every assertion rests on an oracle that is independent of the
engine internals:
  * physical envelopes (per-GPU power lies between 0 and the GPU's published TDP),
  * derived layout invariants (total_gpus = gpus_per_node x number_of_nodes),
  * the predictor's documented CONTRACT (error Prediction for unsupported configs,
    step-time-only runtime semantics),
  * scaling laws (more GPUs raise throughput, but strictly sub-linearly).
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
    """Arrange helper: a Kavier-supported LoRA workload with optional overrides."""
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


# Published thermal design power (watts) per GPU SKU — an analytic reference point
# independent of Kavier. Per-GPU draw under any workload must sit inside (0, TDP].
# (model, gpu, gpu_tdp_watts) spans Kavier's supported model x GPU catalog so the
# parametrization varies BEHAVIOR, not just a number.
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
    """A supported (model, GPU, LoRA) config -> finite throughput and power in (0, TDP]."""
    from coastline.sdk.predictors.performance.physics.kavier_predictor import KavierPredictor

    context = SystemContext(
        available_gpu_models=[gpu],
        max_gpus=32,
        gpu_memory={gpu: 80},
        constraints=Constraints(max_gpus=32, gpus_per_node=8, max_nodes=4),
    )
    workload = _workload(llm_model=model, gpu_model=gpu)

    prediction = predictor_predict(KavierPredictor(), workload, context)

    # Throughput must be a finite positive number (a bug returning 0/NaN/inf fails).
    thr = prediction.predicted_throughput
    assert thr is not None and thr > 0 and thr < float("inf"), f"throughput not finite-positive: {thr}"
    # Per-GPU power envelope: strictly above zero, at or below the published TDP.
    # Catches a bug that reports TOTAL power (8x per-GPU >> TDP) or garbage.
    power = prediction.predicted_power
    assert power is not None and 0 < power <= gpu_tdp_watts, (
        f"per-GPU power {power}W outside (0, {gpu_tdp_watts}]W envelope for {gpu}"
    )
    # gpus_per_node=8 x number_of_nodes=1 -> total_gpus=8 (derived by hand).
    assert prediction.total_gpus == 8
    assert prediction.metadata["predictor"] == "kavier"
    assert prediction.metadata["model_used"] == "physics_based"


def test_kavier_node_layout_propagates_to_prediction(test_context):
    """Multi-node request: total_gpus = gpus_per_node x number_of_nodes = 8 x 2 = 16."""
    from coastline.sdk.predictors.performance.physics.kavier_predictor import KavierPredictor

    workload = _workload(gpus_per_node=8, number_of_nodes=2)

    prediction = predictor_predict(KavierPredictor(), workload, test_context)

    # Hand-derived layout: 8 GPUs/node x 2 nodes = 16 total. The Prediction model
    # itself enforces total == gpus_per_node * number_of_nodes, so a mis-propagated
    # node count would fail construction or these equalities.
    assert prediction.gpus_per_node == 8
    assert prediction.number_of_nodes == 2
    assert prediction.total_gpus == 16
    # A supported config must still yield usable throughput at multi-node.
    assert prediction.predicted_throughput > 0


def test_kavier_runtime_is_step_time_only_not_job_runtime(test_context):
    """Contract: Kavier reports per-step timing, so job runtime is deliberately None."""
    from coastline.sdk.predictors.performance.physics.kavier_predictor import KavierPredictor

    prediction = predictor_predict(KavierPredictor(), _workload(), test_context)

    # Documented contract (docstring: "predicted_runtime_seconds is always None").
    assert prediction.predicted_runtime_seconds is None
    assert prediction.metadata["runtime_semantics"] == "step_time_only"


# (bad_field, bad_value) each drives a DIFFERENT rejection axis of Kavier's library:
# an unknown model vs an unknown GPU. Both must surface the same error CONTRACT.
@pytest.mark.parametrize(
    "bad_field,bad_value",
    [
        ("llm_model", "unsupported-model-xyz-12345"),
        ("gpu_model", "FAKE-GPU-999"),
    ],
)
def test_kavier_unsupported_config_returns_error_prediction(test_context, bad_field, bad_value):
    """Unsupported model/GPU -> a null Prediction carrying unsupported_config metadata."""
    from coastline.sdk.predictors.performance.physics.kavier_predictor import KavierPredictor

    # WorkloadSpec canonicalizes llm_model (lowercase, drop org prefix); the chosen
    # bad values are already canonical so they reach Kavier verbatim.
    workload = _workload(**{bad_field: bad_value})

    prediction = predictor_predict(KavierPredictor(), workload, test_context)

    # Contract: a Prediction object (never None) with no numeric outputs and an
    # error tag. A bug that fabricated a throughput here would flip the first assert.
    assert prediction is not None
    assert prediction.predicted_throughput is None
    assert prediction.predicted_power is None
    assert prediction.metadata["error"] == "unsupported_config"
    # The offending identifier must be echoed so callers can see WHAT was rejected.
    assert bad_value in prediction.metadata["error_detail"]
    # Layout metadata is still preserved on the error path: 8 x 1 = 8.
    assert prediction.total_gpus == 8


def test_kavier_throughput_scales_sublinearly_with_gpu_count(test_context):
    """More GPUs raise throughput, but strictly sub-linearly (communication overhead).

    Independent oracle = the parallel-scaling law, no magic engine numbers:
      * monotonic: thr(1) < thr(4) < thr(8) (adding GPUs never hurts here),
      * doubling GPUs 4->8 yields < 2x throughput (ideal linear = exactly 2x),
      * 8 GPUs yield < 8x a single GPU (sub-linear across the whole range).
    """
    from coastline.sdk.predictors.performance.physics.kavier_predictor import KavierPredictor

    predictor = KavierPredictor()
    thr_1 = predictor_predict(predictor, _workload(gpus_per_node=1), test_context).predicted_throughput
    thr_4 = predictor_predict(predictor, _workload(gpus_per_node=4), test_context).predicted_throughput
    thr_8 = predictor_predict(predictor, _workload(gpus_per_node=8), test_context).predicted_throughput

    # Monotone in GPU count.
    assert thr_1 < thr_4 < thr_8, f"throughput not monotone: {thr_1} {thr_4} {thr_8}"
    # Sub-linear: doubling 4->8 GPUs must yield strictly LESS than 2x (would be
    # exactly 2x under perfect scaling; >=2x would be non-physical super-linear).
    assert thr_8 / thr_4 < 2.0, f"4->8 speedup {thr_8 / thr_4:.3f} not sub-linear"
    # Still meaningfully parallel (not a token gain): well above 1.5x for 2x GPUs.
    assert thr_8 / thr_4 > 1.5, f"4->8 speedup {thr_8 / thr_4:.3f} unreasonably low"
    # Sub-linear across the full 1->8 range: 8 GPUs < 8x one GPU.
    assert thr_8 < 8 * thr_1, f"8-GPU throughput {thr_8} >= 8x single-GPU {thr_1} (super-linear)"


def predictor_predict(predictor, workload, context):
    """Act helper: run predict and assert a Prediction came back (Kavier supports these)."""
    prediction = predictor.predict(workload, context)
    assert prediction is not None, "Kavier returned None for a supported config"
    return prediction


def test_kavier_ground_truth_validation():
    """Cross-check Kavier against real experimental measurements (external oracle).

    The oracle here is genuinely independent: recorded tokens/sec from profiling
    runs. Kavier's paper claims <20% median error; we hold it to that. Skips
    cleanly when the profiling trace is absent from the checkout.
    """
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

    # Kavier paper: median error < 20%; mean allowed a wider 30% (tail-sensitive).
    assert median_error < 0.20, f"Kavier median error {median_error:.1%} exceeds 20% threshold"
    assert mean_error < 0.30, f"Kavier mean error {mean_error:.1%} exceeds 30% threshold"
