"""Contract tests for the data-driven ML performance predictors.

These are thin wrappers over pickled featv3 models. The learned throughput/runtime
NUMBERS are black-box, so we never pin a specific model output. Instead every
assertion rests on an oracle independent of the model weights:

  * hand-derived arithmetic  — the Prediction's GPU layout is total_gpus =
    gpus_per_node * number_of_nodes = 8 * 2 = 16 for `known_workload`, computed by
    `finalize_ml_prediction` from the WorkloadSpec, NOT from the model.
  * a pinned-bug cross-check — each predictor must stamp its OWN name into
    metadata["predictor"]. An earlier duplicate resolver silently collapsed every
    named model to CatBoost (see PolicyFactory note in CLAUDE.md); tagging each of
    the 7 models in the catalog re-catches that regression.
  * an invariant/contract — an in-library workload yields finite, non-negative
    throughput; an out-of-library model yields None (not garbage); a Gaussian-Process
    std is non-negative and only appears when explicitly requested.

Predictors are imported lazily per test so each can run in its own process — the
native ML backends (xgboost/lightgbm/catboost/torch) can crash if many load in
one interpreter (see the KMP_DUPLICATE_LIB_OK note in the dev docs).
"""

import importlib
import math

import pytest

from coastline.sdk.models.workload import WorkloadSpec

# Native ML backends (xgboost/lightgbm/catboost/torch) can crash when co-loaded in one
# interpreter, so this module is deselected from the default `pytest` run and executed in
# its own process: `uv run pytest -m ml_isolated -p no:cacheprovider`.
pytestmark = pytest.mark.ml_isolated


def _predictor(module: str, cls: str):
    mod = importlib.import_module(f"coastline.sdk.predictors.performance.data_driven.{module}")
    return getattr(mod, cls)


# (module, class, predictor-name, one metadata key it must emit)
_MODELS = [
    ("catboost_predictor", "CatBoostPredictor", "catboost", "iterations"),
    ("xgboost_predictor", "XGBoostPredictor", "xgboost", "n_estimators"),
    ("lightgbm_predictor", "LightGBMPredictor", "lightgbm", "num_leaves"),
    ("random_forest_predictor", "RandomForestPredictor", "random_forest", "oob_score"),
    ("svr_predictor", "SVRPredictor", "svr", "kernel"),
    ("knn_predictor", "KNNPredictor", "knn", "n_neighbors"),
    ("gaussian_process_predictor", "GaussianProcessPredictor", "gaussian_process", "kernel"),
]
_IDS = [m[2] for m in _MODELS]

_OUT_OF_LIBRARY = WorkloadSpec(
    llm_model="totally-unknown-xyz-model",
    fine_tuning_method="lora",
    gpu_model="NVIDIA-A100-SXM4-80GB",
    tokens_per_sample=1024,
    batch_size=32,
    gpus_per_node=8,
    number_of_nodes=1,
)


@pytest.mark.parametrize("module, cls, name, meta_key", _MODELS, ids=_IDS)
def test_predicts_for_known_workload_with_derivable_gpu_layout(
    module, cls, name, meta_key, a100_context, known_workload
):
    """In-library workload -> finite non-negative throughput, the predictor's OWN
    metadata tag (guards the 'every model collapses to catboost' bug), and a GPU
    layout hand-derivable from the WorkloadSpec independent of the model weights."""
    p = _predictor(module, cls)().predict(known_workload, a100_context)
    assert p is not None, f"{name} should predict for a known workload"

    # Pinned-bug cross-check: each of the 7 models must stamp its own name, not a
    # single shared/collapsed value.
    assert p.metadata.get("predictor") == name
    assert meta_key in p.metadata, f"{name} must expose its own hyperparameter '{meta_key}'"

    # Invariant: finalize_ml_prediction returns None on non-finite throughput and
    # clamps negatives to 0, so any returned Prediction is finite and >= 0; an
    # in-library workload should land strictly positive.
    assert math.isfinite(p.predicted_throughput) and p.predicted_throughput > 0
    assert p.predicted_runtime_seconds is not None
    assert math.isfinite(p.predicted_runtime_seconds) and p.predicted_runtime_seconds > 0

    # Hand-derived, model-independent: known_workload has gpus_per_node=8,
    # number_of_nodes=2 => total_gpus = 8 * 2 = 16. finalize_ml_prediction copies
    # these off the WorkloadSpec; a bug that defaulted a field to 1 would break it.
    assert p.gpus_per_node == 8
    assert p.number_of_nodes == 2
    assert p.total_gpus == 16


@pytest.mark.parametrize(
    "module, cls",
    [(m[0], m[1]) for m in _MODELS] + [("deep_learning_predictor", "DeepLearningPredictor")],
    ids=_IDS + ["deep_learning"],
)
def test_returns_none_for_out_of_library_model(module, cls, a100_context):
    """An out-of-library model yields NaN deep-spec features at inference (training
    median-filled them), so every predictor must return None rather than a
    garbage/NaN throughput. Contract, not smoke: a body that returned any Prediction
    (or a clamped-to-0 one) would fail this."""
    assert _predictor(module, cls)().predict(_OUT_OF_LIBRARY, a100_context) is None


def test_gaussian_process_std_is_nonnegative_and_only_present_when_requested(a100_context, known_workload):
    """return_std toggles the uncertainty metadata: absent by default, and when
    requested a non-negative std (std is a standard deviation) plus the correlation."""
    cls = _predictor("gaussian_process_predictor", "GaussianProcessPredictor")

    # Default: no uncertainty requested -> no std leaked into metadata.
    p_plain = cls().predict(known_workload, a100_context)
    assert p_plain is not None
    assert "std" not in p_plain.metadata
    assert "uncertainty_correlation" not in p_plain.metadata

    # return_std=True -> std present and non-negative (invariant), plus correlation.
    p_std = cls().predict(known_workload, a100_context, return_std=True)
    assert p_std is not None
    assert "std" in p_std.metadata
    assert p_std.metadata["std"] >= 0.0
    assert "uncertainty_correlation" in p_std.metadata


def test_bayesian_ridge_prediction_is_wellformed_even_when_extrapolating(a100_context, known_workload):
    """bayesian_ridge is a known-weak degree-2 polynomial that can extrapolate to
    non-physical values, so we don't pin its number. The finalize contract must still
    hold: it either returns None (non-finite -> dropped) or a Prediction that is
    finite, clamped >= 0, tagged 'bayesian_ridge', with the hand-derivable layout
    (8 * 2 = 16 total GPUs). A wrong metadata tag or GPU arithmetic fails this."""
    p = _predictor("bayesian_ridge_predictor", "BayesianRidgePredictor")().predict(known_workload, a100_context)
    if p is None:
        return  # non-finite extrapolation legitimately dropped by finalize_ml_prediction
    assert p.metadata.get("predictor") == "bayesian_ridge"
    assert math.isfinite(p.predicted_throughput) and p.predicted_throughput >= 0.0
    assert p.total_gpus == 16  # 8 gpus_per_node * 2 nodes
