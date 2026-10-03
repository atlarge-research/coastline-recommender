"""Tests for the data-driven ML performance predictors, wrappers over pickled featv3 models.

The tests check only what does not depend on the model weights:
  * the GPU layout, which ``finalize_ml_prediction`` copies from the WorkloadSpec
    (8 x 2 = 16 GPUs for ``known_workload``);
  * that each of the 7 models writes its own name to ``metadata["predictor"]``;
  * finite, non-negative throughput for an in-library workload, None for an out-of-library
    model, and a non-negative Gaussian-process std that appears only when requested.

Predictors are built per test with ``_build_named_ml_predictor``. The six sklearn-portfolio
models share one class and keep distinct names. random_forest is stored in Git LFS, so its cases
are skipped in a checkout that holds the pointer file (see ``lfs_model`` in conftest.py).
"""

import math

import pytest

from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.policies import _build_named_ml_predictor

# Native ML backends (xgboost, lightgbm, catboost, torch) can crash when loaded together in one
# interpreter, so this module is left out of the default run. Run it in its own process:
# `uv run pytest -m ml_isolated -p no:cacheprovider tests/test_predictors/test_ml_predictors.py`.
pytestmark = pytest.mark.ml_isolated


def _build(name: str):
    """Build a predictor by canonical name with the production resolver."""
    return _build_named_ml_predictor(name)


# (predictor name, one metadata key it emits)
_MODELS = [
    ("catboost", "iterations"),
    ("xgboost", "n_estimators"),
    ("lightgbm", "num_leaves"),
    ("random_forest", "oob_score"),
    ("svr", "kernel"),
    ("knn", "n_neighbors"),
    ("gaussian_process", "kernel"),
]
_IDS = [m[0] for m in _MODELS]
# Model files stored in Git LFS (.gitattributes).
_LFS_MODELS = {"random_forest"}


def _case(name: str, *values: str):
    """One parametrize case, skipped where the model file is an LFS pointer."""
    marks = [pytest.mark.lfs_model(name)] if name in _LFS_MODELS else []
    return pytest.param(name, *values, marks=marks, id=name)


_OUT_OF_LIBRARY = WorkloadSpec(
    llm_model="totally-unknown-xyz-model",
    fine_tuning_method="lora",
    gpu_model="NVIDIA-A100-SXM4-80GB",
    tokens_per_sample=1024,
    batch_size=32,
    gpus_per_node=8,
    number_of_nodes=1,
)


@pytest.mark.parametrize("name, meta_key", [_case(*model) for model in _MODELS])
def test_predicts_for_known_workload_with_derivable_gpu_layout(name, meta_key, a100_context, known_workload):
    """An in-library workload gives a finite positive throughput and runtime, the predictor's own
    metadata tag and hyperparameter key, and the GPU layout of the WorkloadSpec."""
    p = _build(name).predict(known_workload, a100_context)
    assert p is not None, f"{name} should predict for a known workload"

    # Each of the 7 models writes its own name.
    assert p.metadata.get("predictor") == name
    assert meta_key in p.metadata, f"{name} must expose its own hyperparameter '{meta_key}'"

    # finalize_ml_prediction drops non-finite throughput and clamps negatives to 0; an in-library
    # workload should be strictly positive.
    assert math.isfinite(p.predicted_throughput) and p.predicted_throughput > 0
    assert p.predicted_runtime_seconds is not None
    assert math.isfinite(p.predicted_runtime_seconds) and p.predicted_runtime_seconds > 0

    # known_workload: 8 GPUs per node x 2 nodes = 16, copied from the WorkloadSpec.
    assert p.gpus_per_node == 8
    assert p.number_of_nodes == 2
    assert p.total_gpus == 16


@pytest.mark.parametrize("name", [_case(name) for name in [*_IDS, "deep_learning"]])
def test_returns_none_for_out_of_library_model(name, a100_context):
    """An out-of-library model has NaN spec features at inference (training filled them with the
    median), so every predictor returns None."""
    assert _build(name).predict(_OUT_OF_LIBRARY, a100_context) is None


def test_gaussian_process_std_is_nonnegative_and_only_present_when_requested(a100_context, known_workload):
    """``return_std`` adds the uncertainty metadata: absent by default, and when requested a
    non-negative std plus the correlation."""
    # By default there is no std in the metadata.
    p_plain = _build("gaussian_process").predict(known_workload, a100_context)
    assert p_plain is not None
    assert "std" not in p_plain.metadata
    assert "uncertainty_correlation" not in p_plain.metadata

    # With return_std=True, std is present and non-negative, plus the correlation.
    p_std = _build("gaussian_process").predict(known_workload, a100_context, return_std=True)
    assert p_std is not None
    assert "std" in p_std.metadata
    assert p_std.metadata["std"] >= 0.0
    assert "uncertainty_correlation" in p_std.metadata


def test_bayesian_ridge_prediction_is_wellformed_even_when_extrapolating(a100_context, known_workload):
    """bayesian_ridge, a weak degree-2 polynomial, can extrapolate to non-physical values. It
    returns None (non-finite output) or a finite, non-negative Prediction tagged
    'bayesian_ridge' with 16 total GPUs (8 x 2)."""
    p = _build("bayesian_ridge").predict(known_workload, a100_context)
    if p is None:
        return  # finalize_ml_prediction drops a non-finite output
    assert p.metadata.get("predictor") == "bayesian_ridge"
    assert math.isfinite(p.predicted_throughput) and p.predicted_throughput >= 0.0
    assert p.total_gpus == 16  # 8 gpus_per_node * 2 nodes
