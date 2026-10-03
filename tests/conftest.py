"""Shared fixtures and test setup.

KMP_DUPLICATE_LIB_OK is set before any native ML backend (torch, xgboost, lightgbm, catboost)
loads, so collecting tests that import several of them in one interpreter does not crash. The
data-driven ML predictor tests still run best in their own process (see the ``ml_isolated``
marker in pyproject.toml).

A test that loads a model file stored in Git LFS carries ``@pytest.mark.lfs_model("<stem>")``
and is skipped when the checkout holds the LFS pointer file in its place (CI checks out without
LFS).
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.workload import WorkloadSpec

_LFS_POINTER_PREFIX = b"version https://git-lfs"


def model_is_materialized(stem: str) -> bool:
    """True if the resolved model file ``<stem>.pkl`` has been pulled from Git LFS.

    A checkout without LFS holds a small ``version https://git-lfs...`` pointer file instead.
    Uses the production resolver, so a model in custom/ takes precedence over the packaged one.
    """
    from coastline.sdk.predictors.performance.data_driven.ml_common import performance_trained_model_path

    path = performance_trained_model_path(stem)
    if not path.is_file():
        return False
    with path.open("rb") as model_file:
        return not model_file.read(len(_LFS_POINTER_PREFIX)).startswith(_LFS_POINTER_PREFIX)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "lfs_model(stem): loads the Git LFS model file <stem>.pkl; skipped when it is a pointer file"
    )


def pytest_runtest_setup(item):
    for marker in item.iter_markers(name="lfs_model"):
        stem = marker.args[0]
        if not model_is_materialized(stem):
            pytest.skip(f"{stem}.pkl is missing or a Git LFS pointer file (run `git lfs pull`)")


@pytest.fixture
def a100_context():
    """Standard A100 system context for tests."""
    return SystemContext(
        available_gpu_models=["NVIDIA-A100-SXM4-80GB"],
        max_gpus=32,
        gpu_memory={"NVIDIA-A100-SXM4-80GB": 80},
        constraints=Constraints(
            max_gpus=32,
            gpus_per_node=8,
            max_nodes=4,
        ),
    )


@pytest.fixture
def known_workload():
    """Standard supported-model/GPU workload used across the predictor tests."""
    return WorkloadSpec(
        llm_model="mistral-7b-v0.1",
        fine_tuning_method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=32,
        gpus_per_node=8,
        number_of_nodes=2,
    )


@pytest.fixture
def unknown_workload():
    """Workload absent from the curated dataset, so a cache lookup misses."""
    return WorkloadSpec(
        llm_model="nonexistent-model",
        fine_tuning_method="full",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=9999,
        batch_size=999,
    )
