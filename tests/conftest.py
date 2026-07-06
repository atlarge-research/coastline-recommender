"""Shared fixtures + test bootstrap.

Set the OpenMP-duplicate workaround before any native ML backend
(torch/xgboost/lightgbm/catboost) loads, so collecting tests that import several of
them in one interpreter doesn't crash. The data-driven ML predictor tests are still
best run in their own process — see the ``ml_isolated`` marker in pyproject.toml.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.workload import WorkloadSpec


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
    """Workload that does NOT exist in the curated dataset (should cache-miss)."""
    return WorkloadSpec(
        llm_model="nonexistent-model",
        fine_tuning_method="full",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=9999,
        batch_size=999,
    )
