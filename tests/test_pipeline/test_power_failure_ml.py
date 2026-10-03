"""The shipped xgboost model with a misspelled fine-tuning method.

xgboost still predicts a throughput for a method it never saw, while Kavier's power model
rejects the method, so the error names the power predictor and its reason.

Marked ``ml_isolated`` (deselected by default) because it loads native xgboost. Run with:
``uv run --all-extras pytest -m ml_isolated -p no:cacheprovider``.
"""

import pytest

from coastline.sdk.exceptions import NoPredictionError
from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.policies import PolicyFactory

pytestmark = pytest.mark.ml_isolated

_GPU = "NVIDIA-A100-SXM4-80GB"


def test_xgboost_with_a_misspelled_method_names_the_power_predictor():
    config = {
        "strategy": {"name": "multi_objective", "preset": "balanced"},
        "predictors": {"performance": "xgboost", "energy": "kavier_power", "feasibility": "rules"},
        "grid": {"batch_sizes": [8], "total_gpus": [1, 2], "top_k": 2},
    }
    workload = WorkloadSpec(
        llm_model="mistral-7b-v0.1",
        fine_tuning_method="lorra",
        gpu_model=_GPU,
        tokens_per_sample=1024,
        batch_size=8,
    )

    with pytest.raises(NoPredictionError) as excinfo:
        PolicyFactory.create_strategy(config=config).recommend(workload, SystemContext.for_gpus([_GPU], max_gpus=8))

    message = str(excinfo.value)
    assert "the kavier_power predictor gave no usable prediction" in message
    assert "unknown method 'lorra'" in message
    assert "xgboost" not in message
