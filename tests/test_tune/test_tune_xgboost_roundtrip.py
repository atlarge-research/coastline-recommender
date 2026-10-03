"""`coastline utils tune --model xgboost` round trip: the predictor that `--method xgboost` builds
(SklearnPortfolioPredictor) serves the tuned file.

Marked ``ml_isolated`` (left out of the default run) because native xgboost bundles libomp and
can crash when loaded with other ML backends in one interpreter. Run with:
``uv run --all-extras pytest -m ml_isolated -p no:cacheprovider tests/test_tune``.

The test checks properties that do not depend on the learned values: a known-library workload
gives a finite, positive throughput, and the prediction has both throughput and runtime.
"""

import math
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.ml_isolated

# Models and GPU in Kavier's library, so the spec features resolve; GPU count and batch vary so
# the model has something to learn.
_MODELS = ["granite-3.1-8b-instruct", "granite-3.1-2b"]
_ROWS = [
    {
        "model_name": m,
        "method": "lora",
        "gpu_model": "NVIDIA-A100-SXM4-80GB",
        "number_nodes": 1,
        "number_gpus": g,
        "tokens_per_sample": tok,
        "batch_size": b,
        "dataset_tokens_per_second": 1000.0 + 100 * g + 10 * b,
        "train_runtime": 500.0 + 50 * b,
        "is_valid": 1.0,
    }
    for m in _MODELS
    for g in (1, 2, 4)
    for tok, b in ((1024, 4), (2048, 8))
]

_WORKLOAD = dict(
    llm_model="granite-3.1-8b-instruct",
    fine_tuning_method="lora",
    gpu_model="NVIDIA-A100-SXM4-80GB",
    tokens_per_sample=2048,
    batch_size=8,
    gpus_per_node=2,
    number_of_nodes=1,
)


def test_tuned_xgboost_is_served_by_the_xgboost_predictor(tmp_path, monkeypatch):
    """With no --output, tune writes PORTFOLIO_DIR/custom/xgboost.pkl (creating custom/, which a
    checkout does not have), and the predictor built by name serves that file."""
    import coastline.sdk.predictors.performance.data_driven.ml_common as ml_common
    from coastline.sdk.models.context import SystemContext
    from coastline.sdk.models.workload import WorkloadSpec
    from coastline.sdk.policies import _build_named_ml_predictor
    from coastline.sdk.predictors.performance.data_driven.tune import tune

    portfolio = tmp_path / "portfolio"
    monkeypatch.setattr(ml_common, "PORTFOLIO_DIR", portfolio)
    data_csv = tmp_path / "runs.csv"
    pd.DataFrame(_ROWS).to_csv(data_csv, index=False)

    result = tune(str(data_csv), model="xgboost", train_percentage=1.0)

    out = portfolio / "custom" / "xgboost.pkl"
    assert Path(result["path"]) == out and out.exists()

    predictor = _build_named_ml_predictor("xgboost")
    assert predictor._model_path == out  # the tuned file shadows the packaged one
    pred = predictor.predict(WorkloadSpec(**_WORKLOAD), SystemContext.for_gpus([_WORKLOAD["gpu_model"]], max_gpus=8))
    assert pred is not None
    assert pred.predicted_throughput > 0 and math.isfinite(pred.predicted_throughput)
    assert pred.metadata["predictor"] == "xgboost"
    assert pred.metadata["dual_output"] is True
