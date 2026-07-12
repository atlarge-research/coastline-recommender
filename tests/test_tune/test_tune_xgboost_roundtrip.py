"""`coastline tune --model xgboost` round-trip: the tuned artifact must be byte-compatible
with the XGBoostPredictor that serves `--method xgboost`.

Marked ``ml_isolated`` (deselected by default) because it loads native xgboost, which bundles
libomp and can crash when co-loaded with other ML backends in one interpreter — run with:
``uv run --all-extras pytest -m ml_isolated -p no:cacheprovider tests/test_tune``.

The learned NUMBERS are black-box, so the oracle is an invariant: a known-library workload yields
a finite, positive throughput and a dual (throughput+runtime) output, and the artifact carries the
exact keys the predictor reads (model / encoders / cat_features / num_features).
"""

from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.ml_isolated

# Models + GPU known to Kavier's library (so feature specs resolve); GPU/batch variety so the
# gradient-boosted model has something to learn.
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


def test_tuned_xgboost_is_served_by_the_xgboost_predictor(tmp_path):
    from coastline.sdk.models.context import SystemContext
    from coastline.sdk.models.workload import WorkloadSpec
    from coastline.sdk.predictors.performance.data_driven.tune import tune
    from coastline.sdk.predictors.performance.data_driven.xgboost_predictor import XGBoostPredictor

    data_csv = tmp_path / "runs.csv"
    pd.DataFrame(_ROWS).to_csv(data_csv, index=False)
    out = tmp_path / "xgboost.pkl"

    result = tune(str(data_csv), model="xgboost", train_percentage=1.0, output=str(out))
    assert Path(result["path"]) == out and out.exists()

    # The predictor loads the tuned artifact and produces a dual (throughput + runtime) prediction.
    pred = XGBoostPredictor(model_path=out).predict(
        WorkloadSpec(
            llm_model="granite-3.1-8b-instruct",
            fine_tuning_method="lora",
            gpu_model="NVIDIA-A100-SXM4-80GB",
            tokens_per_sample=2048,
            batch_size=8,
            gpus_per_node=2,
            number_of_nodes=1,
        ),
        SystemContext.for_gpus(["NVIDIA-A100-SXM4-80GB"], max_gpus=8),
    )
    assert pred is not None
    assert pred.predicted_throughput > 0 and pred.predicted_throughput == pred.predicted_throughput  # finite
    assert pred.metadata["predictor"] == "xgboost"
    assert pred.metadata["dual_output"] is True
