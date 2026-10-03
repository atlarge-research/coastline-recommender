"""A lookup CSV with a non-numeric throughput, runtime or config cell loads, minus that row.

Measured-runs exports carry cells such as 'OOM' or '' for runs that failed. The cache drops such
rows, as `coastline utils tune` does (pd.to_numeric(errors='coerce')).
"""

import logging

import pandas as pd
import pytest

from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.performance.retrieval.cache_predictor import RetrievalPredictor

_GPU = "NVIDIA-A100-SXM4-80GB"
_GOOD = dict(
    model_name="mistralai/Mistral-7B-v0.1",
    method="lora",
    gpu_model=_GPU,
    number_nodes=1,
    number_gpus=4,
    tokens_per_sample=2048,
    batch_size=8,
    dataset_tokens_per_second=10000.0,
    train_runtime=100.0,
)


def _workload(batch_size: int) -> WorkloadSpec:
    return WorkloadSpec(
        llm_model="mistral-7b-v0.1",
        fine_tuning_method="lora",
        gpu_model=_GPU,
        tokens_per_sample=2048,
        batch_size=batch_size,
        gpus_per_node=4,
        number_of_nodes=1,
    )


def _load(tmp_path, rows) -> RetrievalPredictor:
    csv = tmp_path / "lookup.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return RetrievalPredictor(dataset_path=csv)


def test_non_numeric_throughput_and_blank_runtime_drop_the_row(tmp_path, caplog):
    rows = [_GOOD, {**_GOOD, "batch_size": 16, "dataset_tokens_per_second": "OOM", "train_runtime": ""}]
    with caplog.at_level(logging.WARNING):
        predictor = _load(tmp_path, rows)

    assert len(predictor.config_index) == 1
    assert "Filtered out 1 rows with missing/zero throughput or runtime" in caplog.text
    context = SystemContext.for_gpus([_GPU], max_gpus=8)
    assert predictor.predict(_workload(8), context).predicted_throughput == pytest.approx(10000.0)
    assert predictor.predict(_workload(16), context) is None


def test_non_numeric_config_cell_drops_the_row(tmp_path, caplog):
    rows = [_GOOD, {**_GOOD, "batch_size": "auto"}, {**_GOOD, "number_gpus": "n/a", "batch_size": 32}]
    with caplog.at_level(logging.WARNING):
        predictor = _load(tmp_path, rows)

    assert len(predictor.config_index) == 1
    assert "Filtered out 2 rows with a missing or non-numeric" in caplog.text
    hit = predictor.predict(_workload(8), SystemContext.for_gpus([_GPU], max_gpus=8))
    assert hit is not None and hit.predicted_runtime_seconds == pytest.approx(100.0)
