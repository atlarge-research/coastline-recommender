"""RetrievalPredictor: similarity scoring and the exact-match cache.

find_similar_configurations scores GPU proximity on total GPUs (number_of_nodes x
gpus_per_node from config_index), so a 32-GPU config (8 per node x 4 nodes) is a poor match for
an 8-GPU target.
"""

from __future__ import annotations

import pytest

from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.performance.retrieval.cache_predictor import RetrievalPredictor


def test_find_similar_scores_categorical_plus_total_gpu_proximity():
    """Score = 0.4 (llm) + 0.3 (method) + 0.2 (gpu) + 0.1 / (1 + total GPU difference fraction).

    Both configs match all three categoricals (0.9), so the GPU term decides. For an 8-GPU target:
      * total_match, 1 node x 8 = 8 GPUs: diff |8 - 8| / 8 = 0, score 0.9 + 0.1 / 1 = 1.000
      * per_node_only, 4 nodes x 8 = 32 GPUs: diff |32 - 8| / 8 = 3, score 0.9 + 0.1 / 4 = 0.925
    """
    predictor = RetrievalPredictor.__new__(RetrievalPredictor)  # bypass trace loading
    base = {"llm_model": "m", "fine_tuning_method": "lora", "gpu_model": "g"}
    predictor.config_index = {
        "total_match": {**base, "number_of_nodes": 1.0, "gpus_per_node": 8.0},  # total 8 == target
        "per_node_only": {**base, "number_of_nodes": 4.0, "gpus_per_node": 8.0},  # total 32
    }
    predictor.dataset = [1]  # non-empty so the method doesn't early-return

    workload = WorkloadSpec(
        llm_model="m",
        fine_tuning_method="lora",
        gpu_model="g",
        tokens_per_sample=1024,
        batch_size=8,
        gpus_per_node=8,
        number_of_nodes=1,  # total 8
    )
    scores = {cfg["number_of_nodes"]: score for cfg, score in predictor.find_similar_configurations(workload)}
    # 0.9 for the categoricals plus the proximity term.
    assert scores[1.0] == pytest.approx(1.0)
    assert scores[4.0] == pytest.approx(0.925)
    # The 32-GPU config ranks below the 8-GPU one.
    assert scores[1.0] > scores[4.0]


def _bundled_predictor(monkeypatch):
    """RetrievalPredictor forced onto the bundled 5-row sample (full trace absent)."""
    monkeypatch.setenv("DATA_DIR", "/tmp/coastline-no-such-trace-dir")
    return RetrievalPredictor()


def test_fallback_indexes_one_entry_per_distinct_sample_config(monkeypatch):
    """Without the private trace the cache loads the bundled sample, whose 5 rows are distinct
    (nodes, gpus, tokens, batch) configs, so the index has 5 entries."""
    predictor = _bundled_predictor(monkeypatch)
    # 5 sample rows: gpus 1/2/4/8 at batch 8, plus gpus 1 at batch 16.
    assert len(predictor.config_index) == 5


def test_exact_match_returns_recorded_first_run_not_an_aggregate(monkeypatch):
    """An exact hit returns the recorded run. The query (demo-llm-3b, lora, A100, 2048 tokens,
    batch 8, 1 node x 1 GPU) matches only sample row synthetic-sample-0001: 12000 tokens/s, 120 s.
    """
    predictor = _bundled_predictor(monkeypatch)
    workload = WorkloadSpec(
        llm_model="demo-llm-3b",
        fine_tuning_method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=2048,
        batch_size=8,
        gpus_per_node=1,
        number_of_nodes=1,
    )
    context = SystemContext(
        available_gpu_models=["NVIDIA-A100-SXM4-80GB"],
        max_gpus=8,
        gpu_memory={"NVIDIA-A100-SXM4-80GB": 80},
        constraints=Constraints(max_gpus=8, gpus_per_node=8, max_nodes=1),
    )
    prediction = predictor.predict(workload, context)
    assert prediction is not None  # an exact match is a hit
    # Values recorded in sample row 0001.
    assert prediction.predicted_throughput == pytest.approx(12000.0)
    assert prediction.predicted_runtime_seconds == pytest.approx(120.0)
    assert prediction.total_gpus == 1
    assert prediction.metadata["cache_hit"] is True


def test_cache_miss_returns_none_for_unrecorded_config(monkeypatch):
    """A config with no recorded run (batch_size=999) misses the index, and predict returns None
    so the caller falls back to a simulation predictor."""
    predictor = _bundled_predictor(monkeypatch)
    workload = WorkloadSpec(
        llm_model="demo-llm-3b",
        fine_tuning_method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=2048,
        batch_size=999,  # no such row in the sample, so a miss
        gpus_per_node=1,
        number_of_nodes=1,
    )
    context = SystemContext(
        available_gpu_models=["NVIDIA-A100-SXM4-80GB"],
        max_gpus=8,
        gpu_memory={"NVIDIA-A100-SXM4-80GB": 80},
        constraints=Constraints(max_gpus=8, gpus_per_node=8, max_nodes=1),
    )
    assert predictor.predict(workload, context) is None
