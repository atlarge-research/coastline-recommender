"""RetrievalPredictor: similarity scoring + exact-match cache contract.

find_similar_configurations must score GPU proximity on TOTAL GPUs, not per-node.
The old bug compared config['gpus_per_node'] (per-node) against the workload's *total*
GPU count, so a 32-GPU config (8 per node x 4 nodes) looked like a perfect match for an
8-GPU target. config_index carries number_of_nodes too, so the fix multiplies them.
"""

from __future__ import annotations

import pytest

from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.performance.retrieval.cache_predictor import RetrievalPredictor


def test_find_similar_scores_categorical_plus_total_gpu_proximity():
    """Score = 0.4(llm) + 0.3(method) + 0.2(gpu) + 0.1/(1 + total_gpu_diff_frac).

    Both indexed configs match all three categoricals (+0.9). GPU proximity is the
    only differentiator and it MUST key off total GPUs (nodes x per_node), not per-node.
    Hand-derived against an 8-GPU target (1 node x 8):
      * total_match  (1 node x 8 = 8 GPUs):  diff=|8-8|/8=0   -> 0.9 + 0.1/(1+0)    = 1.000
      * per_node_only(4 nodes x 8 = 32 GPUs):diff=|32-8|/8=3  -> 0.9 + 0.1/(1+3)    = 0.925
    The old per-node comparison read gpus_per_node==8==target for BOTH, tying them at 1.0.
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
    # Exact hand-derived scores (a different form than the impl: 0.9 categorical + proximity term).
    assert scores[1.0] == pytest.approx(1.0)
    assert scores[4.0] == pytest.approx(0.925)
    # The regression the name promises: the 32-GPU config must rank strictly below the 8-GPU one.
    assert scores[1.0] > scores[4.0]


def _bundled_predictor(monkeypatch):
    """RetrievalPredictor forced onto the bundled 5-row sample (full trace absent)."""
    monkeypatch.setenv("DATA_DIR", "/tmp/coastline-no-such-trace-dir")
    return RetrievalPredictor()


def test_fallback_indexes_one_entry_per_distinct_sample_config(monkeypatch):
    """With the private trace absent the cache loads the bundled sample so `pip install`
    works out of the box. The sample has 5 rows, each a DISTINCT (nodes,gpus,tokens,batch)
    config -> exactly 5 unique hash-index entries (no collapse, no phantom rows)."""
    predictor = _bundled_predictor(monkeypatch)
    # 5 sample rows, all distinct configs (gpus=1/2/4/8 @ batch8, plus gpus=1 @ batch16).
    assert len(predictor.config_index) == 5


def test_exact_match_returns_recorded_first_run_not_an_aggregate(monkeypatch):
    """An exact cache HIT echoes the RECORDED first run verbatim (~0% error), not a median.

    The queried config (demo-llm-3b/lora/A100/2048tok/batch8, 1 node x 1 gpu) matches
    sample row synthetic-sample-0001 uniquely: dataset_tokens_per_second=12000, runtime=120.
    total_gpus = int(nodes x gpus_per_node) = int(1 x 1) = 1.
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
    assert prediction is not None  # guard: exact match must be a HIT, not a miss
    # Recorded values from sample row 0001 (independent of any predictor arithmetic).
    assert prediction.predicted_throughput == pytest.approx(12000.0)
    assert prediction.predicted_runtime_seconds == pytest.approx(120.0)
    assert prediction.total_gpus == 1
    assert prediction.metadata["cache_hit"] is True


def test_cache_miss_returns_none_for_unrecorded_config(monkeypatch):
    """A config with NO recorded run misses the hash index -> predict returns None so the
    orchestrator falls back to a simulation predictor. Same workload as the HIT test but
    batch_size=999 (never profiled), which changes the SHA256 key -> guaranteed miss."""
    predictor = _bundled_predictor(monkeypatch)
    workload = WorkloadSpec(
        llm_model="demo-llm-3b",
        fine_tuning_method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=2048,
        batch_size=999,  # no such row in the sample -> cache MISS
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
