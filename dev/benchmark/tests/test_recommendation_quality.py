"""Recommendation-quality regression guard.

Unit tests prove the predictors are accurate; this asserts coastline keeps *ranking* real
configurations well against the ground-truth trace. Floors sit below the current numbers
(kavier top-1 0.78 / p90-regret 0.017 / spearman 0.87; cache perfect) with headroom, so a
genuine regression trips them without flaking. Pins the analytical ``kavier`` predictor so
no trained-ML artifact is unpickled (host segfault).
"""

from __future__ import annotations

from benchmark.recommendation_quality import _load_trace, evaluate_predictor


def test_kavier_recommendation_quality_above_floor():
    result = evaluate_predictor(_load_trace(), "kavier")
    assert result["workloads"] >= 40, result
    assert result["top1_hit_rate"] >= 0.70, result
    assert result["p90_regret"] <= 0.05, result
    assert result["mean_spearman"] >= 0.75, result


def test_cache_exact_match_ranks_perfectly():
    # The cache returns the measured value for a trace config, so its ranking must be exact;
    # any drop means the exact-match retrieval (the live recommend default's first stage) broke.
    result = evaluate_predictor(_load_trace(), "cache")
    assert result["top1_hit_rate"] == 1.0, result
    assert result["p90_regret"] == 0.0, result
