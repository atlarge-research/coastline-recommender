"""Tests how well Coastline ranks the measured configurations of each workload by throughput.

The floors sit below the current values (kavier: top-1 0.78, p90 regret 0.017, Spearman 0.87;
cache: perfect) with some headroom. The tests use the kavier and cache predictors, so no trained
ML model is unpickled (that can segfault the host).
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
    # The cache returns the measured value for a trace config, so its ranking is exact. The
    # default `intelligent` predictor tries this exact-match lookup first.
    result = evaluate_predictor(_load_trace(), "cache")
    assert result["top1_hit_rate"] == 1.0, result
    assert result["p90_regret"] == 0.0, result
