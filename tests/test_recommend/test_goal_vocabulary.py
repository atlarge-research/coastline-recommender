"""Phase 5.2 characterization: the goal vocabulary (balanced/performance/energy/min_gpu) is
encoded once in `_goals` and the engine + batch API derive their views from it. These tests
pin the EXACT tables the three surfaces exposed before consolidation, so the derived views
must reproduce them byte-for-byte (behavior-preserving).
"""

from __future__ import annotations

from coastline.sdk.recommend import _goals, engine

# The literal tables as they stood before consolidation (independent oracle).
_EXPECTED_ENGINE_GOALS = {
    "Multi-objective balanced": ("multi_objective", "balanced"),
    "Multi-objective lowest runtime": ("multi_objective", "performance"),
    "Multi-objective energy-saver": ("multi_objective", "energy"),
    "Fewest GPUs that fit": ("min_gpu", None),
}
_EXPECTED_GOAL_TO_LABEL = {
    "balanced": "Multi-objective balanced",
    "performance": "Multi-objective lowest runtime",
    "energy": "Multi-objective energy-saver",
    "min_gpu": "Fewest GPUs that fit",
}
_EXPECTED_RATIONALE = {
    "balanced": "the best throughput-vs-energy balance",
    "performance": "the highest throughput",
    "energy": "the lowest energy",
    "min_gpu": "the fewest GPUs that fit",
}


def test_engine_goals_table_unchanged():
    assert dict(engine.GOALS) == _EXPECTED_ENGINE_GOALS


def test_goal_to_label_unchanged():
    for goal, label in _EXPECTED_GOAL_TO_LABEL.items():
        assert _goals.goal_to_label(goal) == label


def test_rationale_phrase_unchanged():
    for goal, phrase in _EXPECTED_RATIONALE.items():
        assert _goals.rationale_phrase(goal) == phrase
    # unknown / multi_objective strategy_name has no phrase (falls through to the default)
    assert _goals.rationale_phrase("multi_objective") is None


def test_strategy_preset_still_resolves_for_facade():
    assert _goals.goal_to_strategy_preset("balanced") == ("multi_objective", "balanced")
    assert _goals.goal_to_strategy_preset("min_gpu") == ("min_gpu", None)
    assert _goals.goal_to_strategy_preset("throughput") == ("multi_objective", "performance")  # alias
