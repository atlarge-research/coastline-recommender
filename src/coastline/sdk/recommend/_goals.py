"""The one objective vocabulary shared by both recommend surfaces.

``coastline.recommend(batch, goal=...)`` and ``Coastline(...).recommend(wl, goal=...)`` accept the
same ``goal`` strings; each maps it to what its own layer needs — a ``(strategy, preset)`` pair for
the facade, an engine label for the batch API.
"""

from __future__ import annotations

GOALS: tuple[str, ...] = ("balanced", "performance", "energy", "min_gpu")

# Friendly spellings → a canonical goal. The canonical names themselves always resolve.
_ALIASES: dict[str, str] = {
    "runtime": "performance",
    "lowest_runtime": "performance",
    "throughput": "performance",
    "energy_saver": "energy",
    "min-gpu": "min_gpu",
    "min_gpus": "min_gpu",
    "fewest": "min_gpu",
}

# Canonical goal → (strategy name, preset). min_gpu is its own strategy with no preset.
_STRATEGY_PRESET: dict[str, tuple[str, str | None]] = {
    "balanced": ("multi_objective", "balanced"),
    "performance": ("multi_objective", "performance"),
    "energy": ("multi_objective", "energy"),
    "min_gpu": ("min_gpu", None),
}


def normalize_goal(goal: str) -> str:
    """Canonical goal for any accepted spelling; ValueError (listing the options) on a typo."""
    key = str(goal).strip().lower().replace(" ", "_")
    key = _ALIASES.get(key, key)
    if key not in GOALS:
        raise ValueError(f"unknown goal {goal!r}; choose from {list(GOALS)} (aliases: {sorted(_ALIASES)})")
    return key


def goal_to_strategy_preset(goal: str) -> tuple[str, str | None]:
    """Map a goal to the facade's ``(strategy, preset)`` pair."""
    return _STRATEGY_PRESET[normalize_goal(goal)]
