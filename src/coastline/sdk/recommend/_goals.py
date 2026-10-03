"""Goal names accepted by both ``coastline.recommend(batch, goal=...)`` and
``Coastline(...).recommend(wl, goal=...)``.

The facade maps a goal to a ``(strategy, preset)`` pair; the batch API maps it to an engine label.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Goal:
    """One goal. The facade reads ``(strategy, preset)``, the engine and REPL read ``label``, the
    batch API maps ``canonical`` to ``label``, and the rationale uses ``phrase``."""

    canonical: str
    label: str  # display label: REPL choice, engine.GOALS key and answers["goal_label"]
    strategy: str
    preset: str | None
    phrase: str  # short reason used in the recommendation rationale


# All goals.
GOAL_SPECS: tuple[Goal, ...] = (
    Goal(
        "balanced",
        "Multi-objective balanced",
        "multi_objective",
        "balanced",
        "the best throughput-vs-energy balance",
    ),
    Goal(
        "performance",
        "Multi-objective lowest runtime",
        "multi_objective",
        "performance",
        "the highest throughput",
    ),
    Goal("energy", "Multi-objective energy-saver", "multi_objective", "energy", "the lowest energy"),
    Goal("min_gpu", "Fewest GPUs that fit", "min_gpu", None, "the fewest GPUs that fit"),
)

GOALS: tuple[str, ...] = tuple(g.canonical for g in GOAL_SPECS)
_BY_CANONICAL: dict[str, Goal] = {g.canonical: g for g in GOAL_SPECS}

# Other accepted spellings of the canonical goals. Canonical names are accepted as well.
_ALIASES: dict[str, str] = {
    "runtime": "performance",
    "lowest_runtime": "performance",
    "throughput": "performance",
    "energy_saver": "energy",
    "min-gpu": "min_gpu",
    "min_gpus": "min_gpu",
    "fewest": "min_gpu",
}


def normalize_goal(goal: str) -> str:
    """The canonical goal for an accepted spelling; ValueError listing the options otherwise."""
    key = str(goal).strip().lower().replace(" ", "_")
    key = _ALIASES.get(key, key)
    if key not in GOALS:
        raise ValueError(f"unknown goal {goal!r}; choose from {list(GOALS)} (aliases: {sorted(_ALIASES)})")
    return key


def goal_to_strategy_preset(goal: str) -> tuple[str, str | None]:
    """Map a goal to the facade's ``(strategy, preset)`` pair."""
    g = _BY_CANONICAL[normalize_goal(goal)]
    return (g.strategy, g.preset)


def engine_goals() -> dict[str, tuple[str, str | None]]:
    """Each display label with its ``(strategy, preset)``; the engine and REPL list these as choices."""
    return {g.label: (g.strategy, g.preset) for g in GOAL_SPECS}


def goal_to_label(goal: str) -> str:
    """The engine display label for any accepted goal spelling."""
    return _BY_CANONICAL[normalize_goal(goal)].label


def rationale_phrase(key: str | None) -> str | None:
    """The rationale phrase for a canonical goal, or None for a key without one (such as the
    ``multi_objective`` strategy name); the caller then uses a generic phrase."""
    # Presets are case-insensitive, so a config's 'Performance' gets the performance phrase.
    g = _BY_CANONICAL.get(str(key).strip().lower()) if key else None
    return g.phrase if g else None
