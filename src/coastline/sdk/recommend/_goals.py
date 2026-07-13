"""The one objective vocabulary shared by both recommend surfaces.

``coastline.recommend(batch, goal=...)`` and ``Coastline(...).recommend(wl, goal=...)`` accept the
same ``goal`` strings; each maps it to what its own layer needs — a ``(strategy, preset)`` pair for
the facade, an engine label for the batch API.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Goal:
    """One objective, described once. Every other surface derives its view from these records:
    the facade wants ``(strategy, preset)``, the engine/REPL want the display ``label``, the
    batch API maps canonical→label, and the recommendation rationale wants the ``phrase``."""

    canonical: str
    label: str  # display label (interactive REPL choices + engine.GOALS key + answers["goal_label"])
    strategy: str
    preset: str | None
    phrase: str  # short reason used in the recommendation rationale


# The single source of truth for the objective vocabulary.
GOAL_SPECS: tuple[Goal, ...] = (
    Goal(
        "balanced", "Multi-objective balanced", "multi_objective", "balanced",
        "the best throughput-vs-energy balance",
    ),
    Goal(
        "performance", "Multi-objective lowest runtime", "multi_objective", "performance",
        "the highest throughput",
    ),
    Goal("energy", "Multi-objective energy-saver", "multi_objective", "energy", "the lowest energy"),
    Goal("min_gpu", "Fewest GPUs that fit", "min_gpu", None, "the fewest GPUs that fit"),
)

GOALS: tuple[str, ...] = tuple(g.canonical for g in GOAL_SPECS)
_BY_CANONICAL: dict[str, Goal] = {g.canonical: g for g in GOAL_SPECS}

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


def normalize_goal(goal: str) -> str:
    """Canonical goal for any accepted spelling; ValueError (listing the options) on a typo."""
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
    """Display label → ``(strategy, preset)`` — the table the engine/REPL enumerate as choices."""
    return {g.label: (g.strategy, g.preset) for g in GOAL_SPECS}


def goal_to_label(goal: str) -> str:
    """Any accepted goal spelling → its engine display label."""
    return _BY_CANONICAL[normalize_goal(goal)].label


def rationale_phrase(key: str) -> str | None:
    """The rationale phrase for a canonical goal (or ``None`` for anything without one, e.g. the
    ``multi_objective`` strategy name — the caller falls back to a generic phrase)."""
    g = _BY_CANONICAL.get(key)
    return g.phrase if g else None
