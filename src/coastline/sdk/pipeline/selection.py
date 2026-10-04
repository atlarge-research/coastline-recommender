"""Policy selection over evaluated feasible candidates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

# Defined in sdk/constants.py and re-exported for callers.
from coastline.sdk.constants import (  # noqa: F401
    PRESET_TO_POLICY,
    PRESET_WEIGHTS,
    NormalizationMode,
    SelectionPolicy,
)


@dataclass
class EvaluatedCandidate:
    gpus_per_node: int
    number_of_nodes: int
    total_gpus: int
    throughput: float
    power: float
    runtime: Optional[float]
    throughput_score: float
    power_score: float
    combined_score: float
    feasibility_metadata: dict
    batch_size: int = 0
    dominated: bool = False  # set by normalize_candidates in frontier mode; dominated configs are skipped at ranking


#: Score difference within which two candidates count as tied. The weighted sum is min-max
#: normalized over the feasible set, so several candidates often land very close to the top; an
#: exact comparison would leave their order to the grid enumeration.
TIE_EPS = 0.01


def tie_break_key(candidate: "EvaluatedCandidate") -> tuple:
    """Sort key for candidates the score cannot separate.

    Highest predicted throughput first, then the fewest GPUs, then the smallest batch: the
    fastest configuration and, among equally fast ones, the one that uses the least of the
    cluster. The node count and GPUs per node make the order total, so grid enumeration order
    never decides.
    """
    return (
        -candidate.throughput,
        candidate.total_gpus,
        candidate.batch_size,
        candidate.number_of_nodes,
        candidate.gpus_per_node,
    )


#: The policies rank_candidates ranks. A tuple, so a plain string such as "balanced" matches too.
WEIGHTED_POLICIES: tuple[SelectionPolicy, ...] = (
    SelectionPolicy.ENERGY,
    SelectionPolicy.BALANCED,
    SelectionPolicy.PERFORMANCE,
)


def check_selection_policy(policy: object) -> SelectionPolicy:
    """The SelectionPolicy for ``policy`` in any letter case; ValueError listing the policies
    for any other value."""
    value = policy.value if isinstance(policy, Enum) else policy
    key = str(value).strip().lower()
    if key not in {p.value for p in SelectionPolicy}:
        choices = ", ".join(p.value for p in SelectionPolicy)
        raise ValueError(f"unknown selection policy {policy!r}; choose from {choices}")
    return SelectionPolicy(key)


def rank_candidates(
    candidates: List[EvaluatedCandidate],
    policy: SelectionPolicy,
    *,
    alpha: float = 0.5,
    beta: float = 0.5,
    top_k: int = 3,
) -> List[EvaluatedCandidate]:
    """Sort feasible candidates on the weighted score of the energy, balanced and performance
    policies: ``alpha * throughput_score + beta * power_score``, the thesis score
    S = alpha * s_r + beta * s_e, with alpha the runtime weight and beta the energy weight.

    Any other policy raises ValueError. The min_gpu policy ranks no grid: the pipeline takes the
    first feasible GPU count.
    """
    if policy == SelectionPolicy.MIN_GPU:
        raise ValueError(
            "rank_candidates ranks energy, balanced or performance; min_gpu picks the first feasible GPU count"
        )
    if policy not in WEIGHTED_POLICIES:
        raise ValueError(f"unknown policy {policy!r}; choose from energy, balanced, performance")
    if not candidates:
        return []

    pool = [c for c in candidates if not c.dominated] or candidates  # drop dominated candidates before ranking

    for c in pool:
        c.combined_score = alpha * c.throughput_score + beta * c.power_score
    ranked = sorted(pool, key=lambda c: c.combined_score, reverse=True)
    # Candidates within TIE_EPS of the top score count as tied and are ordered by the shared
    # tie-break (highest throughput, fewest GPUs, smallest batch).
    if ranked:
        top = ranked[0].combined_score
        leaders = [c for c in ranked if c.combined_score >= top - TIE_EPS]
        leaders.sort(key=tie_break_key)
        ranked = leaders + [c for c in ranked if c.combined_score < top - TIE_EPS]
    return ranked[: max(1, min(top_k, len(ranked)))]


def _power_cost(c: "EvaluatedCandidate") -> float:
    """Total power [W]: per-GPU watts x GPU count. Lower is better."""
    return c.power * c.total_gpus


def _time_cost(c: "EvaluatedCandidate") -> float:
    """Runtime proxy: 1/throughput (the work is the same for all configs and cancels). Lower is better."""
    return (1.0 / c.throughput) if c.throughput > 0 else float("inf")


def normalize_candidates(
    candidates: List["EvaluatedCandidate"],
    mode: NormalizationMode = NormalizationMode.GRID,
) -> None:
    """Set throughput_score (the runtime score) and power_score (the energy score) in [0, 1],
    where higher is better.

    Axes: power = per-GPU watts x total_gpus; time = 1/throughput (the work cancels).
    mode ``grid``: min-max over all feasible candidates; ``frontier``: drop dominated ones first.
    """
    if not candidates:
        return
    for c in candidates:
        c.dominated = False

    if mode == NormalizationMode.FRONTIER:
        for c in candidates:
            cp, ct = _power_cost(c), _time_cost(c)
            c.dominated = any(
                _power_cost(o) <= cp and _time_cost(o) <= ct and (_power_cost(o) < cp or _time_cost(o) < ct)
                for o in candidates
                if o is not c
            )
        pool = [c for c in candidates if not c.dominated] or candidates
    else:  # "grid"
        pool = candidates

    pc = [_power_cost(c) for c in pool]
    tc = [_time_cost(c) for c in pool]
    p_min, p_max = min(pc), max(pc)
    t_min, t_max = min(tc), max(tc)
    for c in candidates:
        if c.dominated:
            c.power_score = c.throughput_score = 0.0
            continue
        c.power_score = (p_max - _power_cost(c)) / (p_max - p_min) if p_max > p_min else 1.0
        c.throughput_score = (t_max - _time_cost(c)) / (t_max - t_min) if t_max > t_min else 1.0
