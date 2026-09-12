"""Policy selection over evaluated feasible candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

# The closed-set vocabulary lives in one home (sdk/constants.py); re-exported here for callers.
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


#: How close two scores must be to count as tied. The weighted sum is min-max normalised over the
#: feasible set, so several candidates routinely land within a hair of the top -- and an exact
#: equality test would leave those decided by grid enumeration order.
TIE_EPS = 0.01


def tie_break_key(candidate: "EvaluatedCandidate") -> tuple:
    """A total order over candidates the scorer cannot separate.

    The score decides the ranking; this decides what happens when it cannot. Highest predicted
    throughput first, then the fewest GPUs, then the smallest batch -- prefer the fastest, and
    among equally fast configurations the one that asks least of the cluster. The last two terms
    are not a preference, they are what makes this a TOTAL order: without them two candidates the
    earlier terms cannot separate would be left in grid enumeration order, which is exactly the
    kind of positional dependence that makes a result irreproducible.
    """
    return (
        -candidate.throughput,
        candidate.total_gpus,
        candidate.batch_size,
        candidate.number_of_nodes,
        candidate.gpus_per_node,
    )


def rank_candidates(
    candidates: List[EvaluatedCandidate],
    policy: SelectionPolicy,
    *,
    alpha: float = 0.5,
    beta: float = 0.5,
    top_k: int = 3,
) -> List[EvaluatedCandidate]:
    """Sort feasible candidates by policy; energy/balanced/performance all use the same
    weighted-sum scorer (α=power, β=time)."""
    if not candidates:
        return []

    pool = [c for c in candidates if not c.dominated] or candidates  # drop dominated candidates before ranking

    if policy == SelectionPolicy.MIN_GPU:
        # Fewest GPUs is this policy's whole point, so it leads; the shared tie-break settles
        # everything under it (and its own first term, throughput, is what used to be here).
        ranked = sorted(pool, key=lambda c: (c.total_gpus, *tie_break_key(c)))
        return ranked[: max(1, min(top_k, len(ranked)))]

    for c in pool:
        c.combined_score = alpha * c.power_score + beta * c.throughput_score
    ranked = sorted(pool, key=lambda c: c.combined_score, reverse=True)
    # Candidates within TIE_EPS of the top score are treated as tied and ordered by the shared
    # tie-break: highest throughput, fewest GPUs, smallest batch. Without it a flat score band
    # collapses to whatever order the grid happened to enumerate.
    if ranked:
        top = ranked[0].combined_score
        leaders = [c for c in ranked if c.combined_score >= top - TIE_EPS]
        leaders.sort(key=tie_break_key)
        ranked = leaders + [c for c in ranked if c.combined_score < top - TIE_EPS]
    return ranked[: max(1, min(top_k, len(ranked)))]


def _power_cost(c: "EvaluatedCandidate") -> float:
    """Total instantaneous power (W) = per-GPU watts × GPU count. Lower is better."""
    return c.power * c.total_gpus


def _time_cost(c: "EvaluatedCandidate") -> float:
    """Runtime proxy = 1/throughput (work is config-invariant so cancels in min-max). Lower is better."""
    return (1.0 / c.throughput) if c.throughput > 0 else float("inf")


def normalize_candidates(
    candidates: List["EvaluatedCandidate"],
    mode: NormalizationMode = NormalizationMode.GRID,
) -> None:
    """Populate throughput_score and power_score in [0,1] (higher=better).

    Axes: power = per-GPU watts × total_gpus; time = 1/throughput (work cancels).
    mode: ``grid`` = min-max over all feasible; ``frontier`` = drop dominated first.
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
