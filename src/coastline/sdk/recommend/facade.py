"""Importable recommender facade over PolicyFactory."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, List, Optional, Union

from coastline.sdk.constants import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_GPUS_PER_NODE,
    GPU_BUDGETS,
    EnergyBackend,
    FeasibilityMode,
)
from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.recommendation import Recommendation
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.policies import normalize_predictor
from coastline.sdk.recommend import engine
from coastline.sdk.recommend._goals import goal_to_strategy_preset

WorkloadInput = Union[WorkloadSpec, dict, str, Path]

# Input keys and CSV columns are WorkloadSpec field names (llm_model, fine_tuning_method,
# gpu_model, ...), with no synonyms.
_WORKLOAD_FIELDS = set(WorkloadSpec.model_fields)


def _coerce_workload(workload: WorkloadInput) -> WorkloadSpec:
    if isinstance(workload, WorkloadSpec):
        return workload
    if isinstance(workload, dict):
        # Keys are WorkloadSpec field names, as in coastline.recommend(batch).
        fields = {key: value for key, value in workload.items() if key in _WORKLOAD_FIELDS}
        return WorkloadSpec(**fields)
    if isinstance(workload, (str, Path)):
        return _workload_from_csv(workload)
    raise TypeError(f"workload must be a WorkloadSpec, dict, or CSV path; got {type(workload).__name__}")


def _workload_from_csv(path: WorkloadInput) -> WorkloadSpec:
    """Build a WorkloadSpec from the first row of a CSV (columns are WorkloadSpec field names)."""
    import pandas as pd

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")
    row = df.iloc[0]
    fields: dict[str, Any] = {
        field: row[field] for field in _WORKLOAD_FIELDS if field in df.columns and pd.notna(row[field])
    }
    for f in ("tokens_per_sample", "batch_size", "gpus_per_node", "number_of_nodes"):
        if f in fields:
            fields[f] = int(fields[f])
    return WorkloadSpec(**fields)


def _default_context(workload: WorkloadSpec, max_gpus: int) -> SystemContext:
    """A context with only the workload's GPU model; an unknown GPU raises."""
    return SystemContext.for_gpus(
        [workload.gpu_model],
        max_gpus=max_gpus,
        gpus_per_node=DEFAULT_GPUS_PER_NODE,
        max_nodes=max(1, math.ceil(max_gpus / DEFAULT_GPUS_PER_NODE)),
    )


class Coastline:
    """A configured recommender: choose a ``predictor`` once, then call it (or ``.recommend(...)``)
    per workload. Each call returns a ``list[Recommendation]``, best first. The batch version,
    ``coastline.recommend(batch)``, returns a DataFrame."""

    def __init__(
        self,
        predictor: str = "kavier",
        *,
        energy: str = EnergyBackend.KAVIER_POWER.value,
        feasibility: str = FeasibilityMode.AUTOCONF.value,
        empirical_oom_guard: bool = False,
    ) -> None:
        """Set the predictor backends.

        ``empirical_oom_guard`` adds the per-device token ceiling ``EMPIRICAL_OOM_TOKEN_BUDGET``
        (fitted to observed OOMs) on top of the ``feasibility`` backend. It can only turn a
        feasible candidate infeasible, so it is off by default. It lets a caller without
        AutoConf, for example on the ``rules`` backend, reject per-device loads that ran out of
        memory in the measured campaigns.
        """
        self.predictor = normalize_predictor(predictor)
        self.energy = energy
        self.feasibility = feasibility
        self.empirical_oom_guard = empirical_oom_guard

    def recommend(
        self,
        workload: WorkloadInput,
        *,
        goal: Optional[str] = None,
        context: Optional[SystemContext] = None,
        strategy: str = "multi_objective",
        preset: str = "balanced",
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        total_gpus: Optional[List[int]] = None,
        batch_sizes: Optional[List[int]] = None,
        top_k: int = 5,
        max_gpus: int = 16,
    ) -> List[Recommendation]:
        """Recommend GPU and node configurations for ``workload`` (WorkloadSpec, dict or CSV path).

        Returns a ``list[Recommendation]``, best first. ``goal`` (``"balanced"``,
        ``"performance"``, ``"energy"`` or ``"min_gpu"``) takes the same values as in
        ``coastline.recommend(batch, goal=...)`` and sets ``strategy`` and ``preset``. Pass
        ``strategy``, ``preset``, ``alpha`` and ``beta`` to set them by hand.
        """
        if max_gpus < 1:
            raise ValueError(f"max_gpus must be >= 1, got {max_gpus}")
        # `goal` resolves to (strategy, preset), overriding those params.
        if goal is not None:
            strategy, goal_preset = goal_to_strategy_preset(goal)
            if goal_preset is not None:
                preset = goal_preset
        wl = _coerce_workload(workload)
        ctx = context if context is not None else _default_context(wl, max_gpus)
        config = {
            "strategy": {"name": strategy, "preset": preset},
            "predictors": {
                "performance": self.predictor,
                "energy": self.energy,
                "feasibility": self.feasibility,
                "empirical_oom_guard": self.empirical_oom_guard,
            },
            "grid": {
                # Without a grid, search the defaults; generate_candidates clips them to max_gpus.
                "batch_sizes": batch_sizes or list(DEFAULT_BATCH_SIZES),
                "total_gpus": total_gpus or list(GPU_BUDGETS),
                "top_k": top_k,
            },
        }
        # Same path as the other entry points: build the strategy, recommend, return a list.
        # total_tokens=0: the facade does not derive runtime or energy.
        recs, _ = engine.run_request(
            engine.RecommendRequest(
                workload=wl,
                context=ctx,
                config=config,
                strategy_name=strategy,
                preset=preset,
                alpha=alpha,
                beta=beta,
            )
        )
        return recs

    __call__ = recommend
