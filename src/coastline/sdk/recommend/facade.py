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

# The one input vocabulary: WorkloadSpec field names. A dict or CSV supplies columns by
# field name (llm_model / fine_tuning_method / gpu_model / ...); no synonyms.
_WORKLOAD_FIELDS = set(WorkloadSpec.model_fields)


def _coerce_workload(workload: WorkloadInput) -> WorkloadSpec:
    if isinstance(workload, WorkloadSpec):
        return workload
    if isinstance(workload, dict):
        # Keys are WorkloadSpec field names (the one vocabulary), same as coastline.recommend(batch).
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
    """Derive a single-GPU-model context from the workload (loud on unknown GPU)."""
    return SystemContext.for_gpus(
        [workload.gpu_model],
        max_gpus=max_gpus,
        gpus_per_node=DEFAULT_GPUS_PER_NODE,
        max_nodes=max(1, math.ceil(max_gpus / DEFAULT_GPUS_PER_NODE)),
    )


class Coastline:
    """A configured recommender: pick a ``predictor`` once, then call it (or ``.recommend(...)``)
    per workload. Each call returns a ``list[Recommendation]`` (typed objects), best-first — the
    single-workload counterpart to ``coastline.recommend(batch)``, which returns a DataFrame."""

    def __init__(
        self,
        predictor: str = "kavier",
        *,
        energy: str = EnergyBackend.KAVIER_POWER.value,
        feasibility: str = FeasibilityMode.AUTOCONF.value,
    ) -> None:
        self.predictor = normalize_predictor(predictor)
        self.energy = energy
        self.feasibility = feasibility

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
        """Recommend GPU/node configurations for ``workload`` (WorkloadSpec, dict, or CSV path), best-first.

        Returns a ``list[Recommendation]`` (typed objects). ``goal`` is the shared, discoverable
        knob (``"balanced"`` | ``"performance"`` | ``"energy"`` | ``"min_gpu"``) — the same vocabulary
        as ``coastline.recommend(batch, goal=...)``; it sets ``strategy``/``preset`` for you.
        ``strategy``/``preset``/``alpha``/``beta`` remain for advanced manual control.
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
            },
            "grid": {
                # No explicit grid -> search the full menu; generate_candidates clips it to max_gpus.
                "batch_sizes": batch_sizes or list(DEFAULT_BATCH_SIZES),
                "total_gpus": total_gpus or list(GPU_BUDGETS),
                "top_k": top_k,
            },
        }
        # Route through the single engine seam (build strategy -> recommend -> normalize).
        # total_tokens=0: the facade returns raw recs and never derives runtime/energy.
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
