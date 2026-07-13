"""Unified recommendation workflow: grid search -> feasibility -> simulate -> policy select."""

from __future__ import annotations

import logging
import math
from typing import List, Optional

from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.recommendation import Recommendation
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.pipeline.feasibility import FeasibilityChecker, create_feasibility_checker
from coastline.sdk.pipeline.grid import GridConfig, generate_candidates, grid_config_from_dict
from coastline.sdk.pipeline.selection import (
    PRESET_WEIGHTS,
    EvaluatedCandidate,
    SelectionPolicy,
    normalize_candidates,
    rank_candidates,
)
from coastline.sdk.predictors.base import BasePredictor

logger = logging.getLogger(__name__)


class GridWorkflowPipeline:
    """Shared grid → feasibility → simulate → policy pipeline."""

    def __init__(
        self,
        *,
        throughput_predictor: BasePredictor,
        power_predictor: BasePredictor,
        feasibility_checker: FeasibilityChecker,
        grid_config: GridConfig,
        selection_policy: SelectionPolicy,
        strategy_name: str,
        alpha: float = 0.5,
        beta: float = 0.5,
        preset: Optional[str] = None,
        normalization: str = "grid",
        energy_objective: str = "energy",
        runtime_guard_k: Optional[float] = None,
    ):
        self.throughput_predictor = throughput_predictor
        self.power_predictor = power_predictor
        self.feasibility_checker = feasibility_checker
        self.grid_config = grid_config
        self.selection_policy = selection_policy
        self.strategy_name = strategy_name
        self.alpha = alpha
        self.beta = beta
        self.preset = preset
        self.normalization = normalization
        self.energy_objective = energy_objective
        # Optional runtime guardrail: cap how slow a recommended config may be
        # relative to the fastest feasible one (None = off; see recommend()).
        self.runtime_guard_k = runtime_guard_k

    @staticmethod
    def _resolve_weights(
        strategy_cfg: dict, preset: Optional[str], alpha: Optional[float], beta: Optional[float]
    ) -> tuple[float, float]:
        """Normalized (alpha, beta): explicit args win, else the preset, else the config, else balanced."""
        if alpha is None or beta is None:
            if preset and preset in PRESET_WEIGHTS:
                a, b = PRESET_WEIGHTS[preset]
            else:
                a, b = strategy_cfg.get("alpha"), strategy_cfg.get("beta")
                if a is not None and b is not None:
                    a, b = float(a), float(b)
                else:
                    a, b = PRESET_WEIGHTS.get("balanced", (0.5, 0.5))
            alpha = a if alpha is None else alpha
            beta = b if beta is None else beta
        total = alpha + beta
        return (alpha / total, beta / total) if total > 0 else (alpha, beta)

    @staticmethod
    def _build_predictors(
        predictor_config: dict,
        throughput_predictor: Optional[BasePredictor],
        power_predictor: Optional[BasePredictor],
        feasibility_checker: Optional[FeasibilityChecker],
    ) -> tuple[BasePredictor, BasePredictor, FeasibilityChecker]:
        """Fill any predictor left unset from the config (a passed-in one is reused as-is)."""
        if throughput_predictor is None:
            throughput_predictor = _create_throughput_predictor(predictor_config)
        if power_predictor is None:
            power_predictor = _create_power_predictor(predictor_config)
        if feasibility_checker is None:
            feasibility_checker = create_feasibility_checker(predictor_config)
        return throughput_predictor, power_predictor, feasibility_checker

    @classmethod
    def from_config(
        cls,
        *,
        config: dict,
        selection_policy: SelectionPolicy,
        strategy_name: str,
        throughput_predictor: Optional[BasePredictor] = None,
        power_predictor: Optional[BasePredictor] = None,
        feasibility_checker: Optional[FeasibilityChecker] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        preset: Optional[str] = None,
        normalization: Optional[str] = None,
        energy_objective: Optional[str] = None,
        runtime_guard_k: Optional[float] = None,
    ) -> "GridWorkflowPipeline":
        strategy_cfg = config.get("strategy", {})
        alpha, beta = cls._resolve_weights(strategy_cfg, preset, alpha, beta)
        throughput_predictor, power_predictor, feasibility_checker = cls._build_predictors(
            config.get("predictors", {}), throughput_predictor, power_predictor, feasibility_checker
        )
        return cls(
            throughput_predictor=throughput_predictor,
            power_predictor=power_predictor,
            feasibility_checker=feasibility_checker,
            grid_config=grid_config_from_dict(config),
            selection_policy=selection_policy,
            strategy_name=strategy_name,
            alpha=alpha,
            beta=beta,
            preset=preset,
            normalization=normalization if normalization is not None else strategy_cfg.get("normalization", "grid"),
            energy_objective=(
                energy_objective if energy_objective is not None else strategy_cfg.get("energy_objective", "energy")
            ),
            runtime_guard_k=runtime_guard_k if runtime_guard_k is not None else strategy_cfg.get("runtime_guard_k"),
        )

    def recommend(
        self,
        workload: WorkloadSpec,
        context: SystemContext,
    ) -> List[Recommendation]:
        logger.info(
            "Workflow (%s): %s — grid → feasibility → simulate → %s",
            self.strategy_name,
            workload.llm_model,
            self.selection_policy,
        )

        candidates = generate_candidates(workload, context, self.grid_config)
        evaluated: List[EvaluatedCandidate] = []

        for variant in candidates:
            feasible, feas_meta = self.feasibility_checker.is_feasible(variant)
            if not feasible:
                continue

            throughput_pred = self.throughput_predictor.predict(variant, context)
            if throughput_pred is None:
                continue
            throughput = throughput_pred.predicted_throughput or 0.0
            # Reject NaN, +inf, -inf, and <=0; a bare x>0 admits +inf which then poisons min-max normalization.
            if not math.isfinite(throughput) or throughput <= 0:
                continue

            # Reuse the power Kavier already returned with throughput (one engine call, not
            # two); other predictors fall through to a dedicated call.
            if getattr(self.power_predictor, "WRAPS_THROUGHPUT_ENGINE", False) and throughput_pred.predicted_power:
                power = throughput_pred.predicted_power
            else:
                power_pred = self.power_predictor.predict(variant, context)
                if power_pred is None:
                    continue
                power = power_pred.predicted_power or 0.0
            # Same NaN/+inf/-inf/<=0 guard as throughput.
            if not math.isfinite(power) or power <= 0:
                continue

            evaluated.append(
                EvaluatedCandidate(
                    gpus_per_node=variant.gpus_per_node or 1,
                    number_of_nodes=variant.number_of_nodes or 1,
                    total_gpus=variant.total_gpus,
                    throughput=throughput,
                    power=power,
                    runtime=throughput_pred.predicted_runtime_seconds,
                    throughput_score=0.0,
                    power_score=0.0,
                    combined_score=0.0,
                    feasibility_metadata=feas_meta,
                    batch_size=variant.batch_size,
                )
            )

        if not evaluated:
            raise RuntimeError(
                f"Workflow ({self.strategy_name}): no feasible candidates found "
                f"in grid of {len(candidates)} configurations. "
                f"Check feasibility settings and predictor availability."
            )

        # Optional SLO guard: keep configs within k× of the fastest feasible (runtime <= k× fastest).
        # Off by default; applied before normalization; falls back to the full set rather than emptying it.
        if self.runtime_guard_k is not None and math.isfinite(self.runtime_guard_k) and self.runtime_guard_k > 0:
            threshold = max(c.throughput for c in evaluated) / self.runtime_guard_k
            guarded = [c for c in evaluated if c.throughput >= threshold]
            if guarded:
                evaluated = guarded

        # Normalize throughput/power scores across the whole feasible set.
        normalize_candidates(evaluated, self.normalization, self.energy_objective)

        # top_k applies to every policy; min_gpu already sorts by (total_gpus, -throughput),
        # so top_k>1 gives a ranked shortlist.
        top_k = self.grid_config.top_k
        ranked = rank_candidates(
            evaluated,
            self.selection_policy,
            alpha=self.alpha,
            beta=self.beta,
            top_k=top_k,
        )

        return [self._to_recommendation(row, rank=i + 1) for i, row in enumerate(ranked)]

    def _to_recommendation(self, row: EvaluatedCandidate, rank: int) -> Recommendation:
        # min_gpu has no score -> inverse-GPU proxy for ordering; other policies use combined_score.
        score = 1.0 / max(row.total_gpus, 1) if self.selection_policy == "min_gpu" else row.combined_score
        metadata = {
            "predicted_power_watts": row.power,
            "combined_score": score,
            "rank": rank,
            "selection_policy": self.selection_policy,
            "tokens_per_watt": row.throughput / row.power if row.power > 0 else 0,
            "throughput_score": row.throughput_score,
            "power_score": row.power_score,
            "feasibility": row.feasibility_metadata,
            "batch_size": row.batch_size,
            "workflow": "grid_feasibility_simulate_policy",
        }
        if self.preset:
            metadata["preset"] = self.preset
            metadata["alpha"] = self.alpha
            metadata["beta"] = self.beta

        return Recommendation(
            gpus_per_node=row.gpus_per_node,
            number_of_nodes=row.number_of_nodes,
            total_gpus=row.total_gpus,
            strategy=self.strategy_name,
            predicted_throughput=row.throughput,
            predicted_runtime_seconds=row.runtime,
            metadata=metadata,
        )


def _create_throughput_predictor(predictor_config: dict) -> BasePredictor:
    # Single source of truth: delegate to PolicyFactory so the workflow and the
    # strategy layer can never diverge on what `performance` resolves to (the old
    # copy here silently collapsed every named model — e.g. tabpfn — to CatBoost).
    # Imported lazily because recommendation_policies imports this module at load time.
    from coastline.sdk.policies import PolicyFactory

    return PolicyFactory.throughput_predictor(predictor_config)


def _create_power_predictor(predictor_config: dict) -> BasePredictor:
    from coastline.sdk.policies import PolicyFactory

    return PolicyFactory.power_predictor(predictor_config)
