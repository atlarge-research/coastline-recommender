"""Recommendation workflow: grid search, feasibility check, simulation, policy selection."""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Union

from coastline.sdk.constants import Preset
from coastline.sdk.exceptions import NoPredictionError
from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.recommendation import Prediction, Recommendation
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.pipeline.feasibility import FeasibilityChecker, create_feasibility_checker
from coastline.sdk.pipeline.grid import GridConfig, generate_candidates, grid_config_from_dict
from coastline.sdk.pipeline.parallel import (
    RUNTIME_SECTION,
    WORKERS_KEY,
    resolve_workers,
    run_feasibility,
    run_simulation,
)
from coastline.sdk.pipeline.selection import (
    PRESET_WEIGHTS,
    EvaluatedCandidate,
    NormalizationMode,
    SelectionPolicy,
    normalize_candidates,
    rank_candidates,
)
from coastline.sdk.predictors.base import BasePredictor
from coastline.sdk.predictors.performance.retrieval.cache_predictor import MEASURED_POWER_KEY

logger = logging.getLogger(__name__)


def check_runtime_guard_k(value: Union[float, str, None]) -> Optional[float]:
    """The runtime guard factor as a float, or None when the guard is off.

    The guard keeps configs at most k times slower than the fastest feasible one, so k must be
    finite and at least 1. Below 1 no config qualifies, and an infinite k sets no bound; both
    raise ValueError.
    """
    if value is None:
        return None
    k = float(value)
    if not math.isfinite(k) or k < 1:
        raise ValueError(f"max_slowdown (runtime_guard_k) must be a finite number >= 1, got {value!r}")
    return k


def simulate_one(
    throughput_predictor: BasePredictor,
    power_predictor: BasePredictor,
    variant: WorkloadSpec,
    context: SystemContext,
) -> Optional[tuple[float, float, Optional[float]]]:
    """(throughput, power, runtime) for one feasible candidate, or None if it drops out.

    A candidate drops out when it gets no throughput or no power, or when either number is
    non-finite or not positive: one infinity would break the min-max normalization used for
    ranking. The power is the power predictor's, or for a cache hit without one, the power
    measured in the matched run.
    """
    throughput_pred = throughput_predictor.predict(variant, context)
    if throughput_pred is None:
        return None
    throughput = throughput_pred.predicted_throughput or 0.0
    # Reject NaN, +inf, -inf and <= 0; a plain x > 0 check would let +inf through.
    if not math.isfinite(throughput) or throughput <= 0:
        return None

    # Kavier returns power with the throughput, so reuse it; other predictors need a separate
    # power call.
    power: Optional[float]
    if getattr(power_predictor, "WRAPS_THROUGHPUT_ENGINE", False) and throughput_pred.predicted_power:
        power = throughput_pred.predicted_power
    else:
        power_pred = power_predictor.predict(variant, context)
        power = power_pred.predicted_power if power_pred is not None else None
    # When the power predictor has no usable power (Kavier has none for a model outside its
    # catalog), a cache hit uses the power measured in its run.
    if not _usable(power):
        power = measured_power(throughput_pred)
    # Same NaN/+inf/-inf/<=0 guard as throughput.
    if power is None or not _usable(power):
        return None
    return throughput, power, throughput_pred.predicted_runtime_seconds


def _usable(value: Optional[float]) -> bool:
    """Whether a predicted throughput or power keeps its candidate: finite and above zero."""
    return value is not None and math.isfinite(value) and value > 0


def measured_power(prediction: Prediction) -> Optional[float]:
    """The per-GPU power measured in the run a cache hit matched, or None."""
    value = (prediction.metadata or {}).get(MEASURED_POWER_KEY)
    return float(value) if value is not None and _usable(float(value)) else None


def simulate_chunk(
    throughput_predictor: BasePredictor,
    power_predictor: BasePredictor,
    workloads: List[WorkloadSpec],
    context: SystemContext,
) -> List[Optional[tuple[float, float, Optional[float]]]]:
    """Simulate a run of feasible candidates, in input order. The pool's unit of work."""
    return [simulate_one(throughput_predictor, power_predictor, variant, context) for variant in workloads]


class GridWorkflowPipeline:
    """Grid, feasibility, simulation and policy selection, shared by all strategies."""

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
        normalization: str = NormalizationMode.GRID.value,
        runtime_guard_k: Optional[float] = None,
        workers: int = 1,
        predictor_config: Optional[dict] = None,
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
        # Optional cap on how much slower than the fastest feasible config a recommendation may
        # be (None = off; see recommend()).
        self.runtime_guard_k = check_runtime_guard_k(runtime_guard_k)
        # Worker processes per stage; 1 runs in this process (the library default). Workers
        # rebuild the checker from predictor_config, since a loaded AutoGluon model cannot be
        # pickled.
        self.workers = resolve_workers(workers)
        self.predictor_config = predictor_config

    @staticmethod
    def _resolve_weights(
        strategy_cfg: dict, preset: Optional[str], alpha: Optional[float], beta: Optional[float]
    ) -> tuple[float, float]:
        """Normalized (alpha, beta): from the arguments, else the preset, else the config, else balanced."""
        if alpha is None or beta is None:
            if preset and preset in PRESET_WEIGHTS:
                a, b = PRESET_WEIGHTS[preset]
            else:
                a, b = strategy_cfg.get("alpha"), strategy_cfg.get("beta")
                if a is not None and b is not None:
                    a, b = float(a), float(b)
                else:
                    a, b = PRESET_WEIGHTS.get(Preset.BALANCED.value, (0.5, 0.5))
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
        runtime_guard_k: Optional[float] = None,
        workers: Optional[int] = None,
        components_from_config: bool = False,
    ) -> "GridWorkflowPipeline":
        strategy_cfg = config.get("strategy", {})
        alpha, beta = cls._resolve_weights(strategy_cfg, preset, alpha, beta)
        # A caller may pass a ready-made predictor or checker instead of naming one in the
        # config. Workers rebuild components from the config and would get different objects,
        # so then predictor_config is withheld and every stage runs in this process.
        #
        # PolicyFactory sets ``components_from_config`` when it built the components from this
        # same predictors block; without it, its pipelines would never fork.
        injected = not components_from_config and any(
            component is not None for component in (throughput_predictor, power_predictor, feasibility_checker)
        )
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
            normalization=(
                normalization
                if normalization is not None
                else strategy_cfg.get("normalization", NormalizationMode.GRID.value)
            ),
            runtime_guard_k=runtime_guard_k if runtime_guard_k is not None else strategy_cfg.get("runtime_guard_k"),
            workers=(workers if workers is not None else (config.get(RUNTIME_SECTION) or {}).get(WORKERS_KEY, 1)),
            predictor_config=None if injected else config.get("predictors", {}),
        )

    def recommend(
        self,
        workload: WorkloadSpec,
        context: SystemContext,
    ) -> List[Recommendation]:
        logger.info(
            "Workflow (%s): %s - grid -> feasibility -> simulate -> %s",
            self.strategy_name,
            workload.llm_model,
            self.selection_policy,
        )

        candidates = generate_candidates(workload, context, self.grid_config)
        evaluated: List[EvaluatedCandidate] = []

        # Check every candidate before simulating any. With workers > 1 and an expensive checker
        # this is split across processes; verdicts come back in candidate order, which keeps the
        # tie-break below stable.
        verdicts = run_feasibility(self.feasibility_checker, self.predictor_config, candidates, self.workers)

        survivors = [(variant, feas_meta) for variant, (feasible, feas_meta) in zip(candidates, verdicts) if feasible]

        # Simulate every feasible candidate. Forked only for data-driven predictors (milliseconds
        # per call); Kavier and the cache run in this process.
        predictions = run_simulation(
            self.throughput_predictor,
            self.power_predictor,
            self.predictor_config,
            [variant for variant, _ in survivors],
            context,
            self.workers,
        )

        for (variant, feas_meta), prediction in zip(survivors, predictions):
            if prediction is None:
                continue
            throughput, power, runtime = prediction
            evaluated.append(
                EvaluatedCandidate(
                    gpus_per_node=variant.gpus_per_node or 1,
                    number_of_nodes=variant.number_of_nodes or 1,
                    total_gpus=variant.total_gpus,
                    throughput=throughput,
                    power=power,
                    runtime=runtime,
                    throughput_score=0.0,
                    power_score=0.0,
                    combined_score=0.0,
                    feasibility_metadata=feas_meta,
                    batch_size=variant.batch_size,
                )
            )

        if not evaluated:
            if survivors:
                # Candidates passed the feasibility check, so a predictor failed; name it and
                # its reason.
                raise NoPredictionError(self._no_prediction_message(survivors[0][0], context, len(survivors)))
            raise RuntimeError(
                f"Workflow ({self.strategy_name}): no feasible candidates found "
                f"in grid of {len(candidates)} configurations. "
                f"Check the feasibility settings and the search grid."
            )

        # Optional runtime guard: keep configs whose runtime is at most k times the fastest
        # feasible one. Off by default and applied before normalization.
        if self.runtime_guard_k is not None and math.isfinite(self.runtime_guard_k) and self.runtime_guard_k > 0:
            threshold = max(c.throughput for c in evaluated) / self.runtime_guard_k
            guarded = [c for c in evaluated if c.throughput >= threshold]
            if guarded:
                if len(guarded) < len(evaluated):
                    logger.info(
                        "Runtime guard (k=%s) dropped %d of %d candidates",
                        self.runtime_guard_k,
                        len(evaluated) - len(guarded),
                        len(evaluated),
                    )
                evaluated = guarded
            else:
                # The guard would drop every config, so the full set is kept, and it may exceed
                # the requested max_slowdown. Warn about it.
                logger.warning(
                    "Runtime guard (k=%s) would reject every candidate, so it was not applied: "
                    "the returned configurations may exceed the requested max slowdown.",
                    self.runtime_guard_k,
                )

        # Normalize throughput/power scores across the whole feasible set.
        normalize_candidates(evaluated, self.normalization)

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

    def _no_prediction_message(self, variant: WorkloadSpec, context: SystemContext, n_survivors: int) -> str:
        """The error text when every candidate that passed feasibility got no usable prediction.

        The simulation stage keeps only the numbers, so the predictors are asked once more, for
        one of those candidates, in the order ``simulate_one`` asks them: the throughput
        predictor, then the power predictor when the throughput was usable. The message names
        the one that gave nothing and the reason it reports (Kavier puts it in
        ``metadata['error_detail']``). This runs only on the failure path.
        """
        config = self.predictor_config or {}
        name, predictor = config.get("performance"), self.throughput_predictor
        prediction = predictor.predict(variant, context)
        if prediction is not None and _usable(prediction.predicted_throughput):
            # The throughput was usable, so the power dropped the candidate.
            name, predictor = config.get("energy"), self.power_predictor
            if not (getattr(predictor, "WRAPS_THROUGHPUT_ENGINE", False) and prediction.predicted_power):
                prediction = predictor.predict(variant, context)
        label = f"the {name} predictor" if name else type(predictor).__name__
        metadata = (prediction.metadata or {}) if prediction is not None else {}
        reason = metadata.get("error_detail") or metadata.get("error")
        if prediction is None and not reason:
            reason = getattr(predictor, "MISS_REASON", None)
        message = (
            f"Workflow ({self.strategy_name}): {label} gave no usable prediction for any of the "
            f"{n_survivors} configurations that passed the feasibility check"
        )
        return f"{message}: {reason}" if reason else f"{message}."

    def _to_recommendation(self, row: EvaluatedCandidate, rank: int) -> Recommendation:
        # min_gpu has no score, so 1/total_gpus stands in for it; other policies use combined_score.
        score = 1.0 / max(row.total_gpus, 1) if self.selection_policy == SelectionPolicy.MIN_GPU else row.combined_score
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
    # PolicyFactory resolves `performance` here too, so the workflow and the strategies agree.
    # Imported lazily because coastline.sdk.policies imports this module.
    from coastline.sdk.policies import PolicyFactory

    return PolicyFactory.throughput_predictor(predictor_config)


def _create_power_predictor(predictor_config: dict) -> BasePredictor:
    from coastline.sdk.policies import PolicyFactory

    return PolicyFactory.power_predictor(predictor_config)
