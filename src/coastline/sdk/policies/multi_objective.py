"""Multi-objective strategy: grid → feasibility → simulate → weighted policy selection."""

import logging
from typing import List, Literal, Optional

from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.recommendation import Recommendation
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.pipeline.selection import PRESET_TO_POLICY, PRESET_WEIGHTS
from coastline.sdk.pipeline.workflow import GridWorkflowPipeline
from coastline.sdk.policies.base import BaseStrategy
from coastline.sdk.predictors.base import BasePredictor

logger = logging.getLogger(__name__)

PolicyPreset = Literal[
    "energy",
    "balanced",
    "performance",
    "energy-frontier",
    "balanced-frontier",
    "performance-frontier",
]


class MultiObjectiveStrategy(BaseStrategy):
    """Multi-objective strategy via unified grid workflow + preset weights."""

    def __init__(
        self,
        throughput_predictor: BasePredictor,
        power_predictor: BasePredictor,
        preset: Optional[PolicyPreset] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        *,
        config: Optional[dict] = None,
        pipeline: Optional[GridWorkflowPipeline] = None,
    ):
        self.throughput_predictor = throughput_predictor
        self.power_predictor = power_predictor

        if alpha is not None and beta is not None:
            # Normalise by sum so the α:β ratio is preserved; floor negatives to 0, fallback to 0.5/0.5 if both zero.
            self.alpha = max(0.0, alpha)
            self.beta = max(0.0, beta)
            total = self.alpha + self.beta
            if total > 0:
                self.alpha /= total
                self.beta /= total
            else:
                logger.warning(
                    "MultiObjectiveStrategy: alpha+beta == 0 (alpha=%s, beta=%s); "
                    "a zero weight sum makes every candidate score 0 and the winner "
                    "arbitrary. Falling back to a 0.5/0.5 balanced split.",
                    alpha,
                    beta,
                )
                self.alpha, self.beta = 0.5, 0.5
            self.preset: str = "custom"
            selection = "balanced"
        elif preset is not None and preset in PRESET_WEIGHTS:
            self.alpha, self.beta = PRESET_WEIGHTS[preset]
            self.preset = preset
            selection = PRESET_TO_POLICY[preset]
        else:
            self.alpha, self.beta = PRESET_WEIGHTS["balanced"]
            self.preset = "balanced"
            selection = "balanced"

        strategy_name = f"multi_objective_{self.preset}"

        # "-frontier" preset → frontier (non-dominated) normalization; same axes/weights as the base trio.
        normalization = "frontier" if str(self.preset).endswith("-frontier") else None

        if pipeline is not None:
            self._pipeline = pipeline
        else:
            self._pipeline = GridWorkflowPipeline.from_config(
                config=config or {},
                selection_policy=selection,
                strategy_name=strategy_name,
                throughput_predictor=throughput_predictor,
                power_predictor=power_predictor,
                alpha=self.alpha,
                beta=self.beta,
                preset=self.preset,
                normalization=normalization,
            )

        logger.info(
            "MultiObjectiveStrategy: preset=%s, alpha=%.2f, beta=%.2f",
            self.preset,
            self.alpha,
            self.beta,
        )

    def get_name(self) -> str:
        return f"multi_objective_{self.preset}"

    def recommend(
        self,
        workload: WorkloadSpec,
        context: SystemContext,
    ) -> List[Recommendation]:
        return self._pipeline.recommend(workload, context)
