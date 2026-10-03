"""Multi-objective strategy: grid, feasibility and simulation, then weighted selection."""

import logging
from enum import Enum
from typing import List, Optional, Union

from coastline.sdk.constants import PRESET_TO_POLICY, PRESET_WEIGHTS, Preset, Strategy
from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.recommendation import Recommendation
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.pipeline.workflow import GridWorkflowPipeline
from coastline.sdk.policies.base import BaseStrategy
from coastline.sdk.predictors.base import BasePredictor

logger = logging.getLogger(__name__)

# Alias of constants.Preset.
PolicyPreset = Preset


def normalize_preset(preset: Union[str, Preset]) -> str:
    """The preset key for any letter case; ValueError listing the presets for an unknown one."""
    value = preset.value if isinstance(preset, Enum) else preset
    key = str(value).strip().lower()
    if key not in PRESET_WEIGHTS:
        raise ValueError(f"unknown preset {preset!r}; choose from {list(PRESET_WEIGHTS)}")
    return key


class MultiObjectiveStrategy(BaseStrategy):
    """Multi-objective strategy: the grid workflow ranked with preset or custom weights."""

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
        components_from_config: bool = False,
    ):
        self.throughput_predictor = throughput_predictor
        self.power_predictor = power_predictor

        if alpha is not None and beta is not None:
            # Clip negatives to 0 and divide by the sum, keeping the alpha:beta ratio; both 0 gives 0.5/0.5.
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
        elif preset is not None:
            # preset=None means balanced; an unknown preset raises.
            self.preset = normalize_preset(preset)
            self.alpha, self.beta = PRESET_WEIGHTS[self.preset]
            selection = PRESET_TO_POLICY[self.preset]
        else:
            self.alpha, self.beta = PRESET_WEIGHTS["balanced"]
            self.preset = "balanced"
            selection = "balanced"

        strategy_name = f"multi_objective_{self.preset}"

        # A "-frontier" preset normalizes over the non-dominated frontier, with its base preset's weights.
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
                components_from_config=components_from_config,
            )

        logger.info(
            "MultiObjectiveStrategy: preset=%s, alpha=%.2f, beta=%.2f",
            self.preset,
            self.alpha,
            self.beta,
        )

    def get_name(self) -> str:
        return f"{Strategy.MULTI_OBJECTIVE.value}_{self.preset}"

    def recommend(
        self,
        workload: WorkloadSpec,
        context: SystemContext,
    ) -> List[Recommendation]:
        return self._pipeline.recommend(workload, context)
