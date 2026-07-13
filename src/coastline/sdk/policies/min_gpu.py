"""MinGPU Strategy — grid + feasibility + simulate, pick minimum feasible GPUs."""

import logging
from typing import Optional

from coastline.sdk.constants import SelectionPolicy, Strategy
from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.recommendation import Recommendation
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.pipeline.workflow import GridWorkflowPipeline
from coastline.sdk.policies.base import BaseStrategy
from coastline.sdk.predictors.base import BasePredictor

logger = logging.getLogger(__name__)


class MinGPUStrategy(BaseStrategy):
    """Minimum-GPU policy: grid → feasibility → simulate → pick min total_gpus."""

    def __init__(
        self,
        pipeline: Optional[GridWorkflowPipeline] = None,
        *,
        config: Optional[dict] = None,
        throughput_predictor: Optional[BasePredictor] = None,
        power_predictor: Optional[BasePredictor] = None,
    ):
        super().__init__()
        if pipeline is not None:
            self._pipeline = pipeline
        else:
            self._pipeline = GridWorkflowPipeline.from_config(
                config=config or {},
                selection_policy=SelectionPolicy.MIN_GPU.value,
                strategy_name=Strategy.MIN_GPU.value,
                throughput_predictor=throughput_predictor,
                power_predictor=power_predictor,
            )
        logger.info("MinGPUStrategy using unified grid workflow (policy=min_gpu)")

    def recommend(
        self,
        workload: WorkloadSpec,
        context: SystemContext,
    ) -> list[Recommendation]:
        return self._pipeline.recommend(workload, context)

    def get_name(self) -> str:
        return Strategy.MIN_GPU.value
