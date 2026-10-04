"""Min-GPU strategy: the thesis min-GPU algorithm, as in IBM AutoConf's min-GPU recommender.

The job's total batch (per-device batch x its GPUs, 1 GPU without a layout) is kept. For g = 1, 2,
4, ... up to the context's maximum GPUs, the workload with g GPUs and that total split evenly over
them is checked for feasibility; the first feasible one is returned. Only the returned
configurations are simulated. See ``GridWorkflowPipeline._recommend_min_gpu``.
"""

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
    """Minimum-GPU policy: the first feasible GPU count in 1, 2, 4, ... for the job's total batch."""

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
        logger.info("MinGPUStrategy: first feasible GPU count in 1, 2, 4, ... (policy=min_gpu)")

    def recommend(
        self,
        workload: WorkloadSpec,
        context: SystemContext,
    ) -> list[Recommendation]:
        return self._pipeline.recommend(workload, context)

    def get_name(self) -> str:
        return Strategy.MIN_GPU.value
