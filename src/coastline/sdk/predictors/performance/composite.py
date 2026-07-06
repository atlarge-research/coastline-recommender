"""Composite performance predictors that cascade across simpler ones."""

from __future__ import annotations

from typing import Optional

from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.recommendation import Prediction
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.base import BasePredictor


class CacheThenPhysicsPredictor(BasePredictor):
    """Exact cache match first, else the Kavier analytical predictor."""

    def __init__(self, cache: BasePredictor, physics: BasePredictor):
        self._cache = cache
        self._physics = physics

    def predict(self, workload: WorkloadSpec, context: SystemContext) -> Optional[Prediction]:
        hit = self._cache.predict(workload, context)
        if hit is not None and hit.predicted_throughput and hit.predicted_throughput > 0:
            return hit
        return self._physics.predict(workload, context)

    def get_name(self) -> str:
        return "intelligent (cache→kavier)"
