"""Composite performance predictors that cascade across simpler ones."""

from __future__ import annotations

from typing import Optional

from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.recommendation import Prediction
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.base import BasePredictor


class CacheThenSimulatePredictor(BasePredictor):
    """Exact cache match first, else a simulation predictor.

    The shape is ``if in database: retrieve() else: simulate(model=...)``. The fallback is a
    plain :class:`BasePredictor` — Kavier physics by default, or any user-selected model (an ML
    portfolio model, …) — so a cache miss simulates with whatever the config chose.
    """

    def __init__(self, cache: BasePredictor, fallback: BasePredictor):
        self._cache = cache
        self._fallback = fallback

    def predict(self, workload: WorkloadSpec, context: SystemContext) -> Optional[Prediction]:
        hit = self._cache.predict(workload, context)
        if hit is not None and hit.predicted_throughput and hit.predicted_throughput > 0:
            return hit
        return self._fallback.predict(workload, context)

    def get_name(self) -> str:
        return f"intelligent (cache→{self._fallback.get_name()})"
