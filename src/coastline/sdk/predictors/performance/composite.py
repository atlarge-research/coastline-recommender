"""Composite performance predictors that cascade across simpler ones."""

from __future__ import annotations

from typing import Optional

from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.recommendation import Prediction
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.base import BasePredictor


class CacheThenSimulatePredictor(BasePredictor):
    """Return an exact cache match if there is one, else the fallback predictor's result.

    As pseudocode: ``if in database: retrieve() else: simulate(model=...)``. The fallback is any
    :class:`BasePredictor`: Kavier by default, or the model the config selects (for example an
    ML portfolio model).
    """

    @property
    def EXPENSIVE(self) -> bool:  # noqa: N802 (mirrors the class-level flag on plain predictors)
        """Follows the fallback: a cache hit is a hash lookup, a miss runs the fallback model.

        Forking helps grids that miss the cache; a grid that hits pays for a dispatch it did not
        need, which is cheap.
        """
        return bool(getattr(self._fallback, "EXPENSIVE", False))

    def __init__(self, cache: BasePredictor, fallback: BasePredictor):
        self._cache = cache
        self._fallback = fallback

    def predict(self, workload: WorkloadSpec, context: SystemContext) -> Optional[Prediction]:
        hit = self._cache.predict(workload, context)
        if hit is not None and hit.predicted_throughput and hit.predicted_throughput > 0:
            return hit
        return self._fallback.predict(workload, context)

    def get_name(self) -> str:
        return f"intelligent (cache->{self._fallback.get_name()})"
