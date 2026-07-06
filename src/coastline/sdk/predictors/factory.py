"""Default predictor constructors for the orchestrator."""

import logging

from coastline.sdk.predictors.base import BasePredictor

logger = logging.getLogger(__name__)


def create_physics_driven() -> BasePredictor:
    """Return the Kavier physics-based predictor."""
    from coastline.sdk.predictors.performance.physics import KavierPredictor

    logger.info("Using KavierPredictor (physics-driven)")
    return KavierPredictor()
