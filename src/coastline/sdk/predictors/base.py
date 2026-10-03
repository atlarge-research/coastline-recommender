"""Base predictor interface."""

from abc import ABC, abstractmethod
from typing import Optional

from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.recommendation import Prediction
from coastline.sdk.models.workload import WorkloadSpec


class BasePredictor(ABC):
    """Base class for the performance and energy predictors."""

    @abstractmethod
    def predict(self, workload: WorkloadSpec, context: SystemContext) -> Optional[Prediction]:
        """Predict how a workload performs on its GPU configuration.

        Returns a Prediction, or None if the predictor cannot make one.
        """

    @abstractmethod
    def get_name(self) -> str:
        """Return the predictor name."""
