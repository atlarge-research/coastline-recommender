"""Base predictor interface."""

from abc import ABC, abstractmethod
from typing import Optional

from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.recommendation import Prediction
from coastline.sdk.models.workload import WorkloadSpec


class BasePredictor(ABC):
    """Base class for performance predictors."""

    @abstractmethod
    def predict(self, workload: WorkloadSpec, context: SystemContext) -> Optional[Prediction]:
        """Predict a GPU configuration for a workload.

        Returns a Prediction, or None if the predictor cannot make one.
        """

    @abstractmethod
    def get_name(self) -> str:
        """Get the predictor name."""
