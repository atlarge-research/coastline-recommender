"""Base strategy interface."""

from abc import ABC, abstractmethod
from typing import List

from coastline.sdk.models import Recommendation, SystemContext, WorkloadSpec


class BaseStrategy(ABC):
    """Base class for recommendation policies."""

    @abstractmethod
    def recommend(self, workload: WorkloadSpec, context: SystemContext) -> List[Recommendation]:
        """Return recommendations sorted best-first; empty list if none feasible."""

    @abstractmethod
    def get_name(self) -> str:
        """Return the strategy name."""
