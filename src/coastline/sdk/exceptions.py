"""Custom exception types for the recommender system."""


class RecommenderSystemError(Exception):
    """Base exception for all recommender system errors."""


class PredictionError(RecommenderSystemError):
    """Raised when prediction fails."""


class NoPredictionError(PredictionError, RuntimeError):
    """Raised when candidates pass the feasibility check but the predictor gives no usable
    number for any of them. Also a RuntimeError, so code that catches the pipeline's
    'no feasible candidates' RuntimeError still catches this case."""


class ValidationError(RecommenderSystemError):
    """Raised when workload or context validation fails."""


class ConfigurationError(RecommenderSystemError):
    """Raised when configuration is invalid or missing."""


class DataLoadError(RecommenderSystemError):
    """Raised when data loading fails."""


class ModelNotFoundError(RecommenderSystemError):
    """Raised when a required model file is not found."""


class UnsupportedGPUError(RecommenderSystemError):
    """Raised when an unsupported GPU model is specified."""


class InsufficientMemoryError(RecommenderSystemError):
    """Raised when workload requires more GPU memory than available."""


class VerificationError(RecommenderSystemError):
    """Raised when verification fails."""


class RecommendationError(RecommenderSystemError):
    """Raised when recommendation generation fails."""


class NoFeasibleGPUCountError(RecommendationError, RuntimeError):
    """Raised when min_gpu finds no GPU count at which the job's total batch passes the
    feasibility check. Also a RuntimeError, like the grid's 'no feasible candidates' error.

    ``total_batch`` is the job's total batch and ``gpu_counts`` the GPU counts that were checked.
    """

    def __init__(self, strategy_name: str, total_batch: int, gpu_counts: list[int]) -> None:
        self.strategy_name = strategy_name
        self.total_batch = total_batch
        self.gpu_counts = list(gpu_counts)
        super().__init__(
            f"Workflow ({strategy_name}): no feasible GPU count: a total batch of {total_batch} fails the "
            f"feasibility check on {self.gpu_counts_text()}. Check the feasibility settings and the "
            f"cluster size, or try a smaller batch."
        )

    def __reduce__(self) -> tuple[type["NoFeasibleGPUCountError"], tuple[str, int, list[int]]]:
        # The constructor takes three arguments, so a copy or a pickle has to pass them all.
        return (type(self), (self.strategy_name, self.total_batch, self.gpu_counts))

    def gpu_counts_text(self) -> str:
        """The checked GPU counts as text with the unit, for example '1, 2, 4 and 8 GPUs' or '1 GPU'."""
        counts = [str(count) for count in self.gpu_counts]
        listed = counts[0] if len(counts) == 1 else f"{', '.join(counts[:-1])} and {counts[-1]}"
        return f"{listed} GPU" if self.gpu_counts[-1] == 1 else f"{listed} GPUs"


__all__ = [
    "RecommenderSystemError",
    "PredictionError",
    "NoPredictionError",
    "ValidationError",
    "ConfigurationError",
    "DataLoadError",
    "ModelNotFoundError",
    "UnsupportedGPUError",
    "InsufficientMemoryError",
    "VerificationError",
    "RecommendationError",
    "NoFeasibleGPUCountError",
]
