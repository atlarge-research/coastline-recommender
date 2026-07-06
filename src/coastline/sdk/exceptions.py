"""Custom exception types for the recommender system."""


class RecommenderSystemError(Exception):
    """Base exception for all recommender system errors."""


class PredictionError(RecommenderSystemError):
    """Raised when prediction fails."""


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


__all__ = [
    "RecommenderSystemError",
    "PredictionError",
    "ValidationError",
    "ConfigurationError",
    "DataLoadError",
    "ModelNotFoundError",
    "UnsupportedGPUError",
    "InsufficientMemoryError",
    "VerificationError",
    "RecommendationError",
]
