"""Recommendation data models."""

from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


def round_floats_for_display(obj: Any, ndigits: int = 2) -> Any:
    """Return a copy of ``obj`` with every float rounded to ``ndigits`` decimals.

    For display (JSON output and ``__str__``): recurses into dicts and lists and leaves ints,
    bools, strings and other values as they are. Builds new containers, so the caller's data
    (such as a Recommendation's ``metadata``) is not changed.
    """
    if isinstance(obj, bool):  # bool is a subclass of int; keep True/False as-is
        return obj
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_floats_for_display(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(round_floats_for_display(v, ndigits) for v in obj)
    return obj


def _ensure_total_gpus_consistent(gpus_per_node: int, number_of_nodes: int, total_gpus: int) -> None:
    """Raise ValueError unless total_gpus == gpus_per_node * number_of_nodes."""
    expected = gpus_per_node * number_of_nodes
    if total_gpus != expected:
        raise ValueError(f"total_gpus ({total_gpus}) must equal gpus_per_node * number_of_nodes ({expected})")


class Prediction(BaseModel):
    """Raw prediction from a predictor."""

    gpus_per_node: int = Field(..., description="GPUs per node")
    number_of_nodes: int = Field(..., description="Number of nodes")
    total_gpus: int = Field(..., description="Total GPUs")
    predicted_throughput: Optional[float] = Field(default=None, description="tokens/sec")
    predicted_runtime_seconds: Optional[float] = Field(default=None, description="seconds")
    predicted_power: Optional[float] = Field(default=None, description="watts")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_total_gpus(self) -> "Prediction":
        _ensure_total_gpus_consistent(self.gpus_per_node, self.number_of_nodes, self.total_gpus)
        return self


class Recommendation(BaseModel):
    """A GPU configuration recommendation."""

    gpus_per_node: int = Field(..., description="GPUs per node")
    number_of_nodes: int = Field(..., description="Number of nodes")
    total_gpus: int = Field(..., description="Total GPUs")
    strategy: str = Field(..., description="Strategy name")
    predicted_throughput: Optional[float] = Field(None, description="tokens/sec")
    predicted_runtime_seconds: Optional[float] = Field(None, description="seconds")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_total_gpus(self) -> "Recommendation":
        _ensure_total_gpus_consistent(self.gpus_per_node, self.number_of_nodes, self.total_gpus)
        return self

    def __str__(self) -> str:
        """Display string with floats rounded to 2 decimals.

        Same ``field=value`` layout as pydantic's default, with long floats (throughput, power,
        scores) shortened for ``print(rec)``. ``__repr__`` and the stored values keep full precision.
        """
        return " ".join(f"{name}={round_floats_for_display(getattr(self, name))!r}" for name in type(self).model_fields)
