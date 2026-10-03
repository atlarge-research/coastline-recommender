"""Workload specification data model."""

from typing import Optional

from pydantic import BaseModel, Field, computed_field, field_validator

from coastline.sdk.library.hardware import canonical_gpu_name


def canonical_model_name(name: str) -> str:
    """Canonicalize an LLM model id to the short key the predictors index on.

    Drops the org prefix and lowercases: ``mistralai/Mistral-7B-v0.1`` -> ``mistral-7b-v0.1``.
    A short key is returned unchanged.
    """
    return str(name).split("/")[-1].lower()


def canonical_method_name(name: str) -> str:
    """Lowercase a fine-tuning method, the spelling Kavier, AutoConf and the traces use."""
    return str(name).strip().lower()


class WorkloadSpec(BaseModel):
    """Specification for a training workload (LLM workload + optional infrastructure request)."""

    llm_model: str = Field(..., description="LLM model")
    fine_tuning_method: str = Field(..., description="Fine-tuning method (full, lora, qlora, etc.)")
    gpu_model: str = Field(..., description="GPU model")
    tokens_per_sample: int = Field(..., gt=0, description="Tokens per sample")
    batch_size: int = Field(
        ...,
        gt=0,
        description="Per-device (per-GPU) train batch size. Effective/global batch = "
        "batch_size * gpus_per_node * number_of_nodes. Kavier consumes it as per-device; the "
        "AutoConf feasibility boundary converts it to the effective batch.",
    )
    gpus_per_node: Optional[int] = Field(None, ge=1, description="GPUs per node")
    number_of_nodes: Optional[int] = Field(None, ge=1, description="Number of nodes")
    torch_dtype: Optional[str] = Field(
        None,
        description="Training dtype hint (e.g. bfloat16); optional for backward compatibility",
    )
    enable_roce: Optional[bool] = Field(
        None,
        description="Whether RoCE networking was enabled for the workload (optional)",
    )
    feasibility_model: Optional[str] = Field(
        None,
        description="Model name used only for the AutoConf feasibility check. Set this "
        "to the real model when llm_model carries an anonymized/proxy name that the "
        "performance predictor (Kavier) requires but AutoConf does not recognize. "
        "Feasibility falls back to llm_model when this is unset.",
    )

    @field_validator("llm_model")
    @classmethod
    def _canonicalize_llm_model(cls, value: str) -> str:
        """Store the short model key that every consumer (Kavier, the SHA256 exact-match cache,
        AutoConf, the ML predictors) indexes on: ``mistralai/Mistral-7B-v0.1`` becomes
        ``mistral-7b-v0.1``, and a short key stays as it is. The cache stores short keys too.
        """
        canonical = canonical_model_name(value)
        if not canonical.strip():
            raise ValueError("llm_model is empty after canonicalization")
        return canonical

    @field_validator("feasibility_model")
    @classmethod
    def _canonicalize_feasibility_model(cls, value: Optional[str]) -> Optional[str]:
        """Same canonical form as llm_model, so AutoConf gives one verdict per model however it is
        spelled ('mistralai/Mistral-7B-v0.1' and 'mistral-7b-v0.1' are one model)."""
        return None if value is None else canonical_model_name(value)

    @field_validator("fine_tuning_method")
    @classmethod
    def _canonicalize_fine_tuning_method(cls, value: str) -> str:
        """'LoRA' is 'lora' for every consumer; Kavier rejects any other case."""
        return canonical_method_name(value)

    @field_validator("gpu_model")
    @classmethod
    def _canonicalize_gpu_model(cls, value: str) -> str:
        """A Coastline GPU alias becomes Kavier's name for the same part (see GPU_ALIASES)."""
        return canonical_gpu_name(value)

    @computed_field  # type: ignore[prop-decorator]  # mypy: @computed_field over @property (pydantic idiom)
    @property
    def total_gpus(self) -> int:
        """Total GPUs across all nodes: gpus_per_node * number_of_nodes."""
        return (self.gpus_per_node or 1) * (self.number_of_nodes or 1)
