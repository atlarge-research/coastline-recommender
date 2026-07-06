"""Workload specification data model."""

from typing import Optional

from pydantic import BaseModel, Field, computed_field, field_validator


def canonical_model_name(name: str) -> str:
    """Canonicalize an LLM model id to the short key the predictors index on.

    Maps a HuggingFace-style id to its short key by dropping the org prefix and
    lowercasing: ``mistralai/Mistral-7B-v0.1`` -> ``mistral-7b-v0.1``. Idempotent on
    the short form (``mistral-7b-v0.1`` -> ``mistral-7b-v0.1``). This is NOT the
    family extractor (``split('/')[0]``); it keeps the full short key.
    """
    return str(name).split("/")[-1].lower()


class WorkloadSpec(BaseModel):
    """Specification for a training workload (LLM workload + optional infrastructure request)."""

    llm_model: str = Field(..., description="LLM model")
    fine_tuning_method: str = Field(..., description="Fine-tuning method (full, lora, qlora, etc.)")
    gpu_model: str = Field(..., description="GPU model")
    tokens_per_sample: int = Field(..., gt=0, description="Tokens per sample")
    batch_size: int = Field(..., gt=0, description="Batch size")
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
        description="Model name used for the AutoConf feasibility check ONLY. Set this "
        "to the real model when llm_model carries an anonymized/proxy name that the "
        "performance predictor (Kavier) requires but AutoConf does not recognize. "
        "Feasibility falls back to llm_model when this is unset.",
    )

    @field_validator("llm_model")
    @classmethod
    def _canonicalize_llm_model(cls, value: str) -> str:
        """Canonicalize the model id at ingestion so ALL consumers (Kavier physics,
        the SHA256 exact-match cache, AutoConf, and the ML predictors) see the short
        key they index on. A real HuggingFace id like ``mistralai/Mistral-7B-v0.1``
        becomes ``mistral-7b-v0.1``; the short form is unchanged (idempotent). The
        cache's stored keys are already short, so this aligns the lookup with them.
        """
        canonical = canonical_model_name(value)
        if not canonical.strip():
            raise ValueError("llm_model is empty after canonicalization")
        return canonical

    @computed_field  # type: ignore[prop-decorator]  # mypy: @computed_field over @property (pydantic idiom)
    @property
    def total_gpus(self) -> int:
        """Total GPUs across all nodes: gpus_per_node × number_of_nodes."""
        return (self.gpus_per_node or 1) * (self.number_of_nodes or 1)
