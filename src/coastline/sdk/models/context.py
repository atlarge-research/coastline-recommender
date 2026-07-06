"""System context data model (available infrastructure)."""

from typing import Optional

from pydantic import BaseModel, Field

from coastline.sdk.library.hardware import get_gpu_memory


class Constraints(BaseModel):
    """Available infrastructure limits."""

    max_gpus: int = Field(..., ge=1, description="Maximum total GPUs available")
    gpus_per_node: int = Field(8, ge=1, description="GPUs per node")
    max_nodes: int = Field(16, ge=1, description="Maximum number of nodes")


class SystemContext(BaseModel):
    """Available infrastructure for context-aware recommendations."""

    available_gpu_models: list[str] = Field(..., description="Available GPU models")
    max_gpus: int = Field(..., ge=1, description="Max GPUs per job")
    gpu_memory: dict[str, int] = Field(..., description="GPU memory in GB")
    constraints: Constraints = Field(..., description="Infrastructure constraints")

    @classmethod
    def for_gpus(
        cls,
        gpu_models: list[str],
        *,
        max_gpus: int,
        gpus_per_node: int = 8,
        max_nodes: Optional[int] = None,
    ) -> "SystemContext":
        """Build a SystemContext from gpu_models; derives max_nodes when not given."""
        if max_nodes is None:
            max_nodes = max(1, -(-max_gpus // gpus_per_node))  # ceil division
        return cls(
            available_gpu_models=list(gpu_models),
            max_gpus=max_gpus,
            gpu_memory={m: get_gpu_memory(m) for m in gpu_models},
            constraints=Constraints(max_gpus=max_gpus, gpus_per_node=gpus_per_node, max_nodes=max_nodes),
        )
