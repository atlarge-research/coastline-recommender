"""Shared workload-validation helpers used by the predictors."""

from coastline.sdk.exceptions import UnsupportedGPUError
from coastline.sdk.library.hardware import get_gpu_memory
from coastline.sdk.models.workload import WorkloadSpec


def estimate_memory_requirement(workload: WorkloadSpec) -> float:
    """Roughly estimate GPU memory (GB) for a workload.

    Deliberately simple: a base 0.5 GB/sample scaled by batch size and
    sequence length (normalized to 1K tokens), then tripled to cover
    params + gradients + Adam optimizer state. Real usage also depends on
    gradient accumulation, activation checkpointing and mixed precision.
    """
    base_memory_per_sample_gb = 0.5
    memory_gb = base_memory_per_sample_gb * workload.batch_size * (workload.tokens_per_sample / 1024.0)
    # params + gradients + optimizer state (Adam: 2x params).
    return memory_gb * 3.0


def validate_gpu_memory(workload: WorkloadSpec) -> tuple[bool, str]:
    """Check the workload fits in GPU memory.

    Returns ``(is_valid, error_message)``; the message is empty when valid.
    """
    required_gb = estimate_memory_requirement(workload)
    try:
        available_gb = get_gpu_memory(workload.gpu_model)
    except UnsupportedGPUError as exc:
        return False, str(exc)

    if required_gb > available_gb:
        return False, (
            f"Estimated memory requirement ({required_gb:.1f}GB) exceeds "
            f"available GPU memory ({available_gb}GB on {workload.gpu_model})"
        )

    return True, ""


def validate_workload(workload: WorkloadSpec) -> tuple[bool, str]:
    """Validate a workload's GPU memory (batch_size / tokens_per_sample are already
    enforced > 0 by WorkloadSpec). Returns ``(is_valid, error_message)``."""
    return validate_gpu_memory(workload)
