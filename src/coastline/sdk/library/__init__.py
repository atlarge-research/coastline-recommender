"""GPU and LLM hardware specs and physics constants."""

from coastline.sdk.library.hardware import (
    GPU_ALIASES,
    GPU_SPECS,
    IDLE_POWER_RATIO,
    LOGP_LATENCY_US,
    LOGP_OVERHEAD_US,
    MFU_BATCH_ALPHA,
    MFU_MODEL_BETA,
    MFU_SEQ_GAMMA,
    canonical_gpu_name,
    get_gpu_idle_power,
    get_gpu_memory,
    get_gpu_specs,
    get_gpu_tdp,
    list_supported_gpus,
)
from coastline.sdk.library.llm_names import LLM_ALIASES, kavier_llm_name

__all__ = [
    "GPU_ALIASES",
    "GPU_SPECS",
    "canonical_gpu_name",
    "LLM_ALIASES",
    "kavier_llm_name",
    "get_gpu_memory",
    "get_gpu_tdp",
    "get_gpu_idle_power",
    "get_gpu_specs",
    "list_supported_gpus",
    "IDLE_POWER_RATIO",
    "LOGP_LATENCY_US",
    "LOGP_OVERHEAD_US",
    "MFU_BATCH_ALPHA",
    "MFU_SEQ_GAMMA",
    "MFU_MODEL_BETA",
]
