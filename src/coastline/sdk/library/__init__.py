"""GPU/LLM hardware spec library + physics constants."""

from coastline.sdk.library.hardware import (
    GPU_SPECS,
    IDLE_POWER_RATIO,
    LOGP_LATENCY_US,
    LOGP_OVERHEAD_US,
    MFU_BATCH_ALPHA,
    MFU_MODEL_BETA,
    MFU_SEQ_GAMMA,
    get_gpu_idle_power,
    get_gpu_memory,
    get_gpu_specs,
    get_gpu_tdp,
    list_supported_gpus,
)

__all__ = [
    "GPU_SPECS",
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
