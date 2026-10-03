"""GPU specs for power prediction and the web app (NVIDIA datasheets and DGX measurements).

The accepted GPU names are the names in Kavier's GPU catalog, plus the aliases below, so every
GPU a context accepts can also be predicted.
"""

from typing import Any, Optional

from coastline.sdk.exceptions import UnsupportedGPUError

try:
    from kavier.sdk.library import GPU_SPEC_LIBRARY as _KAVIER_GPUS
except ImportError:  # Kavier is a core dependency; without it only GPU_SPECS is known
    _KAVIER_GPUS = {}

# Coastline spellings of GPUs that Kavier's catalog names differently (same part, same specs).
# They are resolved when a workload or a context is built, before any prediction.
GPU_ALIASES: dict[str, str] = {
    "A100-SXM4-80GB": "NVIDIA-A100-SXM4-80GB",
    "A100-PCIE-80GB": "NVIDIA-A100-80GB-PCIe",
    "A100-PCIE-40GB": "A100-40GB",
}

# Source: NVIDIA datasheets and DGX measurements, for the GPUs in the traces. Other GPUs in
# Kavier's catalog take their values from Kavier (see _spec).
GPU_SPECS: dict[str, dict[str, Any]] = {
    # Names as they appear in the dataset / Kavier libraries / web UI.
    "NVIDIA-A100-SXM4-80GB": {
        "memory_gb": 80,
        "tdp_watts": 400,
        "idle_watts": 75,
        "compute_tflops_fp16": 312,
        "memory_bandwidth_gbps": 2039,
        "nvlink_bandwidth_gbps": 600,
    },
    "NVIDIA-A100-80GB-PCIe": {
        "memory_gb": 80,
        "tdp_watts": 300,
        "idle_watts": 60,
        "compute_tflops_fp16": 312,
        "memory_bandwidth_gbps": 2039,
        "nvlink_bandwidth_gbps": 0,
    },
    "L40S": {
        "memory_gb": 48,
        "tdp_watts": 350,
        "idle_watts": 40,
        "compute_tflops_fp16": 362,
        "memory_bandwidth_gbps": 864,
        "nvlink_bandwidth_gbps": 0,
    },
    "NVIDIA-H100-PCIe": {
        "memory_gb": 80,
        "tdp_watts": 350,
        "idle_watts": 50,
        "compute_tflops_fp16": 756,
        "memory_bandwidth_gbps": 2000,
        "nvlink_bandwidth_gbps": 0,
    },
}


def canonical_gpu_name(gpu_model: str) -> str:
    """Kavier's name for a GPU: an alias is mapped, any other name is returned unchanged."""
    return GPU_ALIASES.get(gpu_model, gpu_model)


def _kavier_spec(gpu_model: str) -> Optional[dict[str, Any]]:
    """Specs of a GPU that only Kavier's catalog lists, in the GPU_SPECS layout."""
    spec = _KAVIER_GPUS.get(gpu_model)
    if spec is None:
        return None
    return {
        "memory_gb": spec.memory_gb,
        "tdp_watts": spec.max_power_w,
        "idle_watts": spec.idle_power_w,
        "compute_tflops_fp16": spec.fp_16_tensor_core_tflops,
        "memory_bandwidth_gbps": spec.bandwidth_bps / 1e9,
    }


def _lookup(gpu_model: str) -> Optional[dict[str, Any]]:
    name = canonical_gpu_name(gpu_model)
    return GPU_SPECS.get(name) or _kavier_spec(name)


def _spec(gpu_model: str) -> dict[str, Any]:
    """Look up GPU specs; raise UnsupportedGPUError for an unknown model."""
    specs = _lookup(gpu_model)
    if specs is None:
        aliases = ", ".join(f"{alias} (same as {name})" for alias, name in GPU_ALIASES.items())
        raise UnsupportedGPUError(
            f"Unknown GPU model {gpu_model!r}. Known models: {list_supported_gpus()}. Also accepted: {aliases}."
        )
    return specs


def get_gpu_memory(gpu_model: str) -> int:
    """GPU memory in GB. Raises UnsupportedGPUError for unknown models."""
    return int(_spec(gpu_model)["memory_gb"])


def get_gpu_tdp(gpu_model: str) -> float:
    """GPU Thermal Design Power in watts. Raises UnsupportedGPUError for unknown models."""
    return float(_spec(gpu_model)["tdp_watts"])


def get_gpu_idle_power(gpu_model: str) -> float:
    """GPU idle power in watts. Raises UnsupportedGPUError for unknown models."""
    return float(_spec(gpu_model)["idle_watts"])


def get_gpu_specs(gpu_model: str) -> Optional[dict[str, Any]]:
    """Complete GPU spec dict, or None if the model is unknown."""
    return _lookup(gpu_model)


def list_supported_gpus() -> list[str]:
    """All GPU model names with known specs (aliases not included)."""
    return sorted({*GPU_SPECS, *_KAVIER_GPUS})


# 75W/400W = 0.1875, rounded up to 0.25 for margin.
IDLE_POWER_RATIO = 0.25

# LogP comm-model params for the Kavier simulator (NVLink 3.0, DGX A100 / NCCL 2.x).
LOGP_LATENCY_US = 5.0  # end-to-end latency for small messages
LOGP_OVERHEAD_US = 2.0  # per-message processing overhead

# MFU efficiency-degradation exponents (Korthikanti et al. 2023, MLSys).
MFU_BATCH_ALPHA = 0.8  # batch size scaling exponent
MFU_SEQ_GAMMA = 0.9  # sequence length scaling exponent
MFU_MODEL_BETA = 0.85  # model size scaling exponent
