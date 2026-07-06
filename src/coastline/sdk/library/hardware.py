"""GPU specs database for power prediction and the webapp (NVIDIA datasheets + DGX measurements)."""

from typing import Any, Optional

from coastline.sdk.exceptions import UnsupportedGPUError

# Source: NVIDIA datasheets and DGX measurements.
GPU_SPECS: dict[str, dict[str, Any]] = {
    "A100-SXM4-40GB": {
        "memory_gb": 40,
        "tdp_watts": 400,
        "idle_watts": 75,
        "compute_tflops_fp16": 312,
        "memory_bandwidth_gbps": 1555,  # HBM2
        "nvlink_bandwidth_gbps": 600,  # NVLink 3.0 per GPU
    },
    "A100-SXM4-80GB": {
        "memory_gb": 80,
        "tdp_watts": 400,
        "idle_watts": 75,
        "compute_tflops_fp16": 312,
        "memory_bandwidth_gbps": 2039,  # HBM2e
        "nvlink_bandwidth_gbps": 600,
    },
    "A100-PCIE-40GB": {
        "memory_gb": 40,
        "tdp_watts": 250,
        "idle_watts": 50,
        "compute_tflops_fp16": 312,
        "memory_bandwidth_gbps": 1555,
        "nvlink_bandwidth_gbps": 0,
    },
    "A100-PCIE-80GB": {
        "memory_gb": 80,
        "tdp_watts": 300,
        "idle_watts": 60,
        "compute_tflops_fp16": 312,
        "memory_bandwidth_gbps": 2039,
        "nvlink_bandwidth_gbps": 0,
    },
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


def _spec(gpu_model: str) -> dict[str, Any]:
    """Look up GPU specs; raises UnsupportedGPUError for unknown models (no silent defaults)."""
    specs = GPU_SPECS.get(gpu_model)
    if specs is None:
        raise UnsupportedGPUError(f"Unknown GPU model {gpu_model!r}. Known models: {sorted(GPU_SPECS)}")
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
    return GPU_SPECS.get(gpu_model)


def list_supported_gpus() -> list[str]:
    """All GPU model names with known specs."""
    return list(GPU_SPECS.keys())


# 75W/400W = 0.1875, rounded up to 0.25 for margin.
IDLE_POWER_RATIO = 0.25

# LogP comm-model params for the Kavier simulator (NVLink 3.0, DGX A100 / NCCL 2.x).
LOGP_LATENCY_US = 5.0  # end-to-end latency for small messages
LOGP_OVERHEAD_US = 2.0  # per-message processing overhead

# MFU efficiency-degradation exponents (Korthikanti et al. 2023, MLSys).
MFU_BATCH_ALPHA = 0.8  # batch size scaling exponent
MFU_SEQ_GAMMA = 0.9  # sequence length scaling exponent
MFU_MODEL_BETA = 0.85  # model size scaling exponent
