"""GPU topology builder for OpenDC simulations (MSE power model)."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

GPU_TOPOLOGY_SPECS: dict[str, dict[str, Any]] = {
    "NVIDIA-A100-SXM4-80GB": {
        "core_count": 6912,
        "core_speed_mhz": 1410.0,
        "memory_bytes": 85_899_345_920,  # 80 GiB
        "power_watts": 400.0,
        "idle_power_watts": 75.0,
        "max_power_watts": 400.0,
    },
    "NVIDIA-A100-80GB-PCIe": {
        "core_count": 6912,
        "core_speed_mhz": 1410.0,
        "memory_bytes": 85_899_345_920,
        "power_watts": 300.0,
        "idle_power_watts": 60.0,
        "max_power_watts": 300.0,
    },
    "NVIDIA-H100-PCIe": {
        "core_count": 14592,
        "core_speed_mhz": 1620.0,
        "memory_bytes": 85_899_345_920,
        "power_watts": 350.0,
        "idle_power_watts": 70.0,
        "max_power_watts": 700.0,
    },
}

DEFAULT_CALIBRATION_FACTOR = 1.0


def build_topology(
    gpu_model: str,
    gpus_per_node: int,
    number_of_nodes: int,
    calibration_factor: float = DEFAULT_CALIBRATION_FACTOR,
) -> dict:
    """Build OpenDC topology dict for a GPU cluster. Raises ValueError for unknown gpu_model."""
    if gpu_model not in GPU_TOPOLOGY_SPECS:
        raise ValueError(
            f"Unsupported GPU model '{gpu_model}' for OpenDC topology. Supported: {list(GPU_TOPOLOGY_SPECS.keys())}"
        )

    spec = GPU_TOPOLOGY_SPECS[gpu_model]

    # Single cluster with one host representing the entire GPU fleet.
    # coreCount = total GPUs; coreSpeed = 1000 matches the workload export's
    # cpu_capacity (see kavier.sdk.io.training_opendc).  Power specs are scaled to
    # the full system so the MSE power model produces correct total power at
    # the GPU utilisation encoded in the fragments' cpu_usage.
    total_gpus = gpus_per_node * number_of_nodes
    host = {
        "name": gpu_model,
        "count": 1,
        "cpu": {
            "coreCount": total_gpus,
            "coreSpeed": 1000.0,
        },
        "memory": {
            "memorySize": spec["memory_bytes"] * total_gpus,
        },
        "cpuPowerModel": {
            "modelType": "mse",
            "power": spec["power_watts"] * total_gpus,
            "idlePower": spec["idle_power_watts"] * total_gpus,
            "maxPower": spec["max_power_watts"] * total_gpus,
            "calibrationFactor": calibration_factor,
        },
    }

    return {"clusters": [{"name": "cluster-0", "hosts": [host]}]}


def write_topology_json(topology: dict, path: Path) -> None:
    """Write topology dict to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(topology, f, indent=2)
    logger.debug("Wrote topology JSON to %s", path)
