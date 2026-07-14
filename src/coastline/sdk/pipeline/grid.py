"""Candidate grid generation (batch_size × total_gpus); node layout auto-derived from total_gpus."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

from coastline.sdk.constants import DEFAULT_BATCH_SIZES
from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.workload import WorkloadSpec

logger = logging.getLogger(__name__)


def _powers_of_two(limit: int) -> List[int]:
    """Return [1, 2, 4, …] up to and including the largest power of 2 ≤ limit."""
    result = []
    g = 1
    while g <= limit:
        result.append(g)
        g *= 2
    return result


@dataclass(frozen=True)
class GridConfig:
    batch_sizes: List[int]
    total_gpus: List[int]
    top_k: int = 5


def grid_config_from_dict(config: Optional[dict], max_gpus: Optional[int] = None) -> GridConfig:
    grid = (config or {}).get("grid", {})
    if "total_gpus" in grid:
        gpu_list = list(grid["total_gpus"])
    elif max_gpus is not None:
        gpu_list = _powers_of_two(max_gpus)
    else:
        gpu_list = []
    return GridConfig(
        batch_sizes=list(grid.get("batch_sizes", DEFAULT_BATCH_SIZES)),
        total_gpus=gpu_list,
        top_k=int(grid.get("top_k", 5)),
    )


def _derive_node_layout(total_gpus: int, max_gpus_per_node: int) -> tuple[int, int]:
    """Return (gpus_per_node, number_of_nodes); packs GPUs per node to minimize inter-node comm."""
    gpus_per_node = min(total_gpus, max_gpus_per_node)
    number_of_nodes = math.ceil(total_gpus / gpus_per_node)
    return gpus_per_node, number_of_nodes


def generate_candidates(
    workload: WorkloadSpec,
    context: SystemContext,
    grid_config: GridConfig,
) -> List[WorkloadSpec]:
    """Build workload variants for each (batch_size, total_gpus) in the grid, clipped to context limits."""
    max_gpus = context.max_gpus
    max_gpus_per_node = context.constraints.gpus_per_node
    max_nodes = context.constraints.max_nodes

    gpu_steps = grid_config.total_gpus or _powers_of_two(max_gpus)

    candidates: List[WorkloadSpec] = []
    for n_gpus in gpu_steps:
        if n_gpus <= 0:
            # Non-positive GPU count: not runnable and would divide-by-zero in _derive_node_layout.
            logger.warning("Grid: skipping non-positive total_gpus=%s", n_gpus)
            continue
        if n_gpus > max_gpus:
            continue
        gpus_per_node, num_nodes = _derive_node_layout(n_gpus, max_gpus_per_node)
        if num_nodes > max_nodes:
            continue
        # A non-power-of-two step rounds its layout UP (e.g. 30 GPUs at 8/node -> 8x4 = 32),
        # so the actual layout can exceed the cap even when the requested step did not. Re-check
        # the derived total so a cluster budget is never overrun.
        if gpus_per_node * num_nodes > max_gpus:
            continue

        for batch_size in grid_config.batch_sizes:
            candidates.append(
                WorkloadSpec(
                    llm_model=workload.llm_model,
                    fine_tuning_method=workload.fine_tuning_method,
                    gpu_model=workload.gpu_model,
                    tokens_per_sample=workload.tokens_per_sample,
                    batch_size=batch_size,
                    gpus_per_node=gpus_per_node,
                    number_of_nodes=num_nodes,
                    torch_dtype=workload.torch_dtype,
                    enable_roce=workload.enable_roce,
                    feasibility_model=workload.feasibility_model,
                )
            )

    logger.info("Grid: %d candidates within context limits", len(candidates))
    return candidates
