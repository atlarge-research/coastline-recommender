"""Candidate grid (batch_size x total_gpus) and the min-GPU candidate sequence; the node layout is
derived from total_gpus."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from coastline.sdk.constants import DEFAULT_BATCH_SIZES, DEFAULT_MIN_GPU_TOP_K, DEFAULT_TOP_K, SelectionPolicy
from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.workload import WorkloadSpec

logger = logging.getLogger(__name__)


def _powers_of_two(limit: int) -> List[int]:
    """Return [1, 2, 4, ...] up to the largest power of 2 <= limit."""
    result = []
    g = 1
    while g <= limit:
        result.append(g)
        g *= 2
    return result


def check_top_k(top_k: int) -> None:
    """Raise ValueError unless ``top_k`` asks for at least one recommendation."""
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")


@dataclass(frozen=True)
class GridConfig:
    batch_sizes: List[int]
    total_gpus: List[int]
    # None when the config sets no top_k; see top_k_for.
    top_k: Optional[int] = None

    def __post_init__(self) -> None:
        # Otherwise the ranking would quietly turn a top_k below 1 into 1.
        if self.top_k is not None:
            check_top_k(self.top_k)

    def top_k_for(self, policy: object) -> int:
        """The number of configurations to return: the configured top_k, else 1 for min_gpu and
        DEFAULT_TOP_K for the weighted policies."""
        if self.top_k is not None:
            return self.top_k
        return DEFAULT_MIN_GPU_TOP_K if policy == SelectionPolicy.MIN_GPU else DEFAULT_TOP_K


def grid_config_from_dict(config: Optional[dict], max_gpus: Optional[int] = None) -> GridConfig:
    grid = (config or {}).get("grid", {})
    if "total_gpus" in grid:
        gpu_list = list(grid["total_gpus"])
    elif max_gpus is not None:
        gpu_list = _powers_of_two(max_gpus)
    else:
        gpu_list = []
    top_k = grid.get("top_k")
    return GridConfig(
        batch_sizes=list(grid.get("batch_sizes", DEFAULT_BATCH_SIZES)),
        total_gpus=gpu_list,
        top_k=None if top_k is None else int(top_k),
    )


def _derive_node_layout(total_gpus: int, max_gpus_per_node: int) -> tuple[int, int]:
    """Return (gpus_per_node, number_of_nodes) for exactly ``total_gpus`` GPUs.

    Uses the largest per-node count within the cap that divides the total, which gives the fewest
    nodes and the least inter-node traffic: 12 GPUs at up to 8 per node become 6 x 2 (8 x 2 would
    be 16). A power of two under a power-of-two cap fills whole nodes.
    """
    gpus_per_node = min(total_gpus, max_gpus_per_node)
    while total_gpus % gpus_per_node:
        gpus_per_node -= 1
    return gpus_per_node, total_gpus // gpus_per_node


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
    seen: set[tuple[int, int, int]] = set()
    for n_gpus in gpu_steps:
        if n_gpus <= 0:
            # Non-positive GPU count: not runnable and would divide-by-zero in _derive_node_layout.
            logger.warning("Grid: skipping non-positive total_gpus=%s", n_gpus)
            continue
        if n_gpus > max_gpus:
            continue
        # The layout uses exactly n_gpus, so it stays within max_gpus; only the node count can
        # rule it out (9 GPUs at <= 8 per node is 3 x 3, which needs three nodes).
        gpus_per_node, num_nodes = _derive_node_layout(n_gpus, max_gpus_per_node)
        if num_nodes > max_nodes:
            continue

        for batch_size in grid_config.batch_sizes:
            # A repeated grid entry would otherwise be scored and returned twice.
            key = (gpus_per_node, num_nodes, batch_size)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(_variant(workload, batch_size, gpus_per_node, num_nodes))

    logger.info("Grid: %d candidates within context limits", len(candidates))
    return candidates


def job_total_batch(workload: WorkloadSpec) -> int:
    """The job's total batch: its per-device batch times its GPUs (gpus_per_node x
    number_of_nodes, 1 when the workload gives no layout)."""
    return workload.batch_size * workload.total_gpus


def min_gpu_candidates(workload: WorkloadSpec, context: SystemContext) -> List[WorkloadSpec]:
    """The candidates of the thesis min-GPU algorithm, as in IBM AutoConf's min-GPU recommender, in
    the order it checks them.

    For g = 1, 2, 4, ... while g <= ``context.max_gpus``, the candidate is the workload with g GPUs
    in total and the job's total batch (:func:`job_total_batch`) split evenly over them. A g that
    does not divide the total, gives less than 1 per device, or needs more than ``max_nodes``
    nodes is skipped. The layout is the grid's exact one (fewest nodes within ``gpus_per_node``).
    """
    total_batch = job_total_batch(workload)
    max_gpus_per_node = context.constraints.gpus_per_node
    max_nodes = context.constraints.max_nodes
    candidates: List[WorkloadSpec] = []
    for n_gpus in _powers_of_two(context.max_gpus):
        if total_batch % n_gpus or total_batch // n_gpus < 1:
            continue
        gpus_per_node, num_nodes = _derive_node_layout(n_gpus, max_gpus_per_node)
        if num_nodes <= max_nodes:
            candidates.append(_variant(workload, total_batch // n_gpus, gpus_per_node, num_nodes))
    return candidates


def _variant(workload: WorkloadSpec, batch_size: int, gpus_per_node: int, num_nodes: int) -> WorkloadSpec:
    """The workload with the given per-device batch size and node layout."""
    return WorkloadSpec(
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
