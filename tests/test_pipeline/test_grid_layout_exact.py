"""Grid generation: every GPU step gets an exact layout, and no candidate appears twice.

Rounding up to whole nodes would turn 12 GPUs at 8 per node into 8 x 2 = 16, which either
duplicates the 16-GPU step or is dropped by a 12-GPU budget.
"""

from __future__ import annotations

import pytest

from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.pipeline.grid import GridConfig, _derive_node_layout, generate_candidates, grid_config_from_dict


def _workload() -> WorkloadSpec:
    return WorkloadSpec(
        llm_model="mistral-7b-v0.1",
        fine_tuning_method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=8,
    )


def _context(max_gpus: int, gpus_per_node: int = 8, max_nodes: int = 4) -> SystemContext:
    return SystemContext(
        available_gpu_models=["NVIDIA-A100-SXM4-80GB"],
        max_gpus=max_gpus,
        gpu_memory={"NVIDIA-A100-SXM4-80GB": 80},
        constraints=Constraints(max_gpus=max_gpus, gpus_per_node=gpus_per_node, max_nodes=max_nodes),
    )


@pytest.mark.parametrize(
    "total_gpus,max_per_node,expected",
    [
        (12, 8, (6, 2)),
        (20, 8, (5, 4)),
        (9, 8, (3, 3)),
        (24, 8, (8, 3)),
        (10, 4, (2, 5)),
        (7, 4, (1, 7)),
    ],
)
def test_layout_is_exact_with_the_fewest_nodes(total_gpus, max_per_node, expected):
    gpus_per_node, nodes = _derive_node_layout(total_gpus, max_per_node)
    assert (gpus_per_node, nodes) == expected
    assert gpus_per_node * nodes == total_gpus
    assert gpus_per_node <= max_per_node
    # Fewest nodes: no exact layout fits on fewer nodes within the per-node cap.
    assert not any(total_gpus % n == 0 and total_gpus // n <= max_per_node for n in range(1, nodes))


@pytest.mark.parametrize("total_gpus", [1, 2, 4, 8, 16, 32, 64, 128, 256])
def test_power_of_two_steps_keep_their_layout(total_gpus):
    # The default grid: powers of two at 8 GPUs per node fill whole nodes.
    assert _derive_node_layout(total_gpus, 8) == (min(total_gpus, 8), max(1, total_gpus // 8))


def test_a_twelve_gpu_budget_admits_the_twelve_gpu_step():
    ctx = _context(max_gpus=12, max_nodes=2)
    cands = generate_candidates(_workload(), ctx, GridConfig(batch_sizes=[32], total_gpus=[12]))
    assert [(c.gpus_per_node, c.number_of_nodes, c.total_gpus) for c in cands] == [(6, 2, 12)]


def test_steps_never_collapse_onto_the_same_layout():
    ctx = _context(max_gpus=16, max_nodes=2)
    cands = generate_candidates(_workload(), ctx, GridConfig(batch_sizes=[32], total_gpus=[12, 16]))
    assert [(c.gpus_per_node, c.number_of_nodes, c.total_gpus) for c in cands] == [(6, 2, 12), (8, 2, 16)]


def test_a_layout_needing_too_many_nodes_is_skipped():
    # 9 GPUs at <= 8 per node is exactly 3 x 3, which needs three nodes; a two-node cluster skips it.
    ctx = _context(max_gpus=16, max_nodes=2)
    cands = generate_candidates(_workload(), ctx, GridConfig(batch_sizes=[8], total_gpus=[9, 8]))
    assert [(c.gpus_per_node, c.number_of_nodes) for c in cands] == [(8, 1)]


def test_repeated_grid_entries_give_one_candidate_each_in_first_seen_order():
    ctx = _context(max_gpus=16, max_nodes=2)
    grid = GridConfig(batch_sizes=[16, 8, 16], total_gpus=[8, 4, 8])
    cands = generate_candidates(_workload(), ctx, grid)
    assert [(c.total_gpus, c.batch_size) for c in cands] == [(8, 16), (8, 8), (4, 16), (4, 8)]


@pytest.mark.parametrize("top_k", [0, -1])
def test_top_k_below_one_is_rejected(top_k):
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        GridConfig(batch_sizes=[8], total_gpus=[1], top_k=top_k)
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        grid_config_from_dict({"grid": {"top_k": top_k}})
