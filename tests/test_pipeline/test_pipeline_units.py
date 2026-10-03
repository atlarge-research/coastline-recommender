"""Unit tests for the pipeline building blocks.

Covers RulesFeasibilityChecker and the grid helpers _powers_of_two, _derive_node_layout,
grid_config_from_dict and generate_candidates. No model or data file is loaded.
"""

from __future__ import annotations

import pytest

from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.pipeline.grid import (
    DEFAULT_BATCH_SIZES,
    GridConfig,
    _derive_node_layout,
    _powers_of_two,
    generate_candidates,
    grid_config_from_dict,
)
from coastline.sdk.predictors.feasibility.autoconf import (
    RulesFeasibilityChecker,
)


def _workload(batch_size: int = 8, gpus_per_node=None, number_of_nodes=None) -> WorkloadSpec:
    return WorkloadSpec(
        llm_model="mistral-7b-v0.1",
        fine_tuning_method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=batch_size,
        gpus_per_node=gpus_per_node,
        number_of_nodes=number_of_nodes,
    )


def _context(max_gpus: int = 16, gpus_per_node: int = 8, max_nodes: int = 2) -> SystemContext:
    return SystemContext(
        available_gpu_models=["NVIDIA-A100-SXM4-80GB"],
        max_gpus=max_gpus,
        gpu_memory={"NVIDIA-A100-SXM4-80GB": 80},
        constraints=Constraints(
            max_gpus=max_gpus,
            gpus_per_node=gpus_per_node,
            max_nodes=max_nodes,
        ),
    )


# RulesFeasibilityChecker: per-device sanity checks (no divisibility rule)
class TestRulesFeasibility:
    def test_valid_per_device_batch_is_feasible(self):
        # A valid per-device workload on 4 GPUs is feasible.
        wl = _workload(batch_size=8, gpus_per_node=4, number_of_nodes=1)
        assert wl.total_gpus == 4
        ok, meta = RulesFeasibilityChecker().is_feasible(wl)
        assert ok is True
        assert meta == {}

    def test_non_divisible_batch_is_feasible(self):
        # batch_size is per device and need not divide the GPU count: batch 8 on 3 GPUs is feasible.
        wl = _workload(batch_size=8, gpus_per_node=3, number_of_nodes=1)
        assert wl.total_gpus == 3
        ok, meta = RulesFeasibilityChecker().is_feasible(wl)
        assert ok is True
        assert meta == {}

    def test_per_device_batch_feasible_regardless_of_total_gpus(self):
        # A per-device batch of 8 is feasible on 16 GPUs too.
        wl = _workload(batch_size=8, gpus_per_node=8, number_of_nodes=2)
        assert wl.total_gpus == 16  # 8 * 2
        ok, meta = RulesFeasibilityChecker().is_feasible(wl)
        assert ok is True
        assert meta == {}

    def test_default_layout_single_gpu_always_divisible(self):
        # Without a layout total_gpus is 1.
        wl = _workload(batch_size=7)
        assert wl.total_gpus == 1
        ok, _ = RulesFeasibilityChecker().is_feasible(wl)
        assert ok is True


class TestPowersOfTwo:
    @pytest.mark.parametrize(
        "limit, expected",
        [
            (1, [1]),
            (2, [1, 2]),
            (8, [1, 2, 4, 8]),
            (16, [1, 2, 4, 8, 16]),
            (10, [1, 2, 4, 8]),  # stops below a limit that is not a power of two
            (0, []),  # nothing fits under 1
        ],
    )
    def test_powers_of_two(self, limit, expected):
        assert _powers_of_two(limit) == expected


# _derive_node_layout - exact layout, fewest nodes
class TestDeriveNodeLayout:
    @pytest.mark.parametrize(
        "total_gpus, max_per_node, expected",
        [
            (1, 8, (1, 1)),  # one GPU on one node
            (8, 8, (8, 1)),  # fills one node
            (16, 8, (8, 2)),  # two full nodes
            (12, 8, (6, 2)),  # 6 per node on 2 nodes (8 x 2 would be 16)
            (3, 8, (3, 1)),  # fits on one node
            (9, 8, (3, 3)),  # one over a node: the only exact split within 8 per node is 3 x 3
            (5, 4, (1, 5)),  # smaller node cap: 5 is prime, so one GPU per node
        ],
    )
    def test_layout(self, total_gpus, max_per_node, expected):
        gpus_per_node, num_nodes = _derive_node_layout(total_gpus, max_per_node)
        assert (gpus_per_node, num_nodes) == expected
        # the layout covers the requested total
        assert gpus_per_node * num_nodes >= total_gpus
        # and one node fewer would not
        assert (num_nodes - 1) * gpus_per_node < total_gpus
        # no node holds more than the cap
        assert gpus_per_node <= max_per_node

    def test_zero_total_gpus_raises_zero_division(self):
        # total_gpus=0 gives gpus_per_node=0 and a modulo by zero. Callers skip non-positive
        # counts before this point.
        with pytest.raises(ZeroDivisionError):
            _derive_node_layout(0, 8)


class TestGridConfigFromDict:
    def test_explicit_total_gpus_list_preserved(self):
        gc = grid_config_from_dict({"grid": {"total_gpus": [2, 4], "batch_sizes": [8, 16], "top_k": 5}})
        assert gc.total_gpus == [2, 4]
        assert gc.batch_sizes == [8, 16]
        assert gc.top_k == 5

    def test_total_gpus_derived_from_max_gpus_when_absent(self):
        # Without a list, the GPU counts are the powers of two up to max_gpus.
        gc = grid_config_from_dict(None, max_gpus=8)
        assert gc.total_gpus == [1, 2, 4, 8]
        assert gc.batch_sizes == DEFAULT_BATCH_SIZES
        assert gc.top_k == 5  # default

    def test_explicit_list_overrides_max_gpus(self):
        gc = grid_config_from_dict({"grid": {"total_gpus": [3, 6]}}, max_gpus=8)
        assert gc.total_gpus == [3, 6]

    def test_no_list_and_no_max_gpus_yields_empty(self):
        gc = grid_config_from_dict({"grid": {"batch_sizes": [4]}})
        assert gc.total_gpus == []
        assert gc.batch_sizes == [4]

    def test_none_config_uses_defaults(self):
        gc = grid_config_from_dict(None)
        assert gc.total_gpus == []
        assert gc.batch_sizes == DEFAULT_BATCH_SIZES
        assert gc.top_k == 5


# generate_candidates: batch_sizes x total_gpus, clipped to the context
class TestGenerateCandidates:
    def test_cartesian_product_and_layouts(self):
        # max_gpus=16 drops 32, leaving 5 GPU counts x 2 batch sizes.
        ctx = _context(max_gpus=16, gpus_per_node=8, max_nodes=2)
        gc = GridConfig(batch_sizes=[4, 8], total_gpus=[1, 2, 4, 8, 16, 32])
        cands = generate_candidates(_workload(), ctx, gc)

        assert len(cands) == 5 * 2  # 32 is over max_gpus
        layouts = sorted({(c.gpus_per_node, c.number_of_nodes, c.total_gpus) for c in cands})
        assert layouts == [(1, 1, 1), (2, 1, 2), (4, 1, 4), (8, 1, 8), (8, 2, 16)]

        # every candidate has one of the requested batch sizes
        assert {c.batch_size for c in cands} == {4, 8}

    def test_max_gpus_bound_excludes_oversized_configs(self):
        ctx = _context(max_gpus=4, gpus_per_node=8, max_nodes=4)
        gc = GridConfig(batch_sizes=[1], total_gpus=[1, 2, 4, 8, 16])
        cands = generate_candidates(_workload(batch_size=1), ctx, gc)
        assert {c.total_gpus for c in cands} == {1, 2, 4}
        assert all(c.total_gpus <= ctx.max_gpus for c in cands)

    def test_max_nodes_bound_excludes_too_many_nodes(self):
        # At 8 GPUs per node and max_nodes=2 at most 16 GPUs fit, although max_gpus allows more,
        # so 24 (3 nodes) and 32 (4 nodes) are dropped.
        ctx = _context(max_gpus=64, gpus_per_node=8, max_nodes=2)
        gc = GridConfig(batch_sizes=[8], total_gpus=[8, 16, 24, 32])
        cands = generate_candidates(_workload(), ctx, gc)
        layouts = sorted({(c.gpus_per_node, c.number_of_nodes, c.total_gpus) for c in cands})
        assert layouts == [(8, 1, 8), (8, 2, 16)]
        assert all(c.number_of_nodes <= ctx.constraints.max_nodes for c in cands)

    def test_gpus_per_node_cap_is_respected(self):
        # With 4 GPUs per node, 8 GPUs take 2 nodes.
        ctx = _context(max_gpus=16, gpus_per_node=4, max_nodes=8)
        gc = GridConfig(batch_sizes=[4], total_gpus=[4, 8])
        cands = generate_candidates(_workload(batch_size=4), ctx, gc)
        layouts = sorted({(c.gpus_per_node, c.number_of_nodes, c.total_gpus) for c in cands})
        assert layouts == [(4, 1, 4), (4, 2, 8)]
        assert all(c.gpus_per_node <= ctx.constraints.gpus_per_node for c in cands)

    def test_empty_grid_total_gpus_falls_back_to_powers_of_two(self):
        # An empty total_gpus list is filled from max_gpus.
        ctx = _context(max_gpus=4, gpus_per_node=8, max_nodes=2)
        gc = GridConfig(batch_sizes=[2], total_gpus=[])
        cands = generate_candidates(_workload(batch_size=2), ctx, gc)
        assert {c.total_gpus for c in cands} == {1, 2, 4}

    def test_candidates_inherit_workload_fields(self):
        ctx = _context()
        gc = GridConfig(batch_sizes=[8], total_gpus=[2])
        wl = _workload(batch_size=999)  # the batch size comes from the grid
        cands = generate_candidates(wl, ctx, gc)
        assert len(cands) == 1
        c = cands[0]
        assert c.llm_model == wl.llm_model
        assert c.fine_tuning_method == wl.fine_tuning_method
        assert c.gpu_model == wl.gpu_model
        assert c.tokens_per_sample == wl.tokens_per_sample
        assert c.batch_size == 8  # from the grid

    def test_explicit_zero_total_gpus_is_skipped_cleanly(self):
        # The 0 is skipped (it would divide by zero in _derive_node_layout), and 2 still gives
        # candidates.
        ctx = _context()
        gc = GridConfig(batch_sizes=[8], total_gpus=[0, 2])
        cands = generate_candidates(_workload(), ctx, gc)
        assert {c.total_gpus for c in cands} == {2}

    def test_negative_total_gpus_is_skipped_cleanly(self):
        # Negative entries are skipped too.
        ctx = _context()
        gc = GridConfig(batch_sizes=[8], total_gpus=[-4, 2])
        cands = generate_candidates(_workload(), ctx, gc)
        assert {c.total_gpus for c in cands} == {2}

    def test_all_non_positive_total_gpus_yields_no_candidates(self):
        # With only non-positive entries the grid is empty and nothing is raised.
        ctx = _context()
        gc = GridConfig(batch_sizes=[8], total_gpus=[0, -1])
        cands = generate_candidates(_workload(), ctx, gc)
        assert cands == []
