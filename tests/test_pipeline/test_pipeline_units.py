"""Focused unit tests for the recommender pipeline building blocks.

Targets the feasibility / grid-generation / scoring units that are only
covered indirectly by the integration tests in test_unified_workflow.py:

* coastline.sdk.predictors.feasibility.autoconf.RulesFeasibilityChecker
* coastline.sdk.pipeline.grid: _powers_of_two, _derive_node_layout,
  grid_config_from_dict, generate_candidates

All inputs are synthetic and deterministic; no model artifacts or data files
are loaded. These tests do not modify production code.
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


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# RulesFeasibilityChecker — divisibility rule + total_gpus bound
# --------------------------------------------------------------------------- #
class TestRulesFeasibility:
    def test_divisible_batch_is_feasible(self):
        # total_gpus = 1 * 4 = 4; batch_size 8 % 4 == 0 -> feasible
        wl = _workload(batch_size=8, gpus_per_node=4, number_of_nodes=1)
        assert wl.total_gpus == 4
        ok, meta = RulesFeasibilityChecker().is_feasible(wl)
        assert ok is True
        assert meta == {}

    def test_indivisible_batch_is_rejected(self):
        # total_gpus = 3; batch_size 8 % 3 != 0 -> rejected with explanatory metadata
        wl = _workload(batch_size=8, gpus_per_node=3, number_of_nodes=1)
        assert wl.total_gpus == 3
        ok, meta = RulesFeasibilityChecker().is_feasible(wl)
        assert ok is False
        assert "error" in meta
        assert "divisible" in meta["error"]

    def test_batch_equals_total_gpus_is_feasible(self):
        # boundary: batch_size == total_gpus -> remainder 0 -> feasible
        wl = _workload(batch_size=8, gpus_per_node=8, number_of_nodes=1)
        ok, meta = RulesFeasibilityChecker().is_feasible(wl)
        assert ok is True
        assert meta == {}  # feasible => no explanatory error payload

    def test_divisibility_keys_off_total_gpus_not_per_node(self):
        # Discriminating case: total_gpus = 8 * 2 = 16, batch_size = 8.
        # Oracle: 8 % 16 == 8 != 0  -> MUST be rejected.
        # A bug keying off gpus_per_node (8) would compute 8 % 8 == 0 and WRONGLY
        # accept; that is exactly the regression this test pins. (batch==total==16
        # cannot discriminate, since 16 is divisible by both 16 and 8.)
        wl = _workload(batch_size=8, gpus_per_node=8, number_of_nodes=2)
        assert wl.total_gpus == 16  # 8 * 2, hand-computed
        ok, meta = RulesFeasibilityChecker().is_feasible(wl)
        assert ok is False
        assert "divisible" in meta["error"]

    def test_multi_node_divisible_by_total_is_feasible(self):
        # Multi-node feasible path: total_gpus = 8 * 2 = 16, batch_size = 32.
        # Oracle: 32 % 16 == 0 -> feasible.
        wl = _workload(batch_size=32, gpus_per_node=8, number_of_nodes=2)
        assert wl.total_gpus == 16
        ok, meta = RulesFeasibilityChecker().is_feasible(wl)
        assert ok is True
        assert meta == {}

    def test_default_layout_single_gpu_always_divisible(self):
        # No layout given -> total_gpus defaults to 1; everything divisible by 1.
        wl = _workload(batch_size=7)
        assert wl.total_gpus == 1
        ok, _ = RulesFeasibilityChecker().is_feasible(wl)
        assert ok is True


# --------------------------------------------------------------------------- #
# _powers_of_two
# --------------------------------------------------------------------------- #
class TestPowersOfTwo:
    @pytest.mark.parametrize(
        "limit, expected",
        [
            (1, [1]),
            (2, [1, 2]),
            (8, [1, 2, 4, 8]),
            (16, [1, 2, 4, 8, 16]),
            (10, [1, 2, 4, 8]),  # truncates below non-power-of-two limit
            (0, []),  # nothing fits under 1
        ],
    )
    def test_powers_of_two(self, limit, expected):
        assert _powers_of_two(limit) == expected


# --------------------------------------------------------------------------- #
# _derive_node_layout — pack GPUs per node, ceil node count
# --------------------------------------------------------------------------- #
class TestDeriveNodeLayout:
    @pytest.mark.parametrize(
        "total_gpus, max_per_node, expected",
        [
            (1, 8, (1, 1)),  # single GPU -> single node
            (8, 8, (8, 1)),  # exactly fills one node
            (16, 8, (8, 2)),  # two full nodes
            (12, 8, (8, 2)),  # 12 GPUs -> 8/node, ceil(12/8) = 2 nodes
            (3, 8, (3, 1)),  # fewer than a node packs into one node
            (9, 8, (8, 2)),  # one over a node -> second node
            (5, 4, (4, 2)),  # smaller node cap
        ],
    )
    def test_layout(self, total_gpus, max_per_node, expected):
        gpus_per_node, num_nodes = _derive_node_layout(total_gpus, max_per_node)
        assert (gpus_per_node, num_nodes) == expected
        # invariant: the allocation covers the requested total ...
        assert gpus_per_node * num_nodes >= total_gpus
        # ... and is minimal — dropping one node would NOT cover it (independent of
        # the impl's ceil formula; asserts the "fewest nodes" contract directly).
        assert (num_nodes - 1) * gpus_per_node < total_gpus
        # invariant: never pack more than the node cap onto a node
        assert gpus_per_node <= max_per_node

    def test_zero_total_gpus_raises_zero_division(self):
        # EDGE CASE / latent bug: total_gpus=0 -> gpus_per_node=min(0,8)=0 -> ceil(0/0).
        # Guarded upstream today (callers feed positive values), documented here.
        with pytest.raises(ZeroDivisionError):
            _derive_node_layout(0, 8)


# --------------------------------------------------------------------------- #
# grid_config_from_dict
# --------------------------------------------------------------------------- #
class TestGridConfigFromDict:
    def test_explicit_total_gpus_list_preserved(self):
        gc = grid_config_from_dict({"grid": {"total_gpus": [2, 4], "batch_sizes": [8, 16], "top_k": 5}})
        assert gc.total_gpus == [2, 4]
        assert gc.batch_sizes == [8, 16]
        assert gc.top_k == 5

    def test_total_gpus_derived_from_max_gpus_when_absent(self):
        # No explicit list, max_gpus given -> powers of two up to max_gpus.
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


# --------------------------------------------------------------------------- #
# generate_candidates — grid (batch_sizes × total_gpus) with context clipping
# --------------------------------------------------------------------------- #
class TestGenerateCandidates:
    def test_cartesian_product_and_layouts(self):
        # total_gpus [1,2,4,8,16,32] clipped at max_gpus=16; 5 surviving steps × 2 bs.
        ctx = _context(max_gpus=16, gpus_per_node=8, max_nodes=2)
        gc = GridConfig(batch_sizes=[4, 8], total_gpus=[1, 2, 4, 8, 16, 32])
        cands = generate_candidates(_workload(), ctx, gc)

        assert len(cands) == 5 * 2  # 32 dropped (> max_gpus)
        layouts = sorted({(c.gpus_per_node, c.number_of_nodes, c.total_gpus) for c in cands})
        assert layouts == [(1, 1, 1), (2, 1, 2), (4, 1, 4), (8, 1, 8), (8, 2, 16)]

        # every candidate carries one of the requested batch sizes
        assert {c.batch_size for c in cands} == {4, 8}

    def test_max_gpus_bound_excludes_oversized_configs(self):
        ctx = _context(max_gpus=4, gpus_per_node=8, max_nodes=4)
        gc = GridConfig(batch_sizes=[1], total_gpus=[1, 2, 4, 8, 16])
        cands = generate_candidates(_workload(batch_size=1), ctx, gc)
        assert {c.total_gpus for c in cands} == {1, 2, 4}
        assert all(c.total_gpus <= ctx.max_gpus for c in cands)

    def test_max_nodes_bound_excludes_too_many_nodes(self):
        # gpus_per_node=8, max_nodes=2 -> only up to 16 total GPUs reachable,
        # even though max_gpus allows more. 24 (3 nodes) and 32 (4 nodes) dropped.
        ctx = _context(max_gpus=64, gpus_per_node=8, max_nodes=2)
        gc = GridConfig(batch_sizes=[8], total_gpus=[8, 16, 24, 32])
        cands = generate_candidates(_workload(), ctx, gc)
        layouts = sorted({(c.gpus_per_node, c.number_of_nodes, c.total_gpus) for c in cands})
        assert layouts == [(8, 1, 8), (8, 2, 16)]
        assert all(c.number_of_nodes <= ctx.constraints.max_nodes for c in cands)

    def test_gpus_per_node_cap_is_respected(self):
        # Node cap of 4 means 8 total GPUs must span 2 nodes (not one node of 8).
        ctx = _context(max_gpus=16, gpus_per_node=4, max_nodes=8)
        gc = GridConfig(batch_sizes=[4], total_gpus=[4, 8])
        cands = generate_candidates(_workload(batch_size=4), ctx, gc)
        layouts = sorted({(c.gpus_per_node, c.number_of_nodes, c.total_gpus) for c in cands})
        assert layouts == [(4, 1, 4), (4, 2, 8)]
        assert all(c.gpus_per_node <= ctx.constraints.gpus_per_node for c in cands)

    def test_empty_grid_total_gpus_falls_back_to_powers_of_two(self):
        # GridConfig.total_gpus == [] -> generate_candidates derives from max_gpus.
        ctx = _context(max_gpus=4, gpus_per_node=8, max_nodes=2)
        gc = GridConfig(batch_sizes=[2], total_gpus=[])
        cands = generate_candidates(_workload(batch_size=2), ctx, gc)
        assert {c.total_gpus for c in cands} == {1, 2, 4}

    def test_candidates_inherit_workload_fields(self):
        ctx = _context()
        gc = GridConfig(batch_sizes=[8], total_gpus=[2])
        wl = _workload(batch_size=999)  # batch_size should come from grid, not workload
        cands = generate_candidates(wl, ctx, gc)
        assert len(cands) == 1
        c = cands[0]
        assert c.llm_model == wl.llm_model
        assert c.fine_tuning_method == wl.fine_tuning_method
        assert c.gpu_model == wl.gpu_model
        assert c.tokens_per_sample == wl.tokens_per_sample
        assert c.batch_size == 8  # from the grid

    def test_explicit_zero_total_gpus_is_skipped_cleanly(self):
        # A 0 in the explicit total_gpus list is skipped by the non-positive
        # guard (it would otherwise hit ceil(0/0) in _derive_node_layout);
        # the remaining valid entry (2) still produces candidates.
        ctx = _context()
        gc = GridConfig(batch_sizes=[8], total_gpus=[0, 2])
        cands = generate_candidates(_workload(), ctx, gc)
        assert {c.total_gpus for c in cands} == {2}

    def test_negative_total_gpus_is_skipped_cleanly(self):
        # Non-positive guard also drops negative entries without error.
        ctx = _context()
        gc = GridConfig(batch_sizes=[8], total_gpus=[-4, 2])
        cands = generate_candidates(_workload(), ctx, gc)
        assert {c.total_gpus for c in cands} == {2}

    def test_all_non_positive_total_gpus_yields_no_candidates(self):
        # If every entry is non-positive, the grid is empty (no error raised).
        ctx = _context()
        gc = GridConfig(batch_sizes=[8], total_gpus=[0, -1])
        cands = generate_candidates(_workload(), ctx, gc)
        assert cands == []
