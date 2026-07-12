"""Cluster-size awareness: resolve_cluster_caps + the grid never proposing a layout > the cluster.

The cluster size is sysadmin-declared in infrastructure.yaml (or a --cluster-gpus flag) — never read
from the workload trace. These tests pin (1) how the flag overrides the declared caps and (2) that
`generate_candidates` never emits a candidate whose ACTUAL layout exceeds the cluster budget, including
the non-power-of-two rounding case (request 30 at 8/node rounds up to 8x4 = 32).
"""

from __future__ import annotations

import coastline.sdk.io.infrastructure as infra_mod
from coastline.sdk.io.infrastructure import Infrastructure, resolve_cluster_caps
from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.pipeline.grid import GridConfig, generate_candidates

_FAKE_INFRA = Infrastructure(total_gpus=32, max_nodes=4, max_gpus_per_node=8, gpu_models=["NVIDIA-A100-SXM4-80GB"])


def _use_fake_infra(monkeypatch):
    monkeypatch.setattr(infra_mod, "load_infrastructure", lambda: _FAKE_INFRA)


def _workload() -> WorkloadSpec:
    return WorkloadSpec(
        llm_model="granite-3.1-8b-instruct",
        fine_tuning_method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=8,
        gpus_per_node=8,
        number_of_nodes=1,
    )


def _context(max_gpus: int, gpus_per_node: int = 8, max_nodes: int = 8) -> SystemContext:
    return SystemContext(
        available_gpu_models=["NVIDIA-A100-SXM4-80GB"],
        max_gpus=max_gpus,
        gpu_memory={"NVIDIA-A100-SXM4-80GB": 80},
        constraints=Constraints(max_gpus=max_gpus, gpus_per_node=gpus_per_node, max_nodes=max_nodes),
    )


# --------------------------------------------------------------------------- #
# resolve_cluster_caps
# --------------------------------------------------------------------------- #
class TestResolveClusterCaps:
    def test_defaults_to_infrastructure_file(self, monkeypatch):
        _use_fake_infra(monkeypatch)
        assert resolve_cluster_caps() == (32, 8, 4)  # declared total / per-node / max_nodes

    def test_cluster_gpus_flag_overrides_total_and_derives_max_nodes(self, monkeypatch):
        _use_fake_infra(monkeypatch)
        assert resolve_cluster_caps(cluster_gpus=64) == (64, 8, 8)  # ceil(64/8) = 8 nodes
        assert resolve_cluster_caps(cluster_gpus=30) == (30, 8, 4)  # ceil(30/8) = 4 nodes

    def test_small_cluster_clamps_per_node(self, monkeypatch):
        _use_fake_infra(monkeypatch)
        assert resolve_cluster_caps(cluster_gpus=4) == (4, 4, 1)  # per-node clamped to total; 1 node


# --------------------------------------------------------------------------- #
# generate_candidates — never exceeds the cluster budget
# --------------------------------------------------------------------------- #
class TestGridNeverExceedsCluster:
    def test_explicit_grid_is_capped_to_cluster(self):
        # Config asks for up to 64 GPUs, but the cluster is 16 -> 32 and 64 must be dropped.
        cands = generate_candidates(
            _workload(), _context(max_gpus=16), GridConfig(batch_sizes=[8], total_gpus=[1, 2, 4, 8, 16, 32, 64])
        )
        totals = {c.gpus_per_node * c.number_of_nodes for c in cands}
        assert totals == {1, 2, 4, 8, 16}
        assert max(totals) <= 16

    def test_empty_grid_is_derived_as_powers_of_two_up_to_cluster(self):
        # No explicit grid -> derive [1,2,4,8,16,32] up to a 32-GPU cluster.
        cands = generate_candidates(_workload(), _context(max_gpus=32), GridConfig(batch_sizes=[8], total_gpus=[]))
        totals = sorted({c.gpus_per_node * c.number_of_nodes for c in cands})
        assert totals == [1, 2, 4, 8, 16, 32]

    def test_non_power_of_two_step_that_rounds_up_is_rejected(self):
        # Request 30 GPUs at 8/node rounds to 8x4 = 32 actual, which exceeds a 30-GPU cluster.
        # The airtight cap must drop it rather than propose a 32-GPU layout.
        cands = generate_candidates(
            _workload(), _context(max_gpus=30), GridConfig(batch_sizes=[8], total_gpus=[8, 16, 30])
        )
        totals = {c.gpus_per_node * c.number_of_nodes for c in cands}
        assert 32 not in totals
        assert all(t <= 30 for t in totals)
        assert totals == {8, 16}  # 8x1 and 8x2; the 30-step layout (8x4=32) is dropped
