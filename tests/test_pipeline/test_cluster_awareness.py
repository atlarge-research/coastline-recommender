"""Cluster size: resolve_cluster_caps, and a grid that stays within the cluster.

The cluster size comes from infrastructure.yaml or the --cluster-gpus flag, never from the workload
trace. The tests cover how the flag overrides the declared caps, and that ``generate_candidates``
keeps every layout within the cluster, including 30 GPUs at up to 8 per node (laid out as 6 x 5).
"""

from __future__ import annotations

import pytest

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

    @pytest.mark.parametrize(
        "kwargs",
        [{"cluster_gpus": 0}, {"cluster_gpus": -8}, {"node_gpus": 0}, {"node_gpus": -1}],
        ids=["no cluster GPUs", "negative cluster GPUs", "no GPUs per node", "negative GPUs per node"],
    )
    def test_a_value_below_one_is_rejected(self, monkeypatch, kwargs):
        # Zero is rejected (it would otherwise read as 'not given'), and so is a negative node width.
        _use_fake_infra(monkeypatch)
        with pytest.raises(ValueError, match="must be >= 1"):
            resolve_cluster_caps(**kwargs)

    def test_one_gpu_is_a_valid_cluster(self, monkeypatch):
        _use_fake_infra(monkeypatch)
        assert resolve_cluster_caps(cluster_gpus=1, node_gpus=1) == (1, 1, 1)


# generate_candidates stays within the cluster budget
class TestGridNeverExceedsCluster:
    def test_explicit_grid_is_capped_to_cluster(self):
        # The grid asks for up to 64 GPUs on a 16-GPU cluster, so 32 and 64 are dropped.
        cands = generate_candidates(
            _workload(), _context(max_gpus=16), GridConfig(batch_sizes=[8], total_gpus=[1, 2, 4, 8, 16, 32, 64])
        )
        totals = {c.gpus_per_node * c.number_of_nodes for c in cands}
        assert totals == {1, 2, 4, 8, 16}
        assert max(totals) <= 16

    def test_empty_grid_is_derived_as_powers_of_two_up_to_cluster(self):
        # Without a GPU list the grid is the powers of two up to the 32-GPU cluster.
        cands = generate_candidates(_workload(), _context(max_gpus=32), GridConfig(batch_sizes=[8], total_gpus=[]))
        totals = sorted({c.gpus_per_node * c.number_of_nodes for c in cands})
        assert totals == [1, 2, 4, 8, 16, 32]

    def test_non_power_of_two_step_gets_an_exact_layout_within_the_cluster(self):
        # 30 GPUs at up to 8 per node is laid out as 6 x 5. Rounding up to 8 x 4 = 32 would
        # exceed the 30-GPU cluster.
        cands = generate_candidates(
            _workload(), _context(max_gpus=30), GridConfig(batch_sizes=[8], total_gpus=[8, 16, 30])
        )
        totals = {c.gpus_per_node * c.number_of_nodes for c in cands}
        assert 32 not in totals
        assert all(t <= 30 for t in totals)
        assert totals == {8, 16, 30}
        assert {(c.gpus_per_node, c.number_of_nodes) for c in cands} == {(8, 1), (8, 2), (6, 5)}
