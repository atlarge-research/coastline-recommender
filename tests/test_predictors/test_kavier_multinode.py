"""Tests for KavierPredictor on multi-node workloads.

Kavier's ``simulate_training_step(..., num_gpus, num_nodes)`` takes ``num_gpus`` as the total GPU
count across all nodes: tokens per step scale with ``num_gpus``, and ``num_nodes`` only adds the
inter-node all-reduce cost (``_comm_time``). ``KavierPredictor.predict`` passes
``num_gpus = WorkloadSpec.total_gpus`` (gpus_per_node x number_of_nodes) and
``num_nodes = number_of_nodes``, so the simulated GPU count matches the one the Prediction
reports. The Exp1 benchmark loads single-node WT1 rows only, so its numbers do not depend on this.
"""

import pytest

from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.workload import WorkloadSpec


# A100 context sized for multi-node (up to 128 GPUs / 16 nodes).
@pytest.fixture
def multinode_context():
    return SystemContext(
        available_gpu_models=["NVIDIA-A100-SXM4-80GB"],
        max_gpus=128,
        gpu_memory={"NVIDIA-A100-SXM4-80GB": 80},
        constraints=Constraints(max_gpus=128, gpus_per_node=8, max_nodes=16),
    )


# Model, GPU and method calibrated in Kavier.
SUPPORTED_MODEL = "mistral-7b-v0.1"
SUPPORTED_GPU = "NVIDIA-A100-SXM4-80GB"
SUPPORTED_METHOD = "lora"


def _engine():
    """Import the Kavier engine, or skip the test if Kavier is unavailable."""
    try:
        from kavier.sdk.training.core.engine import simulate_training_step
    except Exception as e:  # pragma: no cover - environment guard
        pytest.skip(f"Kavier engine not importable: {e}")
    return simulate_training_step


def _predictor():
    from coastline.sdk.predictors.performance.physics.kavier_predictor import (
        KAVIER_AVAILABLE,
        KavierPredictor,
    )

    if not KAVIER_AVAILABLE:  # pragma: no cover - environment guard
        pytest.skip("KavierPredictor reports KAVIER_AVAILABLE=False")
    return KavierPredictor()


def _wl(gpus_per_node, number_of_nodes, model=SUPPORTED_MODEL):
    return WorkloadSpec(
        llm_model=model,
        fine_tuning_method=SUPPORTED_METHOD,
        gpu_model=SUPPORTED_GPU,
        tokens_per_sample=2048,
        batch_size=16,
        gpus_per_node=gpus_per_node,
        number_of_nodes=number_of_nodes,
    )


def _engine_tps(sim, num_gpus, num_nodes, model=SUPPORTED_MODEL):
    return sim(
        model_name=model,
        gpu_model=SUPPORTED_GPU,
        tokens_per_sample=2048,
        batch_size=16,
        method=SUPPORTED_METHOD,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
    )["tokens_per_second"]


# Engine: num_gpus is the total GPU count


def test_engine_holding_total_gpus_fixed_adding_a_node_only_adds_comm_cost():
    """At a fixed total of 16 GPUs, splitting 1 node into 2 lowers throughput.

    Tokens per step scale with num_gpus, which is 16 in both calls; the second node only adds an
    inter-node all-reduce term. If num_gpus were read per node, the 2-node call would do twice the
    work and report a higher throughput.
    """
    sim = _engine()
    one_node = _engine_tps(sim, num_gpus=16, num_nodes=1)
    two_node = _engine_tps(sim, num_gpus=16, num_nodes=2)
    assert one_node > 0 and two_node > 0  # the config is supported
    assert two_node < one_node


def test_engine_num_gpus_below_num_nodes_stays_positive():
    """With fewer GPUs than nodes (2 GPUs on 4 nodes), throughput stays positive.

    Tokens per step scale with num_gpus, and ``_comm_time`` clamps gpus_per_node to at least 1, so
    ``2 // 4 == 0`` does not zero the result.
    """
    sim = _engine()
    assert _engine_tps(sim, num_gpus=2, num_nodes=4) > 0.0


# Wrapper multi-node behavior: the engine gets the total GPU count


def test_wrapper_feeds_total_gpus_to_engine(multinode_context):
    """For 8 GPUs/node x 4 nodes, the wrapper's throughput equals the engine's at num_gpus=32,
    num_nodes=4."""
    sim = _engine()
    wl = _wl(gpus_per_node=8, number_of_nodes=4)
    # 8 GPUs/node x 4 nodes = 32.
    assert wl.total_gpus == 32
    pred = _predictor().predict(wl, multinode_context)
    assert pred is not None
    expected = _engine_tps(sim, num_gpus=32, num_nodes=4)
    assert pred.predicted_throughput == pytest.approx(expected)
    # Passing the per-node count (8) as the total gives a different number.
    assert pred.predicted_throughput != pytest.approx(_engine_tps(sim, num_gpus=8, num_nodes=4))


def test_wrapper_reports_layout_and_derived_total(multinode_context):
    """The Prediction echoes the requested layout (8 GPUs/node, 4 nodes) and reports total_gpus = 32."""
    pred = _predictor().predict(_wl(gpus_per_node=8, number_of_nodes=4), multinode_context)
    assert pred is not None
    assert pred.gpus_per_node == 8
    assert pred.number_of_nodes == 4
    assert pred.total_gpus == 32  # 8 x 4


def test_wrapper_low_per_node_multinode_predicts(multinode_context):
    """2 GPUs/node x 4 nodes (8 total) gives the engine's throughput at num_gpus=8, num_nodes=4."""
    sim = _engine()
    pred = _predictor().predict(_wl(gpus_per_node=2, number_of_nodes=4), multinode_context)
    assert pred is not None
    # 2 GPUs/node x 4 nodes = 8.
    assert pred.total_gpus == 8
    # The same check as the 8 x 4 test, on a layout with few GPUs per node.
    assert pred.predicted_throughput == pytest.approx(_engine_tps(sim, num_gpus=8, num_nodes=4))


def test_wrapper_per_gpu_power_within_idle_tdp_envelope(multinode_context):
    """Multi-node per-GPU power stays within the A100-SXM4-80GB's [idle, TDP] range, [75, 400] W
    in Kavier's GPU catalog. ``predicted_power`` is per GPU and does not depend on the GPU count.
    """
    from kavier.sdk.library.lookup import get_gpu

    spec = get_gpu(SUPPORTED_GPU)
    idle, tdp = spec.idle_power_w, spec.max_power_w
    assert idle == 75 and tdp == 400  # the catalog values this test relies on
    pred = _predictor().predict(_wl(gpus_per_node=8, number_of_nodes=4), multinode_context)
    assert pred is not None and pred.predicted_power is not None
    assert idle <= pred.predicted_power <= tdp


def test_wrapper_single_node_unchanged(multinode_context):
    """With one node, total_gpus equals gpus_per_node and the engine gets (8, 1). WT1's curated
    set, used in Exp1, is single-node only.
    """
    sim = _engine()
    pred = _predictor().predict(
        _wl(gpus_per_node=8, number_of_nodes=1, model="mistral-7b-v0.1"),
        multinode_context,
    )
    assert pred is not None
    # 8 GPUs/node x 1 node = 8.
    assert pred.total_gpus == 8
    assert pred.predicted_throughput == pytest.approx(
        _engine_tps(sim, num_gpus=8, num_nodes=1, model="mistral-7b-v0.1")
    )


def test_wrapper_multinode_is_deterministic(multinode_context):
    """Three identical multi-node predictions give the same throughput."""
    predictor = _predictor()
    wl = _wl(gpus_per_node=8, number_of_nodes=2)
    vals = {predictor.predict(wl, multinode_context).predicted_throughput for _ in range(3)}
    assert len(vals) == 1


# dev/benchmark/run_benchmark.py::evaluate_kavier calls the engine directly with
# num_gpus = number_gpus * number_nodes. WT1's number_gpus is already the total, so this is right
# only for single-node rows (all of WT1's curated set) and would over-count multi-node rows.
