"""Tests for KavierPredictor under MULTI-NODE workloads.

These pin the wrapper's GPU-count contract for multi-node configs. They were
originally characterization tests documenting a wrapper/engine inconsistency;
that inconsistency has since been FIXED (the wrapper now feeds the engine the
canonical total), so the tests assert the *correct* behavior.

Run:
    cd <repo-root>
    PYTHONPATH=coastline:coastline/common:kavier/src \
        DATA_DIR=./trace-archive \
        .venv/bin/python -m pytest \
        coastline/tests/test_kavier_multinode.py -q

----------------------------------------------------------------------------
GROUND TRUTH — Kavier engine GPU-count contract (kavier/src/.../core/engine.py)
----------------------------------------------------------------------------
``simulate_training_step(..., num_gpus, num_nodes)`` treats ``num_gpus`` as the
TOTAL GPU count across all nodes:

    tokens_per_step = ... * num_gpus                # throughput scales with the TOTAL
    # num_nodes only adds inter-node all-reduce cost (see _comm_time)

----------------------------------------------------------------------------
WRAPPER CONTRACT (fixed)
----------------------------------------------------------------------------
``KavierPredictor.predict`` feeds ``num_gpus = WorkloadSpec.total_gpus`` (the
canonical total = gpus_per_node × number_of_nodes) and ``num_nodes =
number_of_nodes``. So the GPU count Kavier simulates matches the count the
Prediction reports, and a grid candidate of 8 GPUs/node × N nodes is simulated
as the full 8·N total. (In WT1 the ``number_gpus`` column is already the total
and the benchmark loads single-node rows only, so this fix does not move the
Exp1 numbers; it corrects genuine multi-node recommendation candidates.)
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


# Model/GPU/method known to be calibrated in Kavier (per the project docs + engine).
SUPPORTED_MODEL = "mistral-7b-v0.1"
SUPPORTED_GPU = "NVIDIA-A100-SXM4-80GB"
SUPPORTED_METHOD = "lora"


def _engine():
    """Import the live engine (skip the whole module if Kavier is unavailable)."""
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


# ===========================================================================
# 0. Engine contract — num_gpus is the TOTAL
# ===========================================================================


def test_engine_holding_total_gpus_fixed_adding_a_node_only_adds_comm_cost():
    """Invariant: at a FIXED total num_gpus=16, splitting 1 node -> 2 nodes can
    only DECREASE throughput (never increase it).

    Oracle (mechanism, independent of the impl's constants): tokens_per_step
    scales with num_gpus, which is 16 in BOTH calls, so the per-step token count
    is identical. num_nodes feeds only ``_comm_time``: 1 node -> single ring
    all-reduce; 2 nodes -> an EXTRA inter-node all-reduce term (INFINIBAND_GBPS).
    Adding a non-negative comm term to step_time_s can only lengthen the step,
    so tps(16,2) <= tps(16,1). If num_gpus were interpreted PER-NODE, tps(16,2)
    would instead double the work and RISE -> the <= direction falsifies that.
    (Measured here: 26192.46 vs 26190.39, a ~2 tok/s inter-node penalty.)
    """
    sim = _engine()
    one_node = _engine_tps(sim, num_gpus=16, num_nodes=1)
    two_node = _engine_tps(sim, num_gpus=16, num_nodes=2)
    assert one_node > 0 and two_node > 0  # guard: supported config predicts
    assert two_node < one_node


def test_engine_num_gpus_below_num_nodes_no_longer_zeroes():
    """Regression: num_gpus < num_nodes must not floor-divide throughput to 0.

    Pinned-bug oracle: the OLD engine derived gpus_per_node = num_gpus //
    num_nodes = 2 // 4 = 0, which zeroed tokens_per_step and hence tps -> the
    exact wrong value is 0.0. The fix scales tokens_per_step by num_gpus
    directly (and clamps gpus_per_node to >=1 inside _comm_time), so this
    degenerate regime now yields a strictly positive throughput. Asserting > 0
    rejects the reintroduced floor-division (which would land back on 0.0).
    """
    sim = _engine()
    assert _engine_tps(sim, num_gpus=2, num_nodes=4) > 0.0  # old bug: == 0.0


# ===========================================================================
# 1. Wrapper multi-node behavior — feeds the canonical TOTAL
# ===========================================================================


def test_wrapper_feeds_total_gpus_to_engine(multinode_context):
    """Wrapper throughput == engine(num_gpus=total_gpus, num_nodes=nodes).

    For 8 GPUs/node × 4 nodes the wrapper must feed the engine 32 (the total),
    NOT 8. Pinned against a live engine call so it can't silently regress.
    """
    sim = _engine()
    wl = _wl(gpus_per_node=8, number_of_nodes=4)
    # Hand-derived total: 8 GPUs/node x 4 nodes = 32 (the WorkloadSpec.total_gpus rule).
    assert wl.total_gpus == 32
    pred = _predictor().predict(wl, multinode_context)
    assert pred is not None
    # Cross-check the WIRING: the wrapper must simulate the TOTAL (32) over 4 nodes.
    # If it fed per-node (8) as the total instead, this would equal engine(8, 4) and
    # diverge by the ~3.5x that motivated the fix.
    expected = _engine_tps(sim, num_gpus=32, num_nodes=4)
    assert pred.predicted_throughput == pytest.approx(expected)
    # And it is demonstrably NOT the per-node-as-total mis-wiring:
    assert pred.predicted_throughput != pytest.approx(_engine_tps(sim, num_gpus=8, num_nodes=4))


def test_wrapper_reports_layout_and_derived_total(multinode_context):
    """The Prediction echoes the requested layout and reports total = per_node x nodes.

    Hand-derived oracle: for 8 GPUs/node x 4 nodes the reported total must be
    8*4 = 32, with the per-node (8) and node (4) fields echoed verbatim. This is
    the metadata half of the fix (the throughput/wiring half is covered by
    test_wrapper_feeds_total_gpus_to_engine); previously the wrapper reported 32
    while simulating only 8, so pinning the reported fields guards the reporting
    path independently.
    """
    pred = _predictor().predict(_wl(gpus_per_node=8, number_of_nodes=4), multinode_context)
    assert pred is not None
    assert pred.gpus_per_node == 8
    assert pred.number_of_nodes == 4
    assert pred.total_gpus == 32  # 8 x 4


def test_wrapper_low_per_node_multinode_now_predicts(multinode_context):
    """2 GPUs/node × 4 nodes (8 total) now yields a positive prediction.

    Under the old wrapper this fed num_gpus=2, num_nodes=4 and the user got None.
    Feeding the total (8) makes the engine see per-node 2 — a valid layout.
    """
    sim = _engine()
    pred = _predictor().predict(_wl(gpus_per_node=2, number_of_nodes=4), multinode_context)
    assert pred is not None
    # Hand-derived total: 2 GPUs/node x 4 nodes = 8. Old wrapper fed num_gpus=2 as
    # the total -> engine floor-divided per-node to 0 -> None; the fix feeds 8.
    assert pred.total_gpus == 8
    # Cross-check the wiring on a DIFFERENT (low-per-node) layout than the 8x4 test.
    assert pred.predicted_throughput == pytest.approx(_engine_tps(sim, num_gpus=8, num_nodes=4))


def test_wrapper_per_gpu_power_within_idle_tdp_envelope(multinode_context):
    """Multi-node per-GPU power stays inside the GPU's [idle, TDP] envelope.

    Independent analytic reference: the physical power of one A100-SXM4-80GB can
    never be below its idle draw nor above its rated max. Those bounds come from
    the GPUSpec catalog (idle_power_w=75, max_power_w=400 for this part), NOT from
    the predictor -> a genuine external oracle. mse_power(u=0)=idle, mse_power at
    full util = max, so any valid utilization lands in between. A power formula
    that dropped the idle floor, double-counted, or read the wrong field would
    escape [75, 400] and fail. (predicted_power is per-GPU, independent of the
    node/GPU count.)
    """
    from kavier.sdk.library.lookup import get_gpu

    spec = get_gpu(SUPPORTED_GPU)
    idle, tdp = spec.idle_power_w, spec.max_power_w
    assert idle == 75 and tdp == 400  # pin the catalog envelope this test relies on
    pred = _predictor().predict(_wl(gpus_per_node=8, number_of_nodes=4), multinode_context)
    assert pred is not None and pred.predicted_power is not None
    assert idle <= pred.predicted_power <= tdp


def test_wrapper_single_node_unchanged(multinode_context):
    """Single-node (the only regime in WT1's curated set) is unaffected.

    With number_of_nodes=1, total_gpus == gpus_per_node, so the fix is a no-op
    here — which is why the Exp1 Kavier numbers do not move.
    """
    sim = _engine()
    pred = _predictor().predict(
        _wl(gpus_per_node=8, number_of_nodes=1, model="mistral-7b-v0.1"),
        multinode_context,
    )
    assert pred is not None
    # Hand-derived: 8 GPUs/node x 1 node = 8, so total == per-node and the
    # per-node<->total fix is a provable no-op here (feeds engine(8, 1) either way).
    assert pred.total_gpus == 8
    assert pred.predicted_throughput == pytest.approx(
        _engine_tps(sim, num_gpus=8, num_nodes=1, model="mistral-7b-v0.1")
    )


def test_wrapper_multinode_is_deterministic(multinode_context):
    """Invariant: the analytical engine is a pure function of its inputs.

    Oracle (property): identical (workload, context) must map to identical
    throughput -> the set of 3 outputs collapses to exactly one element. A
    non-deterministic path (unseeded RNG, wall-clock, dict-order-dependent
    calibration lookup) would yield >1 distinct value and fail.
    """
    predictor = _predictor()
    wl = _wl(gpus_per_node=8, number_of_nodes=2)
    vals = {predictor.predict(wl, multinode_context).predicted_throughput for _ in range(3)}
    assert len(vals) == 1


# ===========================================================================
# VERDICT
# ===========================================================================
# FIXED. The wrapper (kavier_predictor.py) now feeds the engine
# ``num_gpus = WorkloadSpec.total_gpus`` (= gpus_per_node × number_of_nodes),
# matching the engine's contract (num_gpus is the total) and the grid's per-node
# candidate semantics. The reported Prediction.total_gpus now equals the GPU
# count Kavier simulated. Single-node behavior is unchanged, so the curated
# (single-node) Exp1 benchmark numbers do not move.
#
# Note: the benchmark's evaluate_kavier (run_benchmark.py) still computes its own
# total as ``number_gpus * number_nodes`` via a *direct* engine call (not this
# wrapper). On WT1 that is correct only because the curated set is single-node
# (× 1); it would over-count if a multi-node model were ever added to the eval
# set. That is a separate, currently-dormant issue in the benchmark, tracked
# apart from this wrapper fix.
