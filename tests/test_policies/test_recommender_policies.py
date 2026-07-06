"""
Integration tests for FR2 recommender policies against the *real* Kavier
physics predictor (the fast-but-fake-predictor variants live in
``test_strategies.py``). These tests catch regressions a fake predictor would
mask: that the policies still wire up and rank correctly when fed Kavier's
analytical throughput/power numbers, that a restrictive ``SystemContext``
actually clamps the grid, and that the min_gpu / preset ordering rules survive
end-to-end.

Every assertion below is pinned to an oracle that is independent of the code
under test: the grid/clamp arithmetic is enumerated by hand, min_gpu's sort is
checked via its *observable* (total_gpus, batch_size) sequence rather than by
re-running its key function, and the preset comparison uses only directional
invariants derived from the preset weights.
"""

import pytest

from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.pipeline.feasibility import NoOpFeasibilityChecker
from coastline.sdk.pipeline.workflow import GridWorkflowPipeline
from coastline.sdk.policies.min_gpu import MinGPUStrategy
from coastline.sdk.policies.multi_objective import MultiObjectiveStrategy
from coastline.sdk.predictors.energy import KavierPowerPredictor
from coastline.sdk.predictors.performance.physics import KavierPredictor


def _min_gpu_strategy(*, batch_sizes, total_gpus, top_k) -> MinGPUStrategy:
    """MinGPU via the unified workflow with a NoOp feasibility checker (no AutoConf
    needed in tests), so the only thing filtering the grid is the SystemContext."""
    pipeline = GridWorkflowPipeline.from_config(
        config={"grid": {"batch_sizes": batch_sizes, "total_gpus": total_gpus, "top_k": top_k}},
        selection_policy="min_gpu",
        strategy_name="min_gpu",
        throughput_predictor=KavierPredictor(),
        power_predictor=KavierPowerPredictor(),
        feasibility_checker=NoOpFeasibilityChecker(),
    )
    return MinGPUStrategy(pipeline=pipeline)


@pytest.fixture
def test_workload():
    """Create a test workload."""
    return WorkloadSpec(
        llm_model="mistral-7b-v0.1",
        fine_tuning_method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=4,
        gpus_per_node=8,
        number_of_nodes=1,
    )


@pytest.fixture
def test_context():
    """Create a test system context."""
    return SystemContext(
        available_gpu_models=["NVIDIA-A100-SXM4-80GB"],
        max_gpus=16,
        gpu_memory={"NVIDIA-A100-SXM4-80GB": 80},
        constraints=Constraints(
            max_gpus=32,
            gpus_per_node=8,
            max_nodes=2,
        ),
    )


@pytest.fixture
def restricted_context():
    """Create a restricted system context for testing limits."""
    return SystemContext(
        available_gpu_models=["NVIDIA-A100-SXM4-80GB"],
        max_gpus=4,  # Very restrictive
        gpu_memory={"NVIDIA-A100-SXM4-80GB": 80},
        constraints=Constraints(
            max_gpus=8,
            gpus_per_node=4,
            max_nodes=1,
        ),
    )


def test_min_gpu_over_context_clamped_grid_returns_ascending_feasible_configs(test_workload, restricted_context):
    """min_gpu over a grid clamped by a restrictive context yields exactly the
    surviving configs, ascending by GPU count, with hand-derived node layouts.

    Grid total_gpus = [1, 2, 4, 8, 16]. restricted_context.max_gpus = 4, so
    ``generate_candidates`` drops 8 and 16 (``n_gpus > max_gpus``). Each survivor
    fits a single node: gpus_per_node = min(g, constraints.gpus_per_node=4),
    number_of_nodes = ceil(g / gpus_per_node) = 1. min_gpu sorts by ascending
    total_gpus, so by hand the ordered layout is:
        1 GPU  -> (gpus_per_node=1, nodes=1)
        2 GPUs -> (gpus_per_node=2, nodes=1)
        4 GPUs -> (gpus_per_node=4, nodes=1)
    A clamp bug (8/16 leaking through) or a reversed sort both break this list.
    """
    recs = _min_gpu_strategy(batch_sizes=[4], total_gpus=[1, 2, 4, 8, 16], top_k=3).recommend(
        test_workload, restricted_context
    )

    assert [(r.total_gpus, r.gpus_per_node, r.number_of_nodes) for r in recs] == [
        (1, 1, 1),
        (2, 2, 1),
        (4, 4, 1),
    ]


def test_min_gpu_breaks_ties_within_equal_gpu_count_by_higher_throughput(test_workload, test_context):
    """Within one GPU count, min_gpu orders configs by higher throughput first.

    min_gpu's dominant key is ascending total_gpus, so every 1-GPU config must
    precede every 2-GPU config. Its documented tie-break is throughput-descending.
    Kavier throughput rises with batch_size at a fixed GPU count (measured on this
    workload at 1 GPU: bs=2 -> 3464.5, bs=4 -> 3603.9, bs=8 -> 3743.2 tok/s), so a
    throughput-descending tie-break must emit batches in the order 8, 4, 2 inside
    each GPU group. We assert that observable (total_gpus, batch_size) sequence —
    independent of min_gpu's numeric sort key. Removing the tie-break would leave
    the grid's insertion order (2, 4, 8); reversing it would give 2, 4, 8 too.
    """
    recs = _min_gpu_strategy(batch_sizes=[2, 4, 8], total_gpus=[1, 2], top_k=10).recommend(test_workload, test_context)

    assert [(r.total_gpus, r.metadata["batch_size"]) for r in recs] == [
        (1, 8),
        (1, 4),
        (1, 2),
        (2, 8),
        (2, 4),
        (2, 2),
    ]


def test_energy_preset_diverges_toward_lower_power_than_performance_preset(test_workload, test_context):
    """The energy and performance presets must diverge in the expected direction.

    The scoring power axis is TOTAL power = per-GPU watts × total_gpus
    (selection.py ``_power_cost``). energy weights power 0.8 / throughput 0.2;
    performance mirrors it (0.2 / 0.8). On a workload where extra GPUs buy more
    throughput at proportionally more total power, the two presets cannot pick the
    same config: performance chases throughput (more GPUs), energy minimises total
    power (fewer GPUs). These are directional oracles — no specific wattage or
    token number is pinned:
      * performance's pick is strictly faster than energy's,
      * performance's pick uses strictly more GPUs,
      * energy's pick draws strictly less total power.
    Inverting the score direction, or ignoring the preset weights (both presets
    collapsing to the same config), flips or nullifies every one of these.
    """
    energy_top = MultiObjectiveStrategy(
        throughput_predictor=KavierPredictor(),
        power_predictor=KavierPowerPredictor(),
        preset="energy",
    ).recommend(test_workload, test_context)[0]
    perf_top = MultiObjectiveStrategy(
        throughput_predictor=KavierPredictor(),
        power_predictor=KavierPowerPredictor(),
        preset="performance",
    ).recommend(test_workload, test_context)[0]

    energy_total_power = energy_top.metadata["predicted_power_watts"] * energy_top.total_gpus
    perf_total_power = perf_top.metadata["predicted_power_watts"] * perf_top.total_gpus

    assert perf_top.predicted_throughput > energy_top.predicted_throughput
    assert perf_top.total_gpus > energy_top.total_gpus
    assert energy_total_power < perf_total_power


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
