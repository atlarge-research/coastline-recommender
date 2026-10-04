"""Integration tests for the FR2 recommendation policies with the Kavier predictor.

test_strategies.py covers the same policies with a fake predictor. Here the policies rank Kavier's
numbers, a restrictive SystemContext limits the grid, and the min_gpu and preset orderings hold end
to end.
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
    """MinGPUStrategy with a NoOp feasibility checker, so only the SystemContext limits the GPU
    counts. min_gpu reads only top_k from the grid."""
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
    return SystemContext(
        available_gpu_models=["NVIDIA-A100-SXM4-80GB"],
        max_gpus=4,
        gpu_memory={"NVIDIA-A100-SXM4-80GB": 80},
        constraints=Constraints(
            max_gpus=8,
            gpus_per_node=4,
            max_nodes=1,
        ),
    )


def test_min_gpu_over_context_clamped_grid_returns_ascending_feasible_configs(test_workload, restricted_context):
    """With max_gpus=4 min_gpu checks 1, 2 and 4 GPUs, each on one node, and returns them in
    doubling order."""
    recs = _min_gpu_strategy(batch_sizes=[4], total_gpus=[1, 2, 4, 8, 16], top_k=3).recommend(
        test_workload, restricted_context
    )

    assert [(r.total_gpus, r.gpus_per_node, r.number_of_nodes) for r in recs] == [
        (1, 1, 1),
        (2, 2, 1),
        (4, 4, 1),
    ]


def test_min_gpu_keeps_the_jobs_total_batch_on_every_gpu_count(test_workload, test_context):
    """min_gpu explores GPU counts only: the grid's batch sizes and GPU counts are not used, and
    every returned configuration splits the job's total batch (4 per device on 8 GPUs, so 32)
    over its GPUs, with Kavier's predictions."""
    recs = _min_gpu_strategy(batch_sizes=[2, 8], total_gpus=[1, 2], top_k=10).recommend(test_workload, test_context)

    # max_gpus is 16, so the doubling stops there.
    assert [(r.total_gpus, r.metadata["batch_size"]) for r in recs] == [(1, 32), (2, 16), (4, 8), (8, 4), (16, 2)]
    assert all(r.predicted_throughput and r.predicted_throughput > 0 for r in recs)


def test_energy_preset_diverges_toward_lower_power_than_performance_preset(test_workload, test_context):
    """The performance preset picks a faster config on more GPUs, and the energy preset one that
    draws less total power (per-GPU watts x total_gpus, as in selection._power_cost)."""
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
