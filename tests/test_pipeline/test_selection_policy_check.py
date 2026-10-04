"""GridWorkflowPipeline checks its selection policy when it is built.

min_gpu, performance, energy and balanced are the policies, in any letter case. Any other value
raises ValueError before a candidate is checked or simulated, with a short message that lists
the valid ones. rank_candidates names only the policies it ranks, and mentions min_gpu only when
it is given min_gpu.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from coastline.sdk.constants import SelectionPolicy
from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.recommendation import Prediction
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.pipeline.selection import EvaluatedCandidate, rank_candidates
from coastline.sdk.pipeline.workflow import GridWorkflowPipeline

GPU = "NVIDIA-A100-SXM4-80GB"


class Counting:
    """Feasibility checker and predictor that count their calls."""

    def __init__(self) -> None:
        self.checks = 0
        self.predictions = 0

    def is_feasible(self, workload: WorkloadSpec) -> tuple[bool, dict[str, Any]]:
        self.checks += 1
        return True, {}

    def predict(self, workload: WorkloadSpec, context: SystemContext) -> Optional[Prediction]:
        self.predictions += 1
        return Prediction(
            gpus_per_node=workload.gpus_per_node or 1,
            number_of_nodes=workload.number_of_nodes or 1,
            total_gpus=workload.total_gpus,
            predicted_throughput=100.0 * workload.total_gpus,
            predicted_power=100.0,
        )

    def get_name(self) -> str:  # pragma: no cover - trivial
        return "counting"


def _build(policy: Any, counter: Counting) -> GridWorkflowPipeline:
    return GridWorkflowPipeline.from_config(
        config={"grid": {"batch_sizes": [4, 8], "total_gpus": [1, 2, 4], "top_k": 2}},
        selection_policy=policy,
        strategy_name="test",
        throughput_predictor=counter,
        power_predictor=counter,
        feasibility_checker=counter,
    )


def _workload() -> WorkloadSpec:
    return WorkloadSpec(
        llm_model="mistral-7b-v0.1", fine_tuning_method="lora", gpu_model=GPU, tokens_per_sample=1024, batch_size=8
    )


def _context() -> SystemContext:
    return SystemContext(
        available_gpu_models=[GPU],
        max_gpus=8,
        gpu_memory={GPU: 80},
        constraints=Constraints(max_gpus=8, gpus_per_node=8, max_nodes=1),
    )


@pytest.mark.parametrize("policy", ["multi_objective", "custom", "", "fewest"])
def test_an_unknown_policy_fails_when_the_pipeline_is_built(policy):
    counter = Counting()
    with pytest.raises(ValueError) as excinfo:
        _build(policy, counter)

    assert str(excinfo.value) == (
        f"unknown selection policy {policy!r}; choose from min_gpu, performance, energy, balanced"
    )
    assert (counter.checks, counter.predictions) == (0, 0)


@pytest.mark.parametrize(
    "policy,expected",
    [
        ("Balanced", SelectionPolicy.BALANCED),
        (" PERFORMANCE ", SelectionPolicy.PERFORMANCE),
        ("Min_GPU", SelectionPolicy.MIN_GPU),
        (SelectionPolicy.ENERGY, SelectionPolicy.ENERGY),
    ],
)
def test_a_policy_in_any_letter_case_is_accepted(policy, expected):
    pipeline = _build(policy, Counting())
    assert pipeline.selection_policy == expected
    assert pipeline.recommend(_workload(), _context())


def _candidate() -> EvaluatedCandidate:
    return EvaluatedCandidate(
        gpus_per_node=1,
        number_of_nodes=1,
        total_gpus=1,
        throughput=100.0,
        power=100.0,
        runtime=None,
        throughput_score=1.0,
        power_score=1.0,
        combined_score=0.0,
        feasibility_metadata={},
    )


def test_rank_candidates_lists_the_policies_it_ranks():
    with pytest.raises(ValueError) as excinfo:
        rank_candidates([_candidate()], "custom", top_k=1)
    assert str(excinfo.value) == "unknown policy 'custom'; choose from energy, balanced, performance"


def test_rank_candidates_says_what_min_gpu_does():
    with pytest.raises(ValueError) as excinfo:
        rank_candidates([_candidate()], SelectionPolicy.MIN_GPU, top_k=1)
    message = str(excinfo.value)
    assert message.startswith("rank_candidates ranks energy, balanced or performance;")
    assert "min_gpu picks the first feasible GPU count" in message
    assert "_recommend_min_gpu" not in message
