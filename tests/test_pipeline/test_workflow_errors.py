"""How GridWorkflowPipeline reports an empty result, and which runtime guard values it accepts.

If every candidate fails the feasibility check, the error is "no feasible candidates". If the
predictor gives no usable number for the candidates that pass, a NoPredictionError names the
predictor and its reason; when only the power is missing, it names the power predictor.
"""

from __future__ import annotations

import math
from typing import Optional

import pytest

from coastline.sdk.exceptions import NoPredictionError, PredictionError
from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.recommendation import Prediction
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.pipeline.feasibility import NoOpFeasibilityChecker
from coastline.sdk.pipeline.workflow import GridWorkflowPipeline
from coastline.sdk.policies import PolicyFactory
from coastline.sdk.predictors.energy import KavierPowerPredictor

_GRID = {"grid": {"batch_sizes": [8], "total_gpus": [1, 2, 4], "top_k": 3}}


class _Predictor:
    """Throughput and power for every candidate, or the given failure for every candidate."""

    def __init__(self, *, error_detail: Optional[str] = None, returns_none: bool = False, throughput: float = 100.0):
        self.error_detail = error_detail
        self.returns_none = returns_none
        self.throughput = throughput

    def predict(self, workload: WorkloadSpec, context: SystemContext) -> Optional[Prediction]:
        if self.returns_none:
            return None
        failed = self.error_detail is not None
        return Prediction(
            gpus_per_node=workload.gpus_per_node or 1,
            number_of_nodes=workload.number_of_nodes or 1,
            total_gpus=workload.total_gpus,
            predicted_throughput=None if failed else self.throughput * workload.total_gpus,
            predicted_power=None if failed else 200.0,
            metadata={"error": "invalid_input", "error_detail": self.error_detail} if failed else {},
        )

    def get_name(self) -> str:
        return "scripted"


class _ThroughputOnly:
    """A throughput model with no power, like xgboost: it gives a number even for a method it never
    saw, so the power predictor runs."""

    def predict(self, workload: WorkloadSpec, context: SystemContext) -> Optional[Prediction]:
        return Prediction(
            gpus_per_node=workload.gpus_per_node or 1,
            number_of_nodes=workload.number_of_nodes or 1,
            total_gpus=workload.total_gpus,
            predicted_throughput=100.0 * workload.total_gpus,
        )

    def get_name(self) -> str:
        return "throughput only"


class _RejectAll:
    def is_feasible(self, workload: WorkloadSpec):
        return False, {"reason": "out of memory"}


@pytest.fixture
def context() -> SystemContext:
    return SystemContext(
        available_gpu_models=["NVIDIA-A100-SXM4-80GB"],
        max_gpus=8,
        gpu_memory={"NVIDIA-A100-SXM4-80GB": 80},
        constraints=Constraints(max_gpus=8, gpus_per_node=8, max_nodes=1),
    )


@pytest.fixture
def workload() -> WorkloadSpec:
    return WorkloadSpec(
        llm_model="mistral-7b-v0.1",
        fine_tuning_method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=8,
    )


def _pipeline(predictor, checker=None, **kw) -> GridWorkflowPipeline:
    return GridWorkflowPipeline.from_config(
        config=_GRID,
        selection_policy="balanced",
        strategy_name="multi_objective_balanced",
        throughput_predictor=predictor,
        power_predictor=predictor,
        feasibility_checker=checker or NoOpFeasibilityChecker(),
        **kw,
    )


# No prediction vs. not feasible
def test_a_predictor_failure_carries_the_predictor_reason(workload, context):
    pipeline = _pipeline(_Predictor(error_detail="unknown method 'LoRA'; valid methods: full, lora"))
    with pytest.raises(NoPredictionError) as excinfo:
        pipeline.recommend(workload, context)
    message = str(excinfo.value)
    assert "unknown method 'LoRA'; valid methods: full, lora" in message
    assert "3 configurations that passed the feasibility check" in message
    assert "no feasible candidates" not in message


def test_a_predictor_returning_nothing_is_not_reported_as_infeasible(workload, context):
    pipeline = _pipeline(_Predictor(returns_none=True))
    with pytest.raises(NoPredictionError, match="no usable prediction") as excinfo:
        pipeline.recommend(workload, context)
    assert "no feasible candidates" not in str(excinfo.value)
    assert "_Predictor" in str(excinfo.value)  # names the predictor that gave nothing


@pytest.mark.parametrize("bad", [math.nan, math.inf, 0.0])
def test_unusable_numbers_are_reported_as_no_prediction(workload, context, bad):
    pipeline = _pipeline(_Predictor(throughput=bad))
    with pytest.raises(NoPredictionError, match="no usable prediction"):
        pipeline.recommend(workload, context)


def test_no_prediction_error_is_a_prediction_error_and_a_runtime_error():
    # A RuntimeError too, so a caller that catches the pipeline's empty-result RuntimeError
    # (the UI maps it to a 404) still catches this case.
    assert issubclass(NoPredictionError, PredictionError)
    assert issubclass(NoPredictionError, RuntimeError)


def test_every_candidate_infeasible_is_still_the_infeasibility_error(workload, context):
    pipeline = _pipeline(_Predictor(), checker=_RejectAll())
    with pytest.raises(RuntimeError, match="no feasible candidates found in grid of 3 configurations") as excinfo:
        pipeline.recommend(workload, context)
    assert not isinstance(excinfo.value, PredictionError)


def test_the_config_predictor_name_is_used_in_the_message(workload, context):
    # Built by PolicyFactory, the message names the predictor the way the caller spelled it.
    config = {
        "strategy": {"name": "multi_objective", "preset": "balanced"},
        "predictors": {"performance": "kavier", "energy": "kavier_power", "feasibility": "rules"},
        "grid": {"batch_sizes": [8], "total_gpus": [1, 2], "top_k": 2},
    }
    strategy = PolicyFactory.create_strategy(config=config)
    bad = workload.model_copy(update={"fine_tuning_method": "not-a-method"})
    with pytest.raises(NoPredictionError) as excinfo:
        strategy.recommend(bad, context)
    message = str(excinfo.value)
    assert "the kavier predictor" in message
    assert "not-a-method" in message


def test_a_power_failure_names_the_power_predictor_and_its_reason(workload, context):
    # The throughput is usable; Kavier's power model rejects the misspelled method.
    pipeline = GridWorkflowPipeline.from_config(
        config=_GRID,
        selection_policy="balanced",
        strategy_name="multi_objective_balanced",
        throughput_predictor=_ThroughputOnly(),
        power_predictor=KavierPowerPredictor(),
        feasibility_checker=NoOpFeasibilityChecker(),
    )
    lorra = workload.model_copy(update={"fine_tuning_method": "lorra"})

    with pytest.raises(NoPredictionError) as excinfo:
        pipeline.recommend(lorra, context)

    message = str(excinfo.value)
    assert "KavierPowerPredictor gave no usable prediction" in message
    assert "unknown method 'lorra'" in message
    assert "_ThroughputOnly" not in message


def test_a_power_failure_uses_the_config_name_of_the_power_predictor(workload, context):
    # As PolicyFactory builds it: the predictors block names the power model.
    pipeline = GridWorkflowPipeline.from_config(
        config={**_GRID, "predictors": {"energy": "kavier_power"}},
        selection_policy="balanced",
        strategy_name="multi_objective_balanced",
        throughput_predictor=_ThroughputOnly(),
        power_predictor=KavierPowerPredictor(),
        feasibility_checker=NoOpFeasibilityChecker(),
        components_from_config=True,
    )
    lorra = workload.model_copy(update={"fine_tuning_method": "lorra"})

    with pytest.raises(NoPredictionError) as excinfo:
        pipeline.recommend(lorra, context)

    message = str(excinfo.value)
    assert "the kavier_power predictor gave no usable prediction" in message
    assert "unknown method 'lorra'" in message


# Runtime guard values
@pytest.mark.parametrize("k", [0, 0.5, -1, math.nan, math.inf, -math.inf])
def test_a_runtime_guard_that_cannot_hold_is_rejected(k):
    # k < 1 can never be met and a non-finite k is no bound.
    with pytest.raises(ValueError, match="max_slowdown"):
        _pipeline(_Predictor(), runtime_guard_k=k)


def test_a_runtime_guard_from_the_config_is_checked_too():
    config = {"strategy": {"runtime_guard_k": 0.5}, **_GRID}
    with pytest.raises(ValueError, match="max_slowdown"):
        GridWorkflowPipeline.from_config(
            config=config,
            selection_policy="balanced",
            strategy_name="multi_objective_balanced",
            throughput_predictor=_Predictor(),
            power_predictor=_Predictor(),
            feasibility_checker=NoOpFeasibilityChecker(),
        )


@pytest.mark.parametrize("k", [1, 1.0, 2, "3"])
def test_a_runtime_guard_of_at_least_one_is_accepted(workload, context, k):
    pipeline = _pipeline(_Predictor(), runtime_guard_k=k)
    assert pipeline.runtime_guard_k == float(k)
    recs = pipeline.recommend(workload, context)
    fastest = max(r.predicted_throughput for r in recs)
    assert all(r.predicted_throughput >= fastest / float(k) for r in recs)


def test_no_runtime_guard_keeps_every_candidate(workload, context):
    pipeline = _pipeline(_Predictor(), runtime_guard_k=None)
    assert pipeline.runtime_guard_k is None
    assert len(pipeline.recommend(workload, context)) == 3
