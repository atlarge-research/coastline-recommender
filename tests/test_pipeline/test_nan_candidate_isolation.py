"""The non-finite prediction filter in GridWorkflowPipeline.recommend.

A NaN or infinite throughput or power on one candidate drops only that candidate. The filter is
``not math.isfinite(x) or x <= 0``: ``x <= 0`` alone lets NaN and +inf through, and one such
value turns every combined_score into NaN (or pushes the others to 0) in the min-max normalization.

The stub predictor gives throughput = 100 * total_gpus and per-GPU power = 50 * total_gpus, so
each test can check the exact scores and order of the surviving candidates.
"""

import pytest

from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.recommendation import Prediction
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.pipeline.feasibility import NoOpFeasibilityChecker
from coastline.sdk.pipeline.workflow import GridWorkflowPipeline
from coastline.sdk.predictors.base import BasePredictor

# nan and +inf fail only the isfinite check (nan <= 0 and +inf <= 0 are both False);
# -inf fails both checks.
NONFINITE = pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "pos_inf", "neg_inf"],
)


@pytest.fixture
def workload():
    return WorkloadSpec(
        llm_model="mistral-7b-v0.1",
        fine_tuning_method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=8,
    )


@pytest.fixture
def context():
    return SystemContext(
        available_gpu_models=["NVIDIA-A100-SXM4-80GB"],
        max_gpus=16,
        gpu_memory={"NVIDIA-A100-SXM4-80GB": 80},
        constraints=Constraints(max_gpus=16, gpus_per_node=8, max_nodes=2),
    )


class _StubPredictor(BasePredictor):
    """Throughput = 100 * total_gpus and per-GPU power = 50 * total_gpus.

    For each total_gpus in ``poison`` the chosen field is set to the given value."""

    def __init__(self, poison: dict[int, float], *, field: str = "predicted_throughput"):
        self._poison = poison
        self._field = field

    def predict(self, workload: WorkloadSpec, context: SystemContext) -> Prediction:
        total = workload.total_gpus
        throughput = 100.0 * total  # strictly increasing in GPUs
        power = 50.0 * total
        if total in self._poison:
            bad = self._poison[total]
            if self._field == "predicted_throughput":
                throughput = bad
            else:
                power = bad
        return Prediction(
            gpus_per_node=workload.gpus_per_node or 1,
            number_of_nodes=workload.number_of_nodes or 1,
            total_gpus=total,
            predicted_throughput=throughput,
            predicted_power=power,
            predicted_runtime_seconds=1000.0 / total,
        )

    def get_name(self) -> str:
        return "stub"


def _pipeline(throughput_predictor, power_predictor):
    # One batch size gives one candidate per total_gpus, and top_k=5 returns every survivor.
    # With no preset or weights in the config, from_config uses balanced weights (alpha = beta = 0.5).
    return GridWorkflowPipeline.from_config(
        config={"grid": {"batch_sizes": [8], "total_gpus": [1, 2, 4], "top_k": 5}},
        selection_policy="performance",
        strategy_name="test",
        throughput_predictor=throughput_predictor,
        power_predictor=power_predictor,
        feasibility_checker=NoOpFeasibilityChecker(),
    )


def _by_gpu(recs):
    return {r.total_gpus: r for r in recs}


def test_all_finite_baseline_ranks_by_worked_out_minmax_scores(workload, context):
    """Without bad values all three candidates survive, with the min-max scores worked out below."""
    pipeline = _pipeline(_StubPredictor({}), _StubPredictor({}))
    recs = pipeline.recommend(workload, context)

    # Grid {1, 2, 4}, all feasible and finite.
    assert {r.total_gpus for r in recs} == {1, 2, 4}

    # With alpha = beta = 0.5, throughput 100g and per-GPU power 50g (g = GPUs), for 1, 2 and 4 GPUs:
    #   power_cost = 50 * g^2: 50, 200, 800; time_cost = 1 / throughput: 0.01, 0.005, 0.0025
    #   power_score = (800 - power_cost) / 750: 1.0, 0.8, 0.0
    #   thr_score = (0.01 - time_cost) / 0.0075: 0.0, 2/3, 1.0
    #   combined = 0.5 * power_score + 0.5 * thr_score: 0.5, 0.4 + 1/3 = 11/15, 0.5
    by_gpu = _by_gpu(recs)
    assert by_gpu[2].metadata["combined_score"] == pytest.approx(11 / 15)
    assert by_gpu[1].metadata["combined_score"] == pytest.approx(0.5)
    assert by_gpu[4].metadata["combined_score"] == pytest.approx(0.5)
    assert by_gpu[1].metadata["power_score"] == pytest.approx(1.0)
    assert by_gpu[2].metadata["power_score"] == pytest.approx(0.8)
    assert by_gpu[4].metadata["throughput_score"] == pytest.approx(1.0)
    # 2 GPUs lead alone. The tie-break covers only scores within TIE_EPS of the top, so 1 and
    # 4 GPUs (both 0.5) keep grid order.
    assert [r.total_gpus for r in recs] == [2, 1, 4]


@NONFINITE
def test_nonfinite_throughput_drops_only_that_candidate(workload, context, bad):
    """A non-finite throughput on the 4-GPU candidate drops only that candidate, and the 1- and
    2-GPU candidates get the scores of a two-candidate grid."""
    poison = {4: bad}
    pipeline = _pipeline(_StubPredictor(poison), _StubPredictor(poison))
    recs = pipeline.recommend(workload, context)

    returned = {r.total_gpus for r in recs}
    assert 4 not in returned, "the non-finite candidate must be dropped"
    assert returned == {1, 2}, "the two finite candidates must survive"

    # Over two candidates power_score is 1.0 and 0.0 and thr_score 0.0 and 1.0, so both combine
    # to 0.5. A NaN or +inf left in the normalization would change these values.
    by_gpu = _by_gpu(recs)
    assert by_gpu[1].metadata["combined_score"] == pytest.approx(0.5)
    assert by_gpu[2].metadata["combined_score"] == pytest.approx(0.5)
    assert by_gpu[1].metadata["power_score"] == pytest.approx(1.0)
    assert by_gpu[2].metadata["throughput_score"] == pytest.approx(1.0)
    # Both score 0.5, and the tie-break puts the higher throughput (2 GPUs) first.
    assert [r.total_gpus for r in recs] == [2, 1]
    # tokens_per_watt = 100g / 50g = 2.0 for every candidate.
    for r in recs:
        assert r.metadata["tokens_per_watt"] == pytest.approx(2.0)


@NONFINITE
def test_nonfinite_power_drops_only_that_candidate(workload, context, bad):
    """A non-finite power on the 4-GPU candidate drops only that candidate.

    Separate stubs are used so the power predictor is called (the throughput stub does not set
    WRAPS_THROUGHPUT_ENGINE)."""
    pipeline = _pipeline(
        _StubPredictor({}),
        _StubPredictor({4: bad}, field="predicted_power"),
    )
    recs = pipeline.recommend(workload, context)

    returned = {r.total_gpus for r in recs}
    assert 4 not in returned, "the non-finite-power candidate must be dropped"
    assert returned == {1, 2}

    # The same scores and order as in the throughput case.
    by_gpu = _by_gpu(recs)
    assert by_gpu[1].metadata["combined_score"] == pytest.approx(0.5)
    assert by_gpu[2].metadata["combined_score"] == pytest.approx(0.5)
    assert [r.total_gpus for r in recs] == [2, 1]
    # The survivors keep their per-GPU power of 50g.
    assert by_gpu[1].metadata["predicted_power_watts"] == pytest.approx(50.0)
    assert by_gpu[2].metadata["predicted_power_watts"] == pytest.approx(100.0)


@NONFINITE
def test_all_candidates_nonfinite_throughput_raises_rather_than_empty(workload, context, bad):
    """If every throughput is non-finite, recommend() raises a RuntimeError saying the predictor
    gave no usable prediction."""
    poison = {1: bad, 2: bad, 4: bad}
    pipeline = _pipeline(_StubPredictor(poison), _StubPredictor(poison))

    with pytest.raises(RuntimeError, match="no usable prediction"):
        pipeline.recommend(workload, context)
