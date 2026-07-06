"""Guard for the non-finite-prediction filter in GridWorkflowPipeline.recommend.

A predictor that returns NaN (or inf) throughput/power for ONE candidate must drop
only that candidate. Before the `not math.isfinite(x) or x <= 0` guard a bare `x <= 0`
let NaN through (`nan <= 0` is False) and let +inf through (`+inf <= 0` is False); a
single non-finite value then poisoned min-max normalization so EVERY candidate's
combined_score became NaN (or a leaked +inf pinned the normalization extreme and
crushed the others to 0) -> zero recommendations returned.

Oracle strategy: the stub predictor is deterministic (throughput = 100·total_gpus,
per-GPU power = 50·total_gpus) so the whole min-max ranking is hand-derivable. Each
test pins the survivor set AND the exact hand-computed scores/order, which is what
proves the surviving candidates' normalization was not poisoned by the dropped one.
"""

import pytest

from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.recommendation import Prediction
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.pipeline.feasibility import NoOpFeasibilityChecker
from coastline.sdk.pipeline.workflow import GridWorkflowPipeline
from coastline.sdk.predictors.base import BasePredictor

# The three non-finite kinds that the guard must all reject. They exercise DIFFERENT
# branches of `not math.isfinite(x) or x <= 0`: nan and +inf fail `isfinite` while
# passing `x <= 0` (`nan<=0`, `+inf<=0` are both False); -inf fails both. A naive
# `x <= 0` guard would catch only -inf, so all three are load-bearing cases.
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
    """Deterministic predictor: throughput = 100·total_gpus, per-GPU power = 50·total_gpus.
    For any total_gpus listed in ``poison`` the named field is overwritten with the
    supplied non-finite value. The linear-in-GPU outputs make the whole min-max
    ranking hand-derivable (see module docstring)."""

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
    # batch_sizes=[8] keeps one candidate per total_gpus, so a poisoned total_gpus
    # maps to exactly one candidate. top_k high enough to return every survivor.
    # No preset/alpha/beta in config => from_config defaults to balanced weights
    # alpha=beta=0.5 (this is what the hand-derived scores below assume).
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


def test_all_finite_baseline_ranks_by_hand_derived_minmax_scores(workload, context):
    """With no poison all three candidates survive and rank in the order fixed by the
    hand-computed min-max scores. This is the reference case: it proves the stub +
    harness produce exactly the grid {1,2,4} and the derivation below is the ground
    truth the poison tests reuse."""
    pipeline = _pipeline(_StubPredictor({}), _StubPredictor({}))
    recs = pipeline.recommend(workload, context)

    # Grid = {1,2,4}, all feasible (NoOp), all finite -> every candidate survives.
    assert {r.total_gpus for r in recs} == {1, 2, 4}

    # Hand derivation (alpha=beta=0.5), stub thr=100g, per-GPU power=50g:
    #   power_cost = power*g = 50*g^2 -> {1:50, 2:200, 4:800}
    #   time_cost  = 1/thr        -> {1:0.01, 2:0.005, 4:0.0025}
    #   power_score = (800-pc)/(800-50): {1:750/750=1.0, 2:600/750=0.8, 4:0.0}
    #   thr_score   = (0.01-tc)/(0.01-0.0025): {1:0.0, 2:0.005/0.0075=2/3, 4:1.0}
    #   combined = 0.5*power_score + 0.5*thr_score:
    #     1 -> 0.5*1.0 + 0.5*0.0   = 0.5
    #     2 -> 0.5*0.8 + 0.5*(2/3) = 0.4 + 1/3 = 11/15 ~ 0.73333
    #     4 -> 0.5*0.0 + 0.5*1.0   = 0.5
    by_gpu = _by_gpu(recs)
    assert by_gpu[2].metadata["combined_score"] == pytest.approx(11 / 15)
    assert by_gpu[1].metadata["combined_score"] == pytest.approx(0.5)
    assert by_gpu[4].metadata["combined_score"] == pytest.approx(0.5)
    assert by_gpu[1].metadata["power_score"] == pytest.approx(1.0)
    assert by_gpu[2].metadata["power_score"] == pytest.approx(0.8)
    assert by_gpu[4].metadata["throughput_score"] == pytest.approx(1.0)
    # 2 is the unique top (0.733); 1 and 4 tie at 0.5, broken toward higher
    # throughput -> gpu1(thr 100) before gpu4... no: tie-break sorts leaders only.
    # 2 leads alone; the 0.5 pair keeps stable grid order (1 before 4).
    assert [r.total_gpus for r in recs] == [2, 1, 4]


@NONFINITE
def test_nonfinite_throughput_drops_only_that_candidate(workload, context, bad):
    """A non-finite throughput on the 4-GPU candidate drops exactly that candidate;
    {1,2} survive and their scores match the two-candidate hand derivation, proving
    the poisoned value never entered normalization."""
    poison = {4: bad}
    pipeline = _pipeline(_StubPredictor(poison), _StubPredictor(poison))
    recs = pipeline.recommend(workload, context)

    returned = {r.total_gpus for r in recs}
    assert 4 not in returned, "the non-finite candidate must be dropped"
    assert returned == {1, 2}, "the two finite candidates must survive"

    # Two-candidate min-max (alpha=beta=0.5), thr=100g, per-GPU power=50g:
    #   power_cost -> {1:50, 2:200}; time_cost -> {1:0.01, 2:0.005}
    #   power_score = (200-pc)/(200-50): {1:1.0, 2:0.0}
    #   thr_score   = (0.01-tc)/(0.01-0.005): {1:0.0, 2:1.0}
    #   combined    = 0.5*1.0+0.5*0.0 = 0.5 (gpu1);  0.5*0.0+0.5*1.0 = 0.5 (gpu2)
    # A leaked NaN would make these NaN; a leaked +inf would pin an extreme and shift
    # them off 0.5 -> pinning 0.5 is the regression oracle.
    by_gpu = _by_gpu(recs)
    assert by_gpu[1].metadata["combined_score"] == pytest.approx(0.5)
    assert by_gpu[2].metadata["combined_score"] == pytest.approx(0.5)
    assert by_gpu[1].metadata["power_score"] == pytest.approx(1.0)
    assert by_gpu[2].metadata["throughput_score"] == pytest.approx(1.0)
    # Both exactly 0.5 -> tie-break orders by higher throughput: gpu2 before gpu1.
    assert [r.total_gpus for r in recs] == [2, 1]
    # tokens_per_watt = thr/power = 100g/50g = 2.0 for every candidate (invariant).
    for r in recs:
        assert r.metadata["tokens_per_watt"] == pytest.approx(2.0)


@NONFINITE
def test_nonfinite_power_drops_only_that_candidate(workload, context, bad):
    """The same guard protects the power axis. Throughput is clean everywhere; power
    is non-finite for the 4-GPU candidate. Distinct predictors force the power
    predictor to be consulted (the throughput stub does not set
    WRAPS_THROUGHPUT_ENGINE)."""
    pipeline = _pipeline(
        _StubPredictor({}),
        _StubPredictor({4: bad}, field="predicted_power"),
    )
    recs = pipeline.recommend(workload, context)

    returned = {r.total_gpus for r in recs}
    assert 4 not in returned, "the non-finite-power candidate must be dropped"
    assert returned == {1, 2}

    # Identical two-candidate derivation as the throughput case -> both 0.5, order [2,1].
    by_gpu = _by_gpu(recs)
    assert by_gpu[1].metadata["combined_score"] == pytest.approx(0.5)
    assert by_gpu[2].metadata["combined_score"] == pytest.approx(0.5)
    assert [r.total_gpus for r in recs] == [2, 1]
    # Surviving candidates carry their real per-GPU power (50g), never the poison.
    assert by_gpu[1].metadata["predicted_power_watts"] == pytest.approx(50.0)
    assert by_gpu[2].metadata["predicted_power_watts"] == pytest.approx(100.0)


@NONFINITE
def test_all_candidates_nonfinite_throughput_raises_rather_than_empty(workload, context, bad):
    """When EVERY candidate's throughput is non-finite the feasible set is empty after
    filtering and recommend() must raise RuntimeError (never silently return [])."""
    poison = {1: bad, 2: bad, 4: bad}
    pipeline = _pipeline(_StubPredictor(poison), _StubPredictor(poison))

    with pytest.raises(RuntimeError, match="no feasible candidates"):
        pipeline.recommend(workload, context)
