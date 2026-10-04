"""Tests for the workflow of grid, feasibility, simulation and selection.

The expected grids and scores are worked out in the comments. Kavier is checked only through
invariants: positive throughput, power within [idle, TDP] and sub-linear scaling.
"""

import math
from collections import Counter
from unittest.mock import MagicMock

import pytest

from coastline.sdk.library.hardware import get_gpu_idle_power, get_gpu_tdp
from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.recommendation import Prediction
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.pipeline.feasibility import NoOpFeasibilityChecker, RulesFeasibilityChecker
from coastline.sdk.pipeline.grid import generate_candidates, grid_config_from_dict
from coastline.sdk.pipeline.selection import (
    PRESET_WEIGHTS,
    EvaluatedCandidate,
    normalize_candidates,
    rank_candidates,
)
from coastline.sdk.pipeline.workflow import GridWorkflowPipeline
from coastline.sdk.predictors.energy import KavierPowerPredictor
from coastline.sdk.predictors.performance.physics import KavierPredictor

GPU = "NVIDIA-A100-SXM4-80GB"


@pytest.fixture
def workload():
    return WorkloadSpec(
        llm_model="mistral-7b-v0.1",
        fine_tuning_method="lora",
        gpu_model=GPU,
        tokens_per_sample=1024,
        batch_size=8,
    )


@pytest.fixture
def context():
    return SystemContext(
        available_gpu_models=[GPU],
        max_gpus=16,
        gpu_memory={GPU: 80},
        constraints=Constraints(max_gpus=16, gpus_per_node=8, max_nodes=2),
    )


def _cand(*, total_gpus, throughput=100.0, power=100.0, throughput_score=0.0, power_score=0.0, batch_size=0):
    """EvaluatedCandidate with a self-consistent (gpus_per_node, nodes) that multiplies to total_gpus."""
    return EvaluatedCandidate(
        gpus_per_node=total_gpus,
        number_of_nodes=1,
        total_gpus=total_gpus,
        throughput=throughput,
        power=power,
        runtime=None,
        throughput_score=throughput_score,
        power_score=power_score,
        combined_score=0.0,
        feasibility_metadata={},
        batch_size=batch_size,
    )


# grid


def test_grid_enumerates_batch_by_gpu_with_node_packing(workload, context):
    grid = grid_config_from_dict({"grid": {"batch_sizes": [4, 8, 16], "total_gpus": [1, 2, 4, 8, 16]}})
    candidates = generate_candidates(workload, context, grid)

    # 3 batch sizes x 5 GPU counts. All fit max_gpus=16 and max_nodes=2 (16 GPUs is 2 nodes of
    # 8), so all 15 remain, 3 per GPU count.
    assert len(candidates) == 15
    assert Counter(c.total_gpus for c in candidates) == {1: 3, 2: 3, 4: 3, 8: 3, 16: 3}

    # Up to 8 GPUs per node: 16 GPUs is (8, 2), 8 GPUs is (8, 1), and fewer stay on one node.
    by_total = {c.total_gpus: (c.gpus_per_node, c.number_of_nodes) for c in candidates}
    assert by_total[16] == (8, 2)
    assert by_total[8] == (8, 1)
    assert by_total[4] == (4, 1)
    assert by_total[1] == (1, 1)


def test_grid_drops_total_gpus_exceeding_max_gpus(workload, context):
    # max_gpus=16 comes from the context, so 32 is dropped.
    grid = grid_config_from_dict({"grid": {"batch_sizes": [8], "total_gpus": [8, 16, 32]}})
    candidates = generate_candidates(workload, context, grid)

    assert {c.total_gpus for c in candidates} == {8, 16}  # 32 is over max_gpus
    assert len(candidates) == 2  # one batch size x two GPU counts


def test_grid_drops_layout_exceeding_max_nodes(workload):
    # 16 GPUs fit max_gpus=32 but need 2 nodes of 8, more than max_nodes=1.
    ctx = SystemContext(
        available_gpu_models=[GPU],
        max_gpus=32,
        gpu_memory={GPU: 80},
        constraints=Constraints(max_gpus=32, gpus_per_node=8, max_nodes=1),
    )
    grid = grid_config_from_dict({"grid": {"batch_sizes": [8], "total_gpus": [8, 16]}})
    candidates = generate_candidates(workload, ctx, grid)

    assert {c.total_gpus for c in candidates} == {8}  # 16 needs 2 nodes > max_nodes=1


# normalize


def test_normalize_candidates_minmax_power_and_time_scores():
    # power_cost = power * total_gpus: 100, 200, 400. time_cost = 1 / throughput: 0.01, 1/300, 0.0025.
    c1 = _cand(total_gpus=1, power=100, throughput=100)
    c2 = _cand(total_gpus=2, power=100, throughput=300)
    c3 = _cand(total_gpus=4, power=100, throughput=400)
    normalize_candidates([c1, c2, c3], "grid")

    # power_score = (400 - power_cost) / 300: 1.0, 2/3, 0.0
    assert c1.power_score == pytest.approx(1.0)
    assert c2.power_score == pytest.approx(2 / 3)
    assert c3.power_score == pytest.approx(0.0)

    # throughput_score = (0.01 - time_cost) / 0.0075: 0.0, 0.8889, 1.0
    assert c1.throughput_score == pytest.approx(0.0)
    assert c2.throughput_score == pytest.approx((0.01 - 1 / 300) / 0.0075)
    assert c3.throughput_score == pytest.approx(1.0)


def test_normalize_frontier_marks_pareto_dominated():
    # Lower power_cost and time_cost are better. b (power_cost 100, time_cost 0.005) dominates
    # a (same power, slower) and c (power_cost 200, time_cost 0.02).
    a = _cand(total_gpus=1, power=100, throughput=100)
    b = _cand(total_gpus=1, power=100, throughput=200)
    c = _cand(total_gpus=2, power=100, throughput=50)
    normalize_candidates([a, b, c], "frontier")

    assert (a.dominated, b.dominated, c.dominated) == (True, False, True)
    # Dominated candidates get zero scores, so ranking skips them; the survivor keeps its score.
    assert a.power_score == 0.0 and a.throughput_score == 0.0
    assert c.power_score == 0.0 and c.throughput_score == 0.0
    assert b.power_score == pytest.approx(1.0)  # the only point on the frontier


# weighted sum


def test_rank_energy_preset_weights_power_heavily():
    # combined = alpha * throughput_score + beta * power_score, with energy weights 0.2 and 0.8.
    a = _cand(total_gpus=1, throughput=100, power_score=1.0, throughput_score=0.0)
    b = _cand(total_gpus=2, throughput=300, power_score=0.6, throughput_score=0.9)
    c = _cand(total_gpus=4, throughput=400, power_score=0.0, throughput_score=1.0)
    ranked = rank_candidates([a, b, c], "energy", alpha=0.2, beta=0.8, top_k=3)

    # a = 0.8, b = 0.18 + 0.48 = 0.66, c = 0.2
    assert a.combined_score == pytest.approx(0.80)
    assert b.combined_score == pytest.approx(0.66)
    assert c.combined_score == pytest.approx(0.20)
    # Energy favours the lowest-power config even though it is the slowest.
    assert [r.total_gpus for r in ranked] == [1, 2, 4]


def test_rank_performance_preset_penalizes_extra_gpus():
    # The same candidates with performance weights alpha=0.8, beta=0.2.
    a = _cand(total_gpus=1, throughput=100, power_score=1.0, throughput_score=0.0)
    b = _cand(total_gpus=2, throughput=300, power_score=0.6, throughput_score=0.9)
    c = _cand(total_gpus=4, throughput=400, power_score=0.0, throughput_score=1.0)
    ranked = rank_candidates([a, b, c], "performance", alpha=0.8, beta=0.2, top_k=3)

    # a = 0.2, b = 0.72 + 0.12 = 0.84, c = 0.8
    assert a.combined_score == pytest.approx(0.20)
    assert b.combined_score == pytest.approx(0.84)
    assert c.combined_score == pytest.approx(0.80)
    # b beats the faster c, which loses on power_cost.
    assert [r.total_gpus for r in ranked] == [2, 4, 1]


def test_rank_breaks_near_ties_toward_higher_throughput():
    # With alpha = beta = 0.5, x scores 0.500 and y 0.495, within TIE_EPS = 0.01 of x, so the
    # tie-break puts y (throughput 500) before x (throughput 100).
    x = _cand(total_gpus=2, throughput=100, power_score=0.5, throughput_score=0.5)
    y = _cand(total_gpus=2, throughput=500, power_score=0.49, throughput_score=0.5)
    ranked = rank_candidates([x, y], "balanced", alpha=0.5, beta=0.5, top_k=2)

    assert x.combined_score == pytest.approx(0.500)
    assert y.combined_score == pytest.approx(0.495)
    assert ranked[0].throughput == 500  # y wins the near-tie on throughput


def test_preset_weights_match_documented_spec():
    # Presets are (runtime weight, energy weight), as in the thesis.
    assert PRESET_WEIGHTS["energy"] == (0.2, 0.8)
    assert PRESET_WEIGHTS["balanced"] == (0.5, 0.5)
    assert PRESET_WEIGHTS["performance"] == (0.8, 0.2)


# feasibility


@pytest.mark.parametrize(
    "gpus_per_node, number_of_nodes, expected",
    [
        # batch_size is per device and need not divide the GPU count, so batch 8 is feasible
        # on every layout, 3 and 16 GPUs included.
        (1, 1, True),  # total 1
        (2, 1, True),  # total 2
        (4, 1, True),  # total 4
        (8, 1, True),  # total 8
        (3, 1, True),  # total 3
        (8, 2, True),  # total 16
    ],
)
def test_rules_feasibility_admits_any_per_device_batch(gpus_per_node, number_of_nodes, expected):
    w = WorkloadSpec(
        llm_model="mistral-7b-v0.1",
        fine_tuning_method="lora",
        gpu_model=GPU,
        tokens_per_sample=1024,
        batch_size=8,
        gpus_per_node=gpus_per_node,
        number_of_nodes=number_of_nodes,
    )
    feasible, _meta = RulesFeasibilityChecker().is_feasible(w)
    assert feasible is expected


def test_rules_feasibility_keeps_all_per_device_configs_in_pipeline(workload, context):
    # Grid [1, 3] with per-device batch 8: the rules checker keeps both GPU counts.
    pred = Prediction(
        gpus_per_node=1,
        number_of_nodes=1,
        total_gpus=1,
        predicted_throughput=100.0,
        predicted_power=50.0,
    )
    pipeline = GridWorkflowPipeline.from_config(
        config={"grid": {"batch_sizes": [8], "total_gpus": [1, 3], "top_k": 3}},
        selection_policy="performance",
        strategy_name="test",
        throughput_predictor=MagicMock(predict=MagicMock(return_value=pred)),
        power_predictor=MagicMock(predict=MagicMock(return_value=pred)),
        feasibility_checker=RulesFeasibilityChecker(),
    )
    recs = pipeline.recommend(workload, context)

    assert len(recs) == 2  # 1 and 3 GPUs


# Kavier integration


def test_workflow_min_gpu_end_to_end_picks_fewest_gpus(workload, context):
    pipeline = GridWorkflowPipeline.from_config(
        config={"grid": {"batch_sizes": [8], "total_gpus": [1, 2], "top_k": 1}},
        selection_policy="min_gpu",
        strategy_name="min_gpu",
        throughput_predictor=KavierPredictor(),
        power_predictor=KavierPowerPredictor(),
        feasibility_checker=NoOpFeasibilityChecker(),
    )
    recs = pipeline.recommend(workload, context)

    # Every GPU count is feasible, so min_gpu returns one recommendation, on 1 GPU.
    assert len(recs) == 1
    assert recs[0].total_gpus == 1
    assert recs[0].strategy == "min_gpu"
    assert recs[0].metadata["workflow"] == "min_gpu_doubling_feasibility_simulate"
    # A supported config has a finite, positive throughput.
    assert recs[0].predicted_throughput is not None
    assert math.isfinite(recs[0].predicted_throughput) and recs[0].predicted_throughput > 0


def test_kavier_throughput_scales_sublinearly_with_gpus(context):
    # More GPUs raise the total throughput, but communication keeps it below N times that of
    # one GPU.
    predictor = KavierPredictor()

    def thr(n):
        w = WorkloadSpec(
            llm_model="mistral-7b-v0.1",
            fine_tuning_method="lora",
            gpu_model=GPU,
            tokens_per_sample=1024,
            batch_size=8,
            gpus_per_node=n,
            number_of_nodes=1,
        )
        return predictor.predict(w, context).predicted_throughput

    t1, t2, t4 = thr(1), thr(2), thr(4)
    assert t1 < t2 < t4  # rises with GPU count
    assert t2 < 2 * t1  # sub-linear
    assert t4 < 4 * t1  # sub-linear


def test_kavier_power_per_gpu_within_idle_and_tdp(workload, context):
    # Per-GPU power lies in [idle, TDP] from the hardware library (75 W and 400 W for the
    # A100-SXM4-80GB).
    idle = get_gpu_idle_power(GPU)
    tdp = get_gpu_tdp(GPU)
    pred = KavierPredictor().predict(workload, context)

    assert idle <= pred.predicted_power <= tdp


def test_kavier_power_predictor_agrees_with_throughput_engine(workload, context):
    # KavierPowerPredictor wraps the same engine (WRAPS_THROUGHPUT_ENGINE), so it reports the
    # per-GPU watts the throughput predictor computed.
    thr_power = KavierPredictor().predict(workload, context).predicted_power
    energy_power = KavierPowerPredictor().predict(workload, context).predicted_power

    assert energy_power == pytest.approx(thr_power)
    assert KavierPowerPredictor.WRAPS_THROUGHPUT_ENGINE is True


def test_kavier_unsupported_model_returns_error_prediction(context):
    # For an unknown model the throughput predictor returns an error Prediction with no
    # throughput, and the power predictor passes it on with no power.
    bogus = WorkloadSpec(
        llm_model="not-a-real-model",
        fine_tuning_method="lora",
        gpu_model=GPU,
        tokens_per_sample=1024,
        batch_size=8,
        gpus_per_node=1,
        number_of_nodes=1,
    )
    pred = KavierPredictor().predict(bogus, context)
    assert pred is not None
    assert pred.predicted_throughput is None
    assert pred.metadata["error"] == "unsupported_config"

    power = KavierPowerPredictor().predict(bogus, context)
    assert power is not None and power.predicted_power is None
    assert power.metadata["error"] == "unsupported_config"


# tie-breaking
#
# Candidates within TIE_EPS of the top score are ordered by highest predicted throughput, then
# fewest GPUs, then smallest batch. This is a total order, so grid order does not matter.


def _tied(**kwargs):
    """A candidate with the same score components as its siblings."""
    return _cand(throughput_score=1.0, power_score=1.0, **kwargs)


def test_a_tie_prefers_the_highest_predicted_throughput():
    slow = _tied(total_gpus=4, throughput=100.0)
    fast = _tied(total_gpus=4, throughput=400.0)

    # Given slowest first, so the result does not come from input order.
    ranked = rank_candidates([slow, fast], "balanced", alpha=0.5, beta=0.5, top_k=2)

    assert [c.throughput for c in ranked] == [400.0, 100.0]


def test_a_tie_on_throughput_prefers_the_fewest_gpus():
    many = _tied(total_gpus=8, throughput=200.0)
    few = _tied(total_gpus=2, throughput=200.0)

    ranked = rank_candidates([many, few], "balanced", alpha=0.5, beta=0.5, top_k=2)

    assert [c.total_gpus for c in ranked] == [2, 8]


def test_a_tie_on_throughput_and_gpus_prefers_the_smallest_batch():
    big = _tied(total_gpus=4, throughput=200.0, batch_size=64)
    small = _tied(total_gpus=4, throughput=200.0, batch_size=8)

    ranked = rank_candidates([big, small], "balanced", alpha=0.5, beta=0.5, top_k=2)

    assert [c.batch_size for c in ranked] == [8, 64]


def test_the_three_levels_apply_in_order():
    # Throughput comes before GPU count and GPU count before batch size, so the fastest wins
    # although it asks for the most GPUs and the biggest batch.
    fastest_but_greedy = _tied(total_gpus=8, throughput=400.0, batch_size=64)
    fewest_gpus = _tied(total_gpus=1, throughput=100.0, batch_size=8)
    middle = _tied(total_gpus=2, throughput=200.0, batch_size=16)

    ranked = rank_candidates([fewest_gpus, middle, fastest_but_greedy], "balanced", top_k=3)

    assert [c.throughput for c in ranked] == [400.0, 200.0, 100.0]


def test_the_tie_break_does_not_depend_on_the_order_the_grid_enumerated():
    # Every permutation of the same tied set ranks the same way.
    import itertools

    def fresh():
        return [
            _tied(total_gpus=4, throughput=200.0, batch_size=8),
            _tied(total_gpus=4, throughput=200.0, batch_size=32),
            _tied(total_gpus=2, throughput=200.0, batch_size=32),
            _tied(total_gpus=2, throughput=500.0, batch_size=64),
        ]

    expected = [(c.throughput, c.total_gpus, c.batch_size) for c in rank_candidates(fresh(), "balanced", top_k=4)]
    for order in itertools.permutations(range(4)):
        shuffled = [fresh()[i] for i in order]
        got = [(c.throughput, c.total_gpus, c.batch_size) for c in rank_candidates(shuffled, "balanced", top_k=4)]
        assert got == expected, order
