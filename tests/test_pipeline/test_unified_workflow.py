"""Tests for the unified grid → feasibility → simulate → policy workflow.

Every assertion here is checked against an oracle that is independent of the code
under test: the grid is enumerated by hand, the min-max / weighted-sum scores are
derived from first principles in the comments, and the Kavier physics engine is
pinned only via invariants (positivity, [idle, TDP] bounds, sub-linear scaling) and
cross-checks — never a copied magic number.
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


def _cand(*, total_gpus, throughput=100.0, power=100.0, throughput_score=0.0, power_score=0.0):
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
    )


# --------------------------------------------------------------------------- grid


def test_grid_enumerates_batch_by_gpu_with_node_packing(workload, context):
    grid = grid_config_from_dict({"grid": {"batch_sizes": [4, 8, 16], "total_gpus": [1, 2, 4, 8, 16]}})
    candidates = generate_candidates(workload, context, grid)

    # Oracle: candidates = batch_sizes × total_gpus. All 5 GPU counts fit (max_gpus=16)
    # and pack within max_nodes=2 (16 GPUs → 2 nodes @ 8/node), so 3 × 5 = 15 survive,
    # each GPU count appearing exactly 3 times (once per batch size).
    assert len(candidates) == 15
    assert Counter(c.total_gpus for c in candidates) == {1: 3, 2: 3, 4: 3, 8: 3, 16: 3}

    # Node layout oracle: packs GPUs to the 8/node cap. 16 GPUs → (8 per node, 2 nodes);
    # 8 GPUs → (8, 1); anything ≤ node cap stays single-node.
    by_total = {c.total_gpus: (c.gpus_per_node, c.number_of_nodes) for c in candidates}
    assert by_total[16] == (8, 2)
    assert by_total[8] == (8, 1)
    assert by_total[4] == (4, 1)
    assert by_total[1] == (1, 1)


def test_grid_drops_total_gpus_exceeding_max_gpus(workload, context):
    # max_gpus=16 (from context); 32 > 16 must be dropped, 8 and 16 kept.
    grid = grid_config_from_dict({"grid": {"batch_sizes": [8], "total_gpus": [8, 16, 32]}})
    candidates = generate_candidates(workload, context, grid)

    assert {c.total_gpus for c in candidates} == {8, 16}  # 32 clipped by max_gpus
    assert len(candidates) == 2  # one batch size × two surviving GPU counts


def test_grid_drops_layout_exceeding_max_nodes(workload):
    # max_gpus=32 leaves 16 GPUs *within* the GPU budget, but with 8/node and max_nodes=1
    # the 16-GPU layout needs 2 nodes → dropped; only the single-node 8-GPU config survives.
    ctx = SystemContext(
        available_gpu_models=[GPU],
        max_gpus=32,
        gpu_memory={GPU: 80},
        constraints=Constraints(max_gpus=32, gpus_per_node=8, max_nodes=1),
    )
    grid = grid_config_from_dict({"grid": {"batch_sizes": [8], "total_gpus": [8, 16]}})
    candidates = generate_candidates(workload, ctx, grid)

    assert {c.total_gpus for c in candidates} == {8}  # 16 needs 2 nodes > max_nodes=1


# ------------------------------------------------------------------------ min_gpu


def test_min_gpu_ranks_by_gpu_count_then_throughput():
    rows = [
        _cand(total_gpus=4, throughput=100),
        _cand(total_gpus=2, throughput=60),
        _cand(total_gpus=2, throughput=90),
        _cand(total_gpus=8, throughput=200),
    ]
    ranked = rank_candidates(rows, "min_gpu", top_k=4)

    # Oracle: sort by (total_gpus asc, throughput desc). Fewest GPUs first; the two
    # 2-GPU rows tie-break on throughput (90 before 60), then 4-GPU, then 8-GPU.
    assert [(c.total_gpus, c.throughput) for c in ranked] == [(2, 90), (2, 60), (4, 100), (8, 200)]


# --------------------------------------------------------------------- normalize


def test_normalize_candidates_minmax_power_and_time_scores():
    # power_cost = power × total_gpus ; time_cost = 1/throughput.
    # c1: pc=100×1=100, tc=1/100=0.01   c2: pc=100×2=200, tc=1/300   c3: pc=100×4=400, tc=1/400=0.0025
    c1 = _cand(total_gpus=1, power=100, throughput=100)
    c2 = _cand(total_gpus=2, power=100, throughput=300)
    c3 = _cand(total_gpus=4, power=100, throughput=400)
    normalize_candidates([c1, c2, c3], "grid")

    # power_score = (p_max − pc)/(p_max − p_min), p_min=100 p_max=400, span=300:
    #   c1 (400−100)/300 = 1.0 ; c2 (400−200)/300 = 0.6667 ; c3 (400−400)/300 = 0.0
    assert c1.power_score == pytest.approx(1.0)
    assert c2.power_score == pytest.approx(2 / 3)
    assert c3.power_score == pytest.approx(0.0)

    # throughput_score = (t_max − tc)/(t_max − t_min), t_min=0.0025 t_max=0.01, span=0.0075:
    #   c1 (0.01−0.01)/0.0075 = 0.0 ; c2 (0.01−1/300)/0.0075 = 0.8889 ; c3 (0.01−0.0025)/0.0075 = 1.0
    assert c1.throughput_score == pytest.approx(0.0)
    assert c2.throughput_score == pytest.approx((0.01 - 1 / 300) / 0.0075)
    assert c3.throughput_score == pytest.approx(1.0)


def test_normalize_frontier_marks_pareto_dominated():
    # In (power_cost, time_cost) space, lower is better on both axes.
    # b: pc=100, tc=1/200=0.005 (fast, cheap) dominates
    #   a: pc=100, tc=1/100=0.010 (same power, slower)  → dominated
    #   c: pc=200, tc=1/50 =0.020 (pricier AND slower)  → dominated
    a = _cand(total_gpus=1, power=100, throughput=100)
    b = _cand(total_gpus=1, power=100, throughput=200)
    c = _cand(total_gpus=2, power=100, throughput=50)
    normalize_candidates([a, b, c], "frontier")

    assert (a.dominated, b.dominated, c.dominated) == (True, False, True)
    # Dominated candidates are zeroed so ranking skips them; the survivor keeps a real score.
    assert a.power_score == 0.0 and a.throughput_score == 0.0
    assert c.power_score == 0.0 and c.throughput_score == 0.0
    assert b.power_score == pytest.approx(1.0)  # sole survivor → best on the single-element frontier


# -------------------------------------------------------------------- weighted sum


def test_rank_energy_preset_weights_power_heavily():
    # Weighted sum: combined = α·power_score + β·throughput_score, energy weights α=0.8, β=0.2.
    a = _cand(total_gpus=1, throughput=100, power_score=1.0, throughput_score=0.0)
    b = _cand(total_gpus=2, throughput=300, power_score=0.6, throughput_score=0.9)
    c = _cand(total_gpus=4, throughput=400, power_score=0.0, throughput_score=1.0)
    ranked = rank_candidates([a, b, c], "energy", alpha=0.8, beta=0.2, top_k=3)

    # By hand: a 0.8·1.0+0.2·0.0=0.80 ; b 0.8·0.6+0.2·0.9=0.66 ; c 0.8·0.0+0.2·1.0=0.20.
    assert a.combined_score == pytest.approx(0.80)
    assert b.combined_score == pytest.approx(0.66)
    assert c.combined_score == pytest.approx(0.20)
    # Energy favours the lowest-power config even though it is the slowest.
    assert [r.total_gpus for r in ranked] == [1, 2, 4]


def test_rank_performance_preset_penalizes_extra_gpus():
    # Same candidates, performance weights α=0.2, β=0.8.
    a = _cand(total_gpus=1, throughput=100, power_score=1.0, throughput_score=0.0)
    b = _cand(total_gpus=2, throughput=300, power_score=0.6, throughput_score=0.9)
    c = _cand(total_gpus=4, throughput=400, power_score=0.0, throughput_score=1.0)
    ranked = rank_candidates([a, b, c], "performance", alpha=0.2, beta=0.8, top_k=3)

    # By hand: a 0.2·1.0+0.8·0.0=0.20 ; b 0.2·0.6+0.8·0.9=0.84 ; c 0.2·0.0+0.8·1.0=0.80.
    assert a.combined_score == pytest.approx(0.20)
    assert b.combined_score == pytest.approx(0.84)
    assert c.combined_score == pytest.approx(0.80)
    # Winner is b, NOT the highest-throughput c: c's larger power_cost costs it the top slot.
    assert [r.total_gpus for r in ranked] == [2, 4, 1]


def test_rank_breaks_near_ties_toward_higher_throughput():
    # Two configs whose scores fall within TIE_EPS=0.01 of each other. balanced α=β=0.5.
    #   x: 0.5·0.5 + 0.5·0.5 = 0.500 (throughput 100)
    #   y: 0.5·0.49 + 0.5·0.5 = 0.495 (throughput 500) — 0.005 lower, inside the 0.01 tie band
    # Raw score orders x above y; the tie-break must promote the higher-throughput y.
    x = _cand(total_gpus=2, throughput=100, power_score=0.5, throughput_score=0.5)
    y = _cand(total_gpus=2, throughput=500, power_score=0.49, throughput_score=0.5)
    ranked = rank_candidates([x, y], "balanced", alpha=0.5, beta=0.5, top_k=2)

    assert x.combined_score == pytest.approx(0.500)
    assert y.combined_score == pytest.approx(0.495)
    assert ranked[0].throughput == 500  # near-tie broken toward throughput, overriding the 0.005 score edge


def test_preset_weights_match_documented_spec():
    # Contract: presets weight (power, throughput). energy is power-dominant, performance
    # throughput-dominant, balanced even. Weights per project spec.
    assert PRESET_WEIGHTS["energy"] == (0.8, 0.2)
    assert PRESET_WEIGHTS["balanced"] == (0.5, 0.5)
    assert PRESET_WEIGHTS["performance"] == (0.2, 0.8)


# --------------------------------------------------------------------- feasibility


@pytest.mark.parametrize(
    "gpus_per_node, number_of_nodes, expected",
    [
        # batch_size=8; rule is batch_size % total_gpus == 0. Divisors of 8 → feasible.
        (1, 1, True),  # total 1  | 8 % 1 == 0
        (2, 1, True),  # total 2  | 8 % 2 == 0
        (4, 1, True),  # total 4  | 8 % 4 == 0
        (8, 1, True),  # total 8  | 8 % 8 == 0
        (3, 1, False),  # total 3  | 8 % 3 == 2
        (8, 2, False),  # total 16 | 8 % 16 == 8
    ],
)
def test_rules_feasibility_divisibility(gpus_per_node, number_of_nodes, expected):
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


def test_rules_feasibility_drops_indivisible_config_in_pipeline(workload, context):
    # Grid [1, 3] with batch_size=8: 8 % 1 == 0 (kept) but 8 % 3 == 2 (dropped by rules).
    # A pipeline that ignored feasibility would emit two recs including the 3-GPU config.
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

    assert [r.total_gpus for r in recs] == [1]  # only the divisible 1-GPU config survives


# -------------------------------------------------------------- Kavier integration


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

    # mistral-7b-v0.1/lora/A100 is a supported Kavier config, so both 1- and 2-GPU
    # candidates are feasible; min_gpu must pick the fewest → exactly one rec at 1 GPU.
    assert len(recs) == 1
    assert recs[0].total_gpus == 1
    assert recs[0].strategy == "min_gpu"
    assert recs[0].metadata["workflow"] == "grid_feasibility_simulate_policy"
    # Contract: a supported config yields a finite positive throughput (physical rate).
    assert recs[0].predicted_throughput is not None
    assert math.isfinite(recs[0].predicted_throughput) and recs[0].predicted_throughput > 0


def test_kavier_throughput_scales_sublinearly_with_gpus(context):
    # Weak-scaling law: adding GPUs raises total throughput but communication overhead
    # keeps it below the ideal N× (no super-linear speedup). Assert the bounds, not the
    # engine's exact tokens/sec.
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
    assert t1 < t2 < t4  # monotonically increasing with GPU count
    assert t2 < 2 * t1  # 2 GPUs deliver < 2× a single GPU (sub-linear)
    assert t4 < 4 * t1  # 4 GPUs deliver < 4× a single GPU (sub-linear)


def test_kavier_power_per_gpu_within_idle_and_tdp(workload, context):
    # Per-GPU power must lie in the device envelope [idle, TDP]; oracle read from the
    # hardware library (75 W idle, 400 W TDP for the A100-SXM4-80GB), independent of Kavier.
    idle = get_gpu_idle_power(GPU)
    tdp = get_gpu_tdp(GPU)
    pred = KavierPredictor().predict(workload, context)

    assert idle <= pred.predicted_power <= tdp


def test_kavier_power_predictor_agrees_with_throughput_engine(workload, context):
    # KavierPowerPredictor wraps the same engine (WRAPS_THROUGHPUT_ENGINE); the power it
    # reports must be the exact per-GPU watts the throughput predictor already computed.
    thr_power = KavierPredictor().predict(workload, context).predicted_power
    energy_power = KavierPowerPredictor().predict(workload, context).predicted_power

    assert energy_power == pytest.approx(thr_power)
    assert KavierPowerPredictor.WRAPS_THROUGHPUT_ENGINE is True


def test_kavier_unsupported_model_returns_error_prediction(context):
    # Contract for an out-of-library model: throughput predictor returns an error
    # Prediction (not a crash) with no throughput, and the power adapter returns None.
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

    assert KavierPowerPredictor().predict(bogus, context) is None
