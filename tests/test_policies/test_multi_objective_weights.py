"""Regression tests for ``MultiObjectiveStrategy`` alpha/beta weight handling.

Pins the contract of ``MultiObjectiveStrategy.__init__`` for *custom* alpha/beta
weights after the ratio-preservation fix:

  - The alpha:beta **ratio** is preserved regardless of magnitude. Previously
    each weight was clamped into [0, 1] *before* normalising, so a caller asking
    for 1:3 (alpha=1.0, beta=3.0) silently got 0.5/0.5. Now weights are
    normalised by their sum, so 1.0:3.0 -> 0.25/0.75.
  - Already-in-range weights (and the built-in presets) behave **identically** to
    before: any non-negative pair is just divided by its sum.
  - Negative weights are floored to 0 ("ignore this objective"); if that leaves
    both objectives at 0, the strategy falls back to an even 0.5/0.5 split.

These tests build the strategy with a fake predictor and a NoOp feasibility
checker so no ML artifacts / Kavier physics are loaded. They assert only the
stored ``self.alpha`` / ``self.beta`` (and, where ratio matters for ranking,
that the weighted score actually re-ranks candidates).

Run:
  cd <repo> && PYTHONPATH=coastline:coastline/common:kavier/src \
    DATA_DIR=./trace-archive .venv/bin/python -m pytest \
    coastline/tests/test_multi_objective_weights.py -q
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import pytest

from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.recommendation import Prediction
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.pipeline.feasibility import NoOpFeasibilityChecker
from coastline.sdk.pipeline.workflow import GridWorkflowPipeline
from coastline.sdk.policies.multi_objective import MultiObjectiveStrategy


# ---------------------------------------------------------------------------
# Fake predictor — deterministic, no ML / no physics. (Same shape as the one in
# test_strategies.py; duplicated to keep this regression file self-contained.)
# ---------------------------------------------------------------------------
class FakePredictor:
    """Scripted predictor used as both the throughput and the power predictor."""

    def __init__(self, table: Dict[Tuple[int, int], Tuple[float, float]]) -> None:
        # table: (total_gpus, batch_size) -> (throughput, power)
        self.table = table

    def predict(self, workload: WorkloadSpec, context: SystemContext) -> Optional[Prediction]:
        key = (workload.total_gpus, workload.batch_size)
        if key not in self.table:
            return None
        throughput, power = self.table[key]
        return Prediction(
            gpus_per_node=workload.gpus_per_node or 1,
            number_of_nodes=workload.number_of_nodes or 1,
            total_gpus=workload.total_gpus,
            predicted_throughput=throughput,
            predicted_runtime_seconds=123.0,
            predicted_power=power,
        )

    def get_name(self) -> str:  # pragma: no cover - trivial
        return "fake"


@pytest.fixture
def context() -> SystemContext:
    return SystemContext(
        available_gpu_models=["NVIDIA-A100-SXM4-80GB"],
        max_gpus=64,
        gpu_memory={"NVIDIA-A100-SXM4-80GB": 80},
        constraints=Constraints(max_gpus=64, gpus_per_node=8, max_nodes=8),
    )


@pytest.fixture
def workload() -> WorkloadSpec:
    return WorkloadSpec(
        llm_model="mistral-7b-v0.1",
        fine_tuning_method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=4,
    )


# Two candidates at the same total_gpus (=2) so only the weighted score can
# separate them: P (bs=4) high throughput/high power, Q (bs=8) low/low.
RATIO_TABLE = {
    (2, 4): (900.0, 380.0),  # P
    (2, 8): (200.0, 40.0),  # Q
}
RATIO_GRID = {"batch_sizes": [4, 8], "total_gpus": [2], "top_k": 3}

# Three candidates used where we only check stored weights, not ranking.
THREE_WAY_TABLE = {
    (1, 4): (100.0, 200.0),
    (2, 4): (500.0, 100.0),
    (4, 4): (900.0, 380.0),
}
THREE_WAY_GRID = {"batch_sizes": [4], "total_gpus": [1, 2, 4], "top_k": 3}


def _multi_objective(
    table: Dict[Tuple[int, int], Tuple[float, float]],
    *,
    grid: dict,
    **kw,
) -> MultiObjectiveStrategy:
    pred = FakePredictor(table)
    pipeline = GridWorkflowPipeline.from_config(
        config={"grid": grid},
        selection_policy="balanced",
        strategy_name="multi_objective_custom",
        throughput_predictor=pred,
        power_predictor=pred,
        feasibility_checker=NoOpFeasibilityChecker(),
        alpha=kw.get("alpha", 0.5),
        beta=kw.get("beta", 0.5),
        preset="custom",
    )
    return MultiObjectiveStrategy(
        throughput_predictor=pred,
        power_predictor=pred,
        pipeline=pipeline,
        **kw,
    )


# ===========================================================================
# Ratio preservation for out-of-[0,1] weights (the fix)
# ===========================================================================
class TestRatioPreserved:
    # Three magnitudes of the SAME 1:3 preference ratio. Oracle is TWO
    # independent constraints the fix guarantees, expressed differently from the
    # impl's per-weight divide-by-sum:
    #   (i)  ratio preserved:  beta == 3 * alpha   (input beta/alpha = 3)
    #   (ii) normalised:       alpha + beta == 1
    # Solving (i)+(ii): alpha = 1/(1+3) = 0.25, beta = 0.75 — independent of the
    # weight magnitude (2:6 and 10:30 must land on the same point). Magnitude-
    # invariance is the distinguishing behaviour, so it is parametrised, not the
    # bare value.
    @pytest.mark.parametrize("alpha_in,beta_in", [(1.0, 3.0), (2.0, 6.0), (10.0, 30.0)])
    def test_out_of_range_ratio_preserved_and_normalised(self, alpha_in, beta_in):
        strat = _multi_objective(THREE_WAY_TABLE, grid=THREE_WAY_GRID, alpha=alpha_in, beta=beta_in)
        # ratio invariant + sum invariant pin the pair to (0.25, 0.75)
        assert strat.beta == pytest.approx(3.0 * strat.alpha)
        assert strat.alpha + strat.beta == pytest.approx(1.0)
        assert strat.alpha == pytest.approx(0.25)
        assert strat.beta == pytest.approx(0.75)
        # cross-check vs the pinned pre-fix bug: clamp beta->1 first gave
        # 1/(1+1) = 0.5/0.5. The fix must NOT collapse the ratio.
        assert strat.alpha != pytest.approx(0.5)
        assert strat.preset == "custom"

    def test_out_of_range_ratio_actually_reranks(self, workload, context):
        """A >1 performance-heavy ratio must let the high-throughput candidate win.

        Hand-derived ranking oracle over RATIO_TABLE (both at total_gpus=2):
          P (bs=4): thr=900, power=380 -> power_cost=380*2=760, time_cost=1/900
          Q (bs=8): thr=200, power= 40 -> power_cost= 40*2= 80, time_cost=1/200
        min-max over {P,Q}: p_min=80, p_max=760 ; t_min=1/900, t_max=1/200
          P: power_score=(760-760)/(760-80)=0 ; throughput_score=1  (fastest)
          Q: power_score=(760-80)/(760-80)=1  ; throughput_score=0  (lowest power)
        Ranking is argmax of alpha*power_score + beta*throughput_score (scale by
        the sum cancels in the argmax, so raw or normalised give the same order).
          perf-heavy  (1:3): P=3*1=3 > Q=1*1=1  -> P (bs=4) wins
          energy-heavy(3:1): Q=3*1=3 > P=1*1=1  -> Q (bs=8) wins
        The old pre-clamp collapsed 1:3 to 0.5/0.5 (a tie), erasing the caller's
        preference; that would break at least one of these two assertions.
        """
        perf_heavy = _multi_objective(RATIO_TABLE, grid=RATIO_GRID, alpha=1.0, beta=3.0)
        recs_p = perf_heavy.recommend(workload, context)
        assert recs_p[0].metadata["batch_size"] == 4  # P, high throughput

        energy_heavy = _multi_objective(RATIO_TABLE, grid=RATIO_GRID, alpha=3.0, beta=1.0)
        recs_e = energy_heavy.recommend(workload, context)
        assert recs_e[0].metadata["batch_size"] == 8  # Q, low power


# ===========================================================================
# In-range behaviour and presets stay IDENTICAL to before the fix
# ===========================================================================
class TestInRangeUnchanged:
    def test_in_range_pair_normalises_by_sum(self):
        """In-range (un-clamped) pair 0.2/0.6 must still be divided by its sum.

        Distinct from the out-of-range test: neither the old nor the new code
        clamps these (both <= 1), so this pins the *unchanged* path. Oracle:
          0.2:0.6 is the 1:3 ratio; 0.2/(0.2+0.6)=0.25, 0.6/(0.2+0.6)=0.75.
        A regression that skipped normalising an in-range pair (leaving 0.2/0.6)
        fails here but NOT the out-of-range test, so both are needed.
        """
        strat = _multi_objective(THREE_WAY_TABLE, grid=THREE_WAY_GRID, alpha=0.2, beta=0.6)
        assert strat.beta == pytest.approx(3.0 * strat.alpha)  # ratio 1:3 kept
        assert strat.alpha + strat.beta == pytest.approx(1.0)
        assert strat.alpha == pytest.approx(0.25)
        assert strat.beta == pytest.approx(0.75)

    def test_in_range_pair_summing_to_one_is_untouched(self):
        """Normalising a pair that already sums to 1 is the identity (idempotence).

        Oracle: 0.9+0.1 == 1, so 0.9/(0.9+0.1)=0.9 and 0.1/1=0.1 — the input is
        returned unchanged. Rejects any bug that unconditionally rewrites the
        weights (e.g. always 0.5/0.5, or re-scales an already-normalised pair).
        """
        strat = _multi_objective(THREE_WAY_TABLE, grid=THREE_WAY_GRID, alpha=0.9, beta=0.1)
        assert strat.alpha == pytest.approx(0.9)
        assert strat.beta == pytest.approx(0.1)
        assert strat.alpha + strat.beta == pytest.approx(1.0)

    # Spec weights are (power=alpha, throughput=beta); pinned as LITERALS from the
    # documented preset table, independent of PRESET_WEIGHTS in the code under test.
    # energy favours low power (0.8), performance favours throughput (0.8),
    # balanced is even. Each already sums to 1, so no renormalisation applies.
    @pytest.mark.parametrize(
        "preset,exp_alpha,exp_beta",
        [("energy", 0.8, 0.2), ("balanced", 0.5, 0.5), ("performance", 0.2, 0.8)],
    )
    def test_presets_match_spec_weights(self, preset, exp_alpha, exp_beta):
        pred = FakePredictor(THREE_WAY_TABLE)
        strat = MultiObjectiveStrategy(
            throughput_predictor=pred,
            power_predictor=pred,
            preset=preset,
            config={"grid": THREE_WAY_GRID},
        )
        assert strat.alpha == pytest.approx(exp_alpha)
        assert strat.beta == pytest.approx(exp_beta)
        assert strat.alpha + strat.beta == pytest.approx(1.0)
        assert strat.preset == preset


# ===========================================================================
# Graceful handling of degenerate inputs
# ===========================================================================
class TestDegenerateWeights:
    def test_negative_weight_is_floored_then_normalised(self):
        """A negative weight is floored to 0; the other objective takes all weight.

        Oracle: max(0,-1.0)=0, max(0,3.0)=3 -> 0/(0+3)=0.0, 3/3=1.0. Contract
        invariants: no stored weight may be negative, and flooring one objective
        to 0 hands the whole budget (beta==1) to the survivor.
        """
        strat = _multi_objective(THREE_WAY_TABLE, grid=THREE_WAY_GRID, alpha=-1.0, beta=3.0)
        assert strat.alpha >= 0.0 and strat.beta >= 0.0  # negatives never leak through
        assert strat.alpha == pytest.approx(0.0)
        assert strat.beta == pytest.approx(1.0)

    def test_both_negative_falls_back_to_balanced(self):
        """Both weights floor to 0 -> zero sum is undefined, so fall back to 0.5/0.5.

        Oracle: max(0,-2)=max(0,-5)=0 -> sum 0; a 0/0 normalisation is undefined,
        and the documented fallback is an even split (every candidate would else
        score 0 and the winner be arbitrary).
        """
        strat = _multi_objective(THREE_WAY_TABLE, grid=THREE_WAY_GRID, alpha=-2.0, beta=-5.0)
        assert strat.alpha == pytest.approx(0.5)
        assert strat.beta == pytest.approx(0.5)
        assert strat.alpha + strat.beta == pytest.approx(1.0)
