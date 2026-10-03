"""Custom alpha/beta weights in ``MultiObjectiveStrategy``.

  - The weights are divided by their sum, so the alpha:beta ratio holds at any magnitude:
    1.0:3.0 becomes 0.25/0.75.
  - Weights already in [0, 1] and the built-in presets go through the same division.
  - A negative weight is set to 0; if both end up at 0, the split is 0.5/0.5.

A fake predictor and a NoOp feasibility checker keep ML models and Kavier out. The tests check the
stored alpha and beta and, where the ratio matters, the ranking.
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


# Fake predictor, the same as in test_strategies.py
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


# Two candidates on the same total_gpus (2), so only the weighted score separates them:
# P (batch 4) has high throughput and high power, Q (batch 8) low throughput and low power.
RATIO_TABLE = {
    (2, 4): (900.0, 380.0),  # P
    (2, 8): (200.0, 40.0),  # Q
}
RATIO_GRID = {"batch_sizes": [4, 8], "total_gpus": [2], "top_k": 3}

# Three candidates for the tests that check only the stored weights.
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


# The ratio holds for weights outside [0, 1]
class TestRatioPreserved:
    # The same 1:3 ratio at three magnitudes. beta = 3 * alpha and alpha + beta = 1 give
    # alpha = 0.25 and beta = 0.75 in all three cases.
    @pytest.mark.parametrize("alpha_in,beta_in", [(1.0, 3.0), (2.0, 6.0), (10.0, 30.0)])
    def test_out_of_range_ratio_preserved_and_normalised(self, alpha_in, beta_in):
        strat = _multi_objective(THREE_WAY_TABLE, grid=THREE_WAY_GRID, alpha=alpha_in, beta=beta_in)
        # the ratio holds and the weights sum to 1
        assert strat.beta == pytest.approx(3.0 * strat.alpha)
        assert strat.alpha + strat.beta == pytest.approx(1.0)
        assert strat.alpha == pytest.approx(0.25)
        assert strat.beta == pytest.approx(0.75)
        # Clamping beta to 1 before the division would give 0.5/0.5.
        assert strat.alpha != pytest.approx(0.5)
        assert strat.preset == "custom"

    def test_out_of_range_ratio_reranks(self, workload, context):
        """Weights above 1 still steer the ranking: 1:3 picks P (batch 4) and 3:1 picks Q (batch 8).

        Over RATIO_TABLE, P has power_score 0 and throughput_score 1, and Q the reverse.
        """
        perf_heavy = _multi_objective(RATIO_TABLE, grid=RATIO_GRID, alpha=1.0, beta=3.0)
        recs_p = perf_heavy.recommend(workload, context)
        assert recs_p[0].metadata["batch_size"] == 4  # P, high throughput

        energy_heavy = _multi_objective(RATIO_TABLE, grid=RATIO_GRID, alpha=3.0, beta=1.0)
        recs_e = energy_heavy.recommend(workload, context)
        assert recs_e[0].metadata["batch_size"] == 8  # Q, low power


# Weights in [0, 1] and the presets
class TestInRangeUnchanged:
    def test_in_range_pair_normalises_by_sum(self):
        """0.2/0.6 is divided by its sum, giving 0.25/0.75."""
        strat = _multi_objective(THREE_WAY_TABLE, grid=THREE_WAY_GRID, alpha=0.2, beta=0.6)
        assert strat.beta == pytest.approx(3.0 * strat.alpha)  # ratio 1:3 kept
        assert strat.alpha + strat.beta == pytest.approx(1.0)
        assert strat.alpha == pytest.approx(0.25)
        assert strat.beta == pytest.approx(0.75)

    def test_in_range_pair_summing_to_one_is_untouched(self):
        """A pair that already sums to 1 (0.9/0.1) is kept as given."""
        strat = _multi_objective(THREE_WAY_TABLE, grid=THREE_WAY_GRID, alpha=0.9, beta=0.1)
        assert strat.alpha == pytest.approx(0.9)
        assert strat.beta == pytest.approx(0.1)
        assert strat.alpha + strat.beta == pytest.approx(1.0)

    # Preset weights as (alpha = power, beta = throughput), written out here instead of read from
    # PRESET_WEIGHTS. Each pair sums to 1.
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


# Negative weights
class TestDegenerateWeights:
    def test_negative_weight_is_floored_then_normalised(self):
        """A negative weight becomes 0, so -1.0/3.0 gives 0.0/1.0."""
        strat = _multi_objective(THREE_WAY_TABLE, grid=THREE_WAY_GRID, alpha=-1.0, beta=3.0)
        assert strat.alpha >= 0.0 and strat.beta >= 0.0  # no negative weight is stored
        assert strat.alpha == pytest.approx(0.0)
        assert strat.beta == pytest.approx(1.0)

    def test_both_negative_falls_back_to_balanced(self):
        """Two negative weights both become 0, and the strategy falls back to 0.5/0.5."""
        strat = _multi_objective(THREE_WAY_TABLE, grid=THREE_WAY_GRID, alpha=-2.0, beta=-5.0)
        assert strat.alpha == pytest.approx(0.5)
        assert strat.beta == pytest.approx(0.5)
        assert strat.alpha + strat.beta == pytest.approx(1.0)
