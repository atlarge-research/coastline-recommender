"""
Unit tests for the recommendation policies in
``coastline/recommendation_policies/``:

    - ``base.BaseStrategy``            — abstract interface contract
    - ``min_gpu.MinGPUStrategy``       — picks the fewest-GPU acceptable config
    - ``multi_objective.MultiObjectiveStrategy`` — alpha/beta weighted ranking + presets
    - ``__init__.PolicyFactory``       — strategy dispatch by name / preset

These tests are deliberately fast and deterministic: they inject a *fake*
predictor (no ML artifacts, no Kavier physics, no host segfault from xgboost)
and a ``NoOpFeasibilityChecker`` so the grid → simulate → policy pipeline is
exercised on fully controlled numbers. The fake predictor serves as both the
throughput and the power predictor; it returns values that are a pure function
of the candidate's ``(total_gpus, batch_size)`` so the policy ranking can be
asserted exactly.

Run:
  cd <repo> && PYTHONPATH=coastline:coastline/common:kavier/src \
    DATA_DIR=./trace-archive .venv/bin/python -m pytest \
    coastline/tests/test_strategies.py -q
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import pytest

from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.recommendation import Prediction
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.pipeline.feasibility import NoOpFeasibilityChecker
from coastline.sdk.pipeline.workflow import GridWorkflowPipeline
from coastline.sdk.policies import PolicyFactory
from coastline.sdk.policies.base import BaseStrategy
from coastline.sdk.policies.min_gpu import MinGPUStrategy
from coastline.sdk.policies.multi_objective import MultiObjectiveStrategy


# ---------------------------------------------------------------------------
# Fake predictor — deterministic, no ML / no physics.
# ---------------------------------------------------------------------------
class FakePredictor:
    """A scripted predictor used as both the throughput and the power predictor.

    ``predict`` looks up ``(total_gpus, batch_size)`` in ``table`` and returns a
    ``Prediction`` carrying both ``predicted_throughput`` and ``predicted_power``
    (the workflow reads one from the throughput predictor and the other from the
    power predictor — using the same fake for both keeps the numbers coupled to
    the candidate). Any candidate missing from the table yields ``None`` (i.e.
    "predictor cannot predict"), which the workflow must skip.
    """

    def __init__(
        self,
        table: Dict[Tuple[int, int], Tuple[float, float]],
        *,
        runtime: float = 123.0,
    ) -> None:
        # table: (total_gpus, batch_size) -> (throughput, power)
        self.table = table
        self.runtime = runtime
        self.calls: list[Tuple[int, int]] = []

    def predict(self, workload: WorkloadSpec, context: SystemContext) -> Optional[Prediction]:
        key = (workload.total_gpus, workload.batch_size)
        self.calls.append(key)
        if key not in self.table:
            return None
        throughput, power = self.table[key]
        return Prediction(
            gpus_per_node=workload.gpus_per_node or 1,
            number_of_nodes=workload.number_of_nodes or 1,
            total_gpus=workload.total_gpus,
            predicted_throughput=throughput,
            predicted_runtime_seconds=self.runtime,
            predicted_power=power,
        )

    def get_name(self) -> str:  # pragma: no cover - trivial
        return "fake"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def context() -> SystemContext:
    """Roomy context so the grid is never the limiting factor."""
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


def _kavier_predictor_config() -> dict:
    """Predictor config that keeps PolicyFactory off the real ML models.

    Forces the analytical Kavier throughput predictor + Kavier power model and
    disables AutoConf (rules-only feasibility), so PolicyFactory.create_strategy
    builds without loading xgboost/catboost artifacts.
    """
    return {"performance": "kavier", "energy": "kavier_power", "feasibility": "rules"}


def _config(grid: dict, predictors: Optional[dict] = None) -> dict:
    cfg: dict = {"grid": grid}
    if predictors is not None:
        cfg["predictors"] = predictors
    return cfg


def _pipeline(
    *,
    grid: dict,
    selection_policy: str,
    strategy_name: str,
    predictor: FakePredictor,
    alpha: float = 0.5,
    beta: float = 0.5,
    preset: Optional[str] = None,
) -> GridWorkflowPipeline:
    """Build a pipeline wired to the fake predictor + NoOp feasibility."""
    return GridWorkflowPipeline.from_config(
        config=_config(grid),
        selection_policy=selection_policy,
        strategy_name=strategy_name,
        throughput_predictor=predictor,
        power_predictor=predictor,
        feasibility_checker=NoOpFeasibilityChecker(),
        alpha=alpha,
        beta=beta,
        preset=preset,
    )


# Three candidates that separate the policies cleanly (batch_size=4 fixed):
#   total_gpus=1: throughput=100, power=200
#   total_gpus=2: throughput=500, power=100   <- lowest power
#   total_gpus=4: throughput=900, power=380   <- highest throughput
# min_gpu  -> 1 GPU ; energy -> 2 GPUs (lowest power) ; performance -> 4 GPUs.
THREE_WAY_TABLE = {
    (1, 4): (100.0, 200.0),
    (2, 4): (500.0, 100.0),
    (4, 4): (900.0, 380.0),
}
THREE_WAY_GRID = {"batch_sizes": [4], "total_gpus": [1, 2, 4], "top_k": 3}


# ===========================================================================
# base.BaseStrategy — interface contract
# ===========================================================================
class TestBaseStrategy:
    def test_recommendation_strategy_field_equals_strategy_name(self, workload, context):
        """Contract invariant: every Recommendation.strategy equals the producing
        strategy's get_name(), and recommend() returns a non-empty list of
        Recommendation (the fake predictor makes all 3 configs feasible).

        Oracle: the name↔stamp identity is a cross-check between two independently
        exposed values (get_name() and the per-rec .strategy field), so a
        mislabelled recommendation (e.g. min_gpu output stamped "multi_objective")
        would fail here regardless of the concrete number.
        """
        pred = FakePredictor(THREE_WAY_TABLE)
        min_gpu = MinGPUStrategy(
            pipeline=_pipeline(
                grid=THREE_WAY_GRID,
                selection_policy="min_gpu",
                strategy_name="min_gpu",
                predictor=pred,
            )
        )
        multi_objective = MultiObjectiveStrategy(
            throughput_predictor=pred,
            power_predictor=pred,
            preset="balanced",
            config=_config(THREE_WAY_GRID),
        )
        assert min_gpu.get_name() == "min_gpu"
        assert multi_objective.get_name() == "multi_objective_balanced"
        for strat in (min_gpu, multi_objective):
            assert isinstance(strat, BaseStrategy)
            recs = strat.recommend(workload, context)
            # 3 feasible configs in, so a non-empty ranked list out.
            assert isinstance(recs, list) and len(recs) == 3
            # Identity invariant: the stamped strategy name matches get_name().
            assert all(r.strategy == strat.get_name() for r in recs)


# ===========================================================================
# PolicyFactory.create_strategy — dispatch by name / preset / errors
# ===========================================================================
class TestPolicyFactoryDispatch:
    def test_min_gpu_name_returns_min_gpu_strategy(self):
        strat = PolicyFactory.create_strategy(
            strategy_name="min_gpu",
            config=_config(THREE_WAY_GRID, _kavier_predictor_config()),
        )
        assert isinstance(strat, MinGPUStrategy)
        assert strat.get_name() == "min_gpu"

    def test_multi_objective_name_returns_multi_objective_strategy(self):
        strat = PolicyFactory.create_strategy(
            strategy_name="multi_objective",
            preset="balanced",
            config=_config(THREE_WAY_GRID, _kavier_predictor_config()),
        )
        assert isinstance(strat, MultiObjectiveStrategy)
        assert strat.get_name() == "multi_objective_balanced"

    # Spec (from the skill's derivability map): preset -> (alpha=power, beta=thr).
    # Hard-coded here as an INDEPENDENT oracle rather than reading the code's own
    # PRESET_WEIGHTS/PRESET_TO_POLICY dicts (which would be tautological).
    @pytest.mark.parametrize(
        "preset, exp_alpha, exp_beta, exp_policy",
        [
            ("balanced", 0.5, 0.5, "balanced"),
            ("performance", 0.2, 0.8, "performance"),
            ("energy", 0.8, 0.2, "energy"),
        ],
    )
    def test_multi_objective_presets_select_expected_weights(self, preset, exp_alpha, exp_beta, exp_policy):
        """Each preset maps to its spec (alpha, beta), selection policy, and name."""
        strat = PolicyFactory.create_strategy(
            strategy_name="multi_objective",
            preset=preset,
            config=_config(THREE_WAY_GRID, _kavier_predictor_config()),
        )
        assert isinstance(strat, MultiObjectiveStrategy)
        assert strat.get_name() == f"multi_objective_{preset}"
        assert strat.alpha == pytest.approx(exp_alpha)
        assert strat.beta == pytest.approx(exp_beta)
        # alpha+beta must sum to 1 (weights are a convex split of the two axes).
        assert strat.alpha + strat.beta == pytest.approx(1.0)
        # The underlying pipeline uses the preset's selection policy.
        assert strat._pipeline.selection_policy == exp_policy

    def test_unknown_strategy_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            PolicyFactory.create_strategy(
                strategy_name="does_not_exist",
                config=_config(THREE_WAY_GRID, _kavier_predictor_config()),
            )

    def test_strategy_name_defaults_to_config_value(self):
        """When strategy_name is omitted, the config's strategy.name is used."""
        cfg = _config(THREE_WAY_GRID, _kavier_predictor_config())
        cfg["strategy"] = {"name": "min_gpu"}
        strat = PolicyFactory.create_strategy(config=cfg)
        assert isinstance(strat, MinGPUStrategy)

    def test_unknown_energy_predictor_raises(self):
        """An unsupported energy predictor name is rejected by the factory."""
        cfg = _config(
            THREE_WAY_GRID,
            {"performance": "kavier", "energy": "bogus", "feasibility": "rules"},
        )
        with pytest.raises(ValueError, match="Unknown energy predictor"):
            PolicyFactory.create_strategy(strategy_name="min_gpu", config=cfg)


# ===========================================================================
# MinGPUStrategy — selects the fewest-GPU acceptable config
# ===========================================================================
class TestMinGPUStrategy:
    def test_selects_fewest_gpus(self, workload, context):
        """Among feasible candidates, MinGPU returns the one with the least total_gpus."""
        pred = FakePredictor(THREE_WAY_TABLE)
        strat = MinGPUStrategy(
            pipeline=_pipeline(
                grid=THREE_WAY_GRID,
                selection_policy="min_gpu",
                strategy_name="min_gpu",
                predictor=pred,
            )
        )
        recs = strat.recommend(workload, context)
        # min_gpu honours the configured top_k (3 here); the fewest-GPU config is
        # ranked first, the rest follow in increasing-GPU order.
        assert recs[0].total_gpus == 1
        assert recs[0].strategy == "min_gpu"
        assert [r.total_gpus for r in recs] == [1, 2, 4]

    def test_skips_infeasible_smallest_and_picks_next(self, workload, context):
        """If the 1-GPU config can't be predicted, the 2-GPU config is chosen."""
        # Drop the 1-GPU entry: predictor returns None -> workflow skips it.
        table = {k: v for k, v in THREE_WAY_TABLE.items() if k != (1, 4)}
        pred = FakePredictor(table)
        strat = MinGPUStrategy(
            pipeline=_pipeline(
                grid=THREE_WAY_GRID,
                selection_policy="min_gpu",
                strategy_name="min_gpu",
                predictor=pred,
            )
        )
        recs = strat.recommend(workload, context)
        # 1-GPU config is infeasible, so the 2-GPU config becomes the fewest-GPU pick.
        assert recs[0].total_gpus == 2
        assert [r.total_gpus for r in recs] == [2, 4]

    def test_tie_break_prefers_higher_throughput(self, workload, context):
        """Same total_gpus -> min_gpu tie-breaks on higher throughput (selection.py)."""
        # Two candidates both at total_gpus=2 (batch_size 4 vs 8); the higher
        # throughput one must win the tie.
        table = {
            (2, 4): (300.0, 100.0),  # lower throughput
            (2, 8): (700.0, 100.0),  # higher throughput -> should be picked
        }
        grid = {"batch_sizes": [4, 8], "total_gpus": [2], "top_k": 3}
        pred = FakePredictor(table)
        strat = MinGPUStrategy(
            pipeline=_pipeline(
                grid=grid,
                selection_policy="min_gpu",
                strategy_name="min_gpu",
                predictor=pred,
            )
        )
        recs = strat.recommend(workload, context)
        # Same total_gpus on both candidates → the higher-throughput one ranks first.
        assert recs[0].total_gpus == 2
        assert recs[0].metadata["batch_size"] == 8
        assert recs[0].predicted_throughput == pytest.approx(700.0)

    def test_no_feasible_candidates_raises(self, workload, context):
        """Empty prediction table -> every candidate skipped -> RuntimeError."""
        pred = FakePredictor({})  # predicts nothing
        strat = MinGPUStrategy(
            pipeline=_pipeline(
                grid=THREE_WAY_GRID,
                selection_policy="min_gpu",
                strategy_name="min_gpu",
                predictor=pred,
            )
        )
        with pytest.raises(RuntimeError, match="no feasible candidates"):
            strat.recommend(workload, context)


# ===========================================================================
# MultiObjectiveStrategy — alpha/beta weighted scoring + presets
# ===========================================================================
class TestMultiObjectiveStrategy:
    def _multi_objective(self, predictor, *, grid, **kw) -> MultiObjectiveStrategy:
        return MultiObjectiveStrategy(
            throughput_predictor=predictor,
            power_predictor=predictor,
            config=_config(grid),
            **kw,
        )

    def test_performance_preset_is_weighted_sum_not_pure_throughput(self, workload, context):
        # SEMANTIC CHANGE: "performance" is now a single weighted sum over the two
        # min-max'd axes (alpha=0.2 power-vs-total-power, beta=0.8 time) rather than a
        # raw-throughput sort. The 4-GPU config has the highest throughput (900) but
        # also by far the highest TOTAL power (380×4 = 1520W -> power_score 0.0), and
        # its runtime edge over the 2-GPU config is compressed by the min-max (2-GPU
        # time_score 0.9 vs 4-GPU 1.0). With a 0.2 power weight the 2-GPU config wins:
        #   2-GPU: 0.2×1.0 + 0.8×0.9 = 0.92   (power_cost 200W, lowest tier)
        #   4-GPU: 0.2×0.0 + 0.8×1.0 = 0.80
        #   1-GPU: 0.2×1.0 + 0.8×0.0 = 0.20
        # So the preset now declines a marginally-faster but much-higher-power config.
        pred = FakePredictor(THREE_WAY_TABLE)
        strat = self._multi_objective(pred, grid=THREE_WAY_GRID, preset="performance")
        recs = strat.recommend(workload, context)
        assert recs[0].total_gpus == 2  # weighted-sum winner, NOT the 900-throughput 4-GPU
        assert recs[0].predicted_throughput == pytest.approx(500.0)
        assert recs[0].strategy == "multi_objective_performance"
        assert recs[0].metadata["combined_score"] == pytest.approx(0.92)

        # PURE performance (alpha=0.0 power, beta=1.0 time) drops the power axis
        # entirely and recovers the max-throughput 4-GPU config.
        pure = self._multi_objective(FakePredictor(THREE_WAY_TABLE), grid=THREE_WAY_GRID, alpha=0.0, beta=1.0)
        recs_pure = pure.recommend(workload, context)
        assert recs_pure[0].total_gpus == 4
        assert recs_pure[0].predicted_throughput == pytest.approx(900.0)
        assert recs_pure[0].metadata["combined_score"] == pytest.approx(1.0)

    def test_energy_preset_picks_lowest_power(self, workload, context):
        # energy preset = (alpha=0.8 power, beta=0.2 time). Grid-normalised scores for
        # THREE_WAY (power_cost = watts×gpus; time = 1/thr):
        #   power_cost 1→200, 2→200, 4→1520 ; p_min=200 p_max=1520
        #     power_score = (1520-cost)/1320 : 1→1.0, 2→1.0, 4→0.0
        #   time 1→1/100, 2→1/500, 4→1/900 ; thr_score : 1→0.0, 2→0.9, 4→1.0
        # combined = 0.8*power_score + 0.2*thr_score :
        #   1-GPU: 0.8*1.0 + 0.2*0.0 = 0.80
        #   2-GPU: 0.8*1.0 + 0.2*0.9 = 0.98  <- winner
        #   4-GPU: 0.8*0.0 + 0.2*1.0 = 0.20
        pred = FakePredictor(THREE_WAY_TABLE)
        strat = self._multi_objective(pred, grid=THREE_WAY_GRID, preset="energy")
        recs = strat.recommend(workload, context)
        assert recs[0].total_gpus == 2
        assert recs[0].metadata["predicted_power_watts"] == pytest.approx(100.0)
        assert recs[0].metadata["combined_score"] == pytest.approx(0.98)

    def test_custom_alpha_beta_override_preset_and_normalise(self, workload, context):
        """Custom alpha/beta override the preset and are renormalised to sum to 1."""
        pred = FakePredictor(THREE_WAY_TABLE)
        strat = self._multi_objective(pred, grid=THREE_WAY_GRID, preset="energy", alpha=0.2, beta=0.6)
        # 0.2:0.6 -> 0.25 / 0.75 after normalisation; preset label becomes "custom".
        assert strat.alpha == pytest.approx(0.25)
        assert strat.beta == pytest.approx(0.75)
        assert strat.preset == "custom"
        assert strat.get_name() == "multi_objective_custom"

    def test_alpha_beta_change_balanced_ranking(self, workload, context):
        """The alpha/beta weights genuinely re-rank the *same* candidate set.

        Two candidates at the same total_gpus (=2) so min_gpu can't separate
        them; only the weighted score decides. Energy-heavy weights prefer the
        low-power candidate, performance-heavy weights prefer the high-throughput
        candidate -> the winner flips.
        """
        # P (bs=4): high throughput, high power ; Q (bs=8): low throughput, low power
        # Two configs, both total_gpus=2:
        #   power_cost P=380×2=760, Q=40×2=80  ⇒ power_score P=0.0, Q=1.0
        #   time      P=1/900,    Q=1/200      ⇒ thr_score  P=1.0, Q=0.0
        table = {
            (2, 4): (900.0, 380.0),  # P
            (2, 8): (200.0, 40.0),  # Q
        }
        grid = {"batch_sizes": [4, 8], "total_gpus": [2], "top_k": 3}

        # Energy-heavy (alpha=0.9 power) -> Q wins: 0.9*1.0 + 0.1*0.0 = 0.90 vs P 0.10.
        energy_heavy = self._multi_objective(FakePredictor(table), grid=grid, alpha=0.9, beta=0.1)
        recs_e = energy_heavy.recommend(workload, context)
        assert recs_e[0].metadata["batch_size"] == 8  # Q
        assert recs_e[0].metadata["combined_score"] == pytest.approx(0.9)

        # Performance-heavy (beta=0.9 thr) -> P wins: 0.1*0.0 + 0.9*1.0 = 0.90 vs Q 0.10.
        perf_heavy = self._multi_objective(FakePredictor(table), grid=grid, alpha=0.1, beta=0.9)
        recs_p = perf_heavy.recommend(workload, context)
        assert recs_p[0].metadata["batch_size"] == 4  # P
        assert recs_p[0].metadata["combined_score"] == pytest.approx(0.9)

        # Both are "custom" presets using the balanced selection policy.
        assert energy_heavy.preset == "custom"
        assert energy_heavy._pipeline.selection_policy == "balanced"

    def test_balanced_scores_hand_derived_from_min_max_normalisation(self, workload, context):
        """Every per-config (power_score, throughput_score, combined_score) matches a
        hand-computed min-max normalisation — NOT a re-call of normalize_candidates.

        Derivation for THREE_WAY at balanced (alpha=beta=0.5):
          power_cost = per-GPU watts × total_gpus :  1→200 · 2→200 · 4→1520
            p_min=200, p_max=1520  ⇒  power_score = (1520-cost)/1320
              1→(1520-200)/1320 = 1.0 · 2→1.0 · 4→(1520-1520)/1320 = 0.0
          time = 1/throughput :  1→1/100 · 2→1/500 · 4→1/900
            t_min=1/900, t_max=1/100  ⇒  thr_score = (1/100 - 1/thr)/(1/100 - 1/900)
              1→0.0 · 2→(4/500)/(8/900) = 3600/4000 = 0.9 · 4→1.0
          combined = 0.5*power_score + 0.5*thr_score
              1→0.50 · 2→0.95 (winner) · 4→0.50
        """
        pred = FakePredictor(THREE_WAY_TABLE)
        strat = self._multi_objective(pred, grid=THREE_WAY_GRID, preset="balanced")
        recs = strat.recommend(workload, context)
        by_gpus = {r.total_gpus: r.metadata for r in recs}

        # power_score (hand-derived, independent of the impl's formula form)
        assert by_gpus[1]["power_score"] == pytest.approx(1.0)
        assert by_gpus[2]["power_score"] == pytest.approx(1.0)
        assert by_gpus[4]["power_score"] == pytest.approx(0.0)
        # throughput_score
        assert by_gpus[1]["throughput_score"] == pytest.approx(0.0)
        assert by_gpus[2]["throughput_score"] == pytest.approx(0.9)
        assert by_gpus[4]["throughput_score"] == pytest.approx(1.0)
        # combined_score = 0.5*power + 0.5*throughput
        assert by_gpus[1]["combined_score"] == pytest.approx(0.50)
        assert by_gpus[2]["combined_score"] == pytest.approx(0.95)
        assert by_gpus[4]["combined_score"] == pytest.approx(0.50)

        # 2-GPU is the unique max, and recs are emitted in descending combined_score.
        assert recs[0].total_gpus == 2
        scores = [r.metadata["combined_score"] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_returns_at_most_top_k(self, workload, context):
        """multi-objective honours grid.top_k AND returns the correct top-k by score.

        top_k must not merely truncate the count: the returned k must be the
        *highest-scoring* k in descending order. For balanced THREE_WAY the
        unambiguous winner is the 2-GPU config (combined_score 0.95, vs the
        1-/4-GPU tie at 0.5), so top_k=2 must start with 2-GPU and the second
        slot must be one of the 0.5-tier configs (1 or 4 GPUs), never re-list 2.
        """
        pred = FakePredictor(THREE_WAY_TABLE)
        grid = {"batch_sizes": [4], "total_gpus": [1, 2, 4], "top_k": 2}
        strat = self._multi_objective(pred, grid=grid, preset="balanced")
        recs = strat.recommend(workload, context)
        assert len(recs) == 2
        # rank metadata is 1-based and sequential.
        assert [r.metadata["rank"] for r in recs] == [1, 2]
        # Identity/order of the top-2: the 2-GPU config is first; the runner-up is
        # a distinct, strictly-lower-scoring config from the 0.5 tier.
        assert recs[0].total_gpus == 2
        assert recs[0].metadata["combined_score"] == pytest.approx(0.95)
        assert recs[1].total_gpus in (1, 4)
        assert recs[1].total_gpus != recs[0].total_gpus
        assert recs[1].metadata["combined_score"] == pytest.approx(0.5)
        assert recs[0].metadata["combined_score"] > recs[1].metadata["combined_score"]

    def test_metadata_carries_preset_alpha_beta(self, workload, context):
        pred = FakePredictor(THREE_WAY_TABLE)
        strat = self._multi_objective(pred, grid=THREE_WAY_GRID, preset="balanced")
        rec = strat.recommend(workload, context)[0]
        assert rec.metadata["preset"] == "balanced"
        assert rec.metadata["alpha"] == pytest.approx(0.5)
        assert rec.metadata["beta"] == pytest.approx(0.5)
        assert rec.metadata["selection_policy"] == "balanced"

    def test_invalid_workload_raises_no_feasible(self, workload, context):
        pred = FakePredictor({})  # predicts nothing
        strat = self._multi_objective(pred, grid=THREE_WAY_GRID, preset="balanced")
        with pytest.raises(RuntimeError, match="no feasible candidates"):
            strat.recommend(workload, context)
