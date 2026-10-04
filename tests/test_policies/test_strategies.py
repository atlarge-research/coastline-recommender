"""Unit tests for the recommendation policies in ``coastline.sdk.policies``:

    - ``base.BaseStrategy``: the strategy interface
    - ``min_gpu.MinGPUStrategy``: the first feasible GPU count for the job's total batch
    - ``multi_objective.MultiObjectiveStrategy``: alpha/beta weighted ranking and presets
    - ``PolicyFactory``: strategy by name and preset

A fake predictor, used for both throughput and power, returns values that depend only on
(total_gpus, batch_size), and a NoOpFeasibilityChecker admits every candidate, so the rankings
can be checked exactly without ML models or Kavier.
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


class FakePredictor:
    """A scripted predictor used for both throughput and power.

    ``predict`` looks up ``(total_gpus, batch_size)`` in ``table``. A candidate missing from the
    table gets None, which the workflow skips.
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


@pytest.fixture
def context() -> SystemContext:
    """A context large enough that the grid is never the limit."""
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
    """Kavier throughput and power with rules feasibility, so PolicyFactory loads no ML model or AutoConf."""
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
    """A pipeline with the fake predictor and NoOp feasibility."""
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


# Three candidates at batch 4, as total_gpus: (throughput, per-GPU power):
#   1: (100, 200)
#   2: (500, 100), the lowest per-GPU power
#   4: (900, 380), the highest throughput
# min_gpu picks 1 GPU, the energy and performance presets pick 2, and alpha=1, beta=0 picks 4.
THREE_WAY_TABLE = {
    (1, 4): (100.0, 200.0),
    (2, 4): (500.0, 100.0),
    (4, 4): (900.0, 380.0),
}
THREE_WAY_GRID = {"batch_sizes": [4], "total_gpus": [1, 2, 4], "top_k": 3}

# min_gpu splits the job's total batch (4, on 1 GPU) over 1, 2 and 4 GPUs: 4, 2 and 1 per device.
MIN_GPU_TABLE = {
    (1, 4): (100.0, 200.0),
    (2, 2): (400.0, 100.0),
    (4, 1): (700.0, 380.0),
}


# base.BaseStrategy
class TestBaseStrategy:
    def test_recommendation_strategy_field_equals_strategy_name(self, workload, context):
        """Both strategies return all 3 configs, each with Recommendation.strategy equal to the
        strategy's get_name()."""
        pred = FakePredictor({**THREE_WAY_TABLE, **MIN_GPU_TABLE})
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
            # all 3 configs are feasible
            assert isinstance(recs, list) and len(recs) == 3
            assert all(r.strategy == strat.get_name() for r in recs)


# PolicyFactory.create_strategy: by name and preset, and errors
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

    # Preset weights as (alpha = runtime, beta = energy), written out here instead of read from
    # PRESET_WEIGHTS and PRESET_TO_POLICY.
    @pytest.mark.parametrize(
        "preset, exp_alpha, exp_beta, exp_policy",
        [
            ("balanced", 0.5, 0.5, "balanced"),
            ("performance", 0.8, 0.2, "performance"),
            ("energy", 0.2, 0.8, "energy"),
        ],
    )
    def test_multi_objective_presets_select_expected_weights(self, preset, exp_alpha, exp_beta, exp_policy):
        """Each preset gives its (alpha, beta), selection policy and name."""
        strat = PolicyFactory.create_strategy(
            strategy_name="multi_objective",
            preset=preset,
            config=_config(THREE_WAY_GRID, _kavier_predictor_config()),
        )
        assert isinstance(strat, MultiObjectiveStrategy)
        assert strat.get_name() == f"multi_objective_{preset}"
        assert strat.alpha == pytest.approx(exp_alpha)
        assert strat.beta == pytest.approx(exp_beta)
        # the weights sum to 1
        assert strat.alpha + strat.beta == pytest.approx(1.0)
        # The pipeline uses the preset's selection policy.
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


class _RejectOneGpu:
    """Feasibility checker that rejects the 1-GPU candidate and admits the rest."""

    def is_feasible(self, workload: WorkloadSpec) -> tuple[bool, dict]:
        return workload.total_gpus != 1, {}


# MinGPUStrategy: the first feasible GPU count in 1, 2, 4, ...
class TestMinGPUStrategy:
    def test_selects_fewest_gpus(self, workload, context):
        """With every count feasible, MinGPUStrategy returns 1 GPU first, then 2 and 4."""
        pred = FakePredictor(MIN_GPU_TABLE)
        strat = MinGPUStrategy(
            pipeline=_pipeline(
                grid=THREE_WAY_GRID,
                selection_policy="min_gpu",
                strategy_name="min_gpu",
                predictor=pred,
            )
        )
        recs = strat.recommend(workload, context)
        # top_k is 3; the next feasible counts follow in doubling order.
        assert recs[0].total_gpus == 1
        assert recs[0].strategy == "min_gpu"
        assert [(r.total_gpus, r.metadata["batch_size"]) for r in recs] == [(1, 4), (2, 2), (4, 1)]

    def test_skips_infeasible_smallest_and_picks_next(self, workload, context):
        """If the 1-GPU config is infeasible, the 2-GPU config is chosen."""
        pred = FakePredictor(MIN_GPU_TABLE)
        pipeline = GridWorkflowPipeline.from_config(
            config=_config(THREE_WAY_GRID),
            selection_policy="min_gpu",
            strategy_name="min_gpu",
            throughput_predictor=pred,
            power_predictor=pred,
            feasibility_checker=_RejectOneGpu(),
        )
        recs = MinGPUStrategy(pipeline=pipeline).recommend(workload, context)
        assert recs[0].total_gpus == 2
        # A total batch of 4 has no third count: 8 GPUs would get half a sample each.
        assert [r.total_gpus for r in recs] == [2, 4]

    def test_keeps_the_jobs_batch_size(self, workload, context):
        """min_gpu does not explore batch sizes: a faster batch in the grid is not picked."""
        table = {
            (1, 4): (300.0, 100.0),  # the job's batch, picked
            (1, 8): (700.0, 100.0),  # faster, at a batch the job does not use
        }
        grid = {"batch_sizes": [4, 8], "total_gpus": [1], "top_k": 1}
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
        assert recs[0].total_gpus == 1
        assert recs[0].metadata["batch_size"] == 4
        assert recs[0].predicted_throughput == pytest.approx(300.0)

    def test_no_feasible_candidates_raises(self, workload, context):
        """With an empty prediction table the pick gets no prediction, and recommend() raises a
        RuntimeError (no usable prediction)."""
        pred = FakePredictor({})  # predicts nothing
        strat = MinGPUStrategy(
            pipeline=_pipeline(
                grid=THREE_WAY_GRID,
                selection_policy="min_gpu",
                strategy_name="min_gpu",
                predictor=pred,
            )
        )
        with pytest.raises(RuntimeError, match="no usable prediction"):
            strat.recommend(workload, context)


# MultiObjectiveStrategy: alpha/beta weighted scoring and presets
class TestMultiObjectiveStrategy:
    def _multi_objective(self, predictor, *, grid, **kw) -> MultiObjectiveStrategy:
        return MultiObjectiveStrategy(
            throughput_predictor=predictor,
            power_predictor=predictor,
            config=_config(grid),
            **kw,
        )

    def test_performance_preset_is_weighted_sum_not_pure_throughput(self, workload, context):
        # performance is a weighted sum with alpha=0.8 on time and beta=0.2 on total power. The
        # 4-GPU config is the fastest but draws 380 x 4 = 1520 W (power_score 0.0), so 2 GPUs win:
        #   2 GPUs: 0.8 * 0.9 + 0.2 * 1.0 = 0.92
        #   4 GPUs: 0.8 * 1.0 + 0.2 * 0.0 = 0.80
        #   1 GPU:  0.8 * 0.0 + 0.2 * 1.0 = 0.20
        pred = FakePredictor(THREE_WAY_TABLE)
        strat = self._multi_objective(pred, grid=THREE_WAY_GRID, preset="performance")
        recs = strat.recommend(workload, context)
        assert recs[0].total_gpus == 2  # the weighted-sum winner
        assert recs[0].predicted_throughput == pytest.approx(500.0)
        assert recs[0].strategy == "multi_objective_performance"
        assert recs[0].metadata["combined_score"] == pytest.approx(0.92)

        # With alpha=1 and beta=0 power is ignored and the fastest config (4 GPUs) wins.
        pure = self._multi_objective(FakePredictor(THREE_WAY_TABLE), grid=THREE_WAY_GRID, alpha=1.0, beta=0.0)
        recs_pure = pure.recommend(workload, context)
        assert recs_pure[0].total_gpus == 4
        assert recs_pure[0].predicted_throughput == pytest.approx(900.0)
        assert recs_pure[0].metadata["combined_score"] == pytest.approx(1.0)

    def test_energy_preset_picks_lowest_power(self, workload, context):
        # energy: alpha=0.2 on time, beta=0.8 on power. For 1, 2 and 4 GPUs:
        #   power_cost = watts * gpus: 200, 200, 1520, so power_score = 1.0, 1.0, 0.0
        #   time = 1 / throughput, so thr_score = 0.0, 0.9, 1.0
        #   combined = 0.2 * thr_score + 0.8 * power_score = 0.80, 0.98, 0.20
        pred = FakePredictor(THREE_WAY_TABLE)
        strat = self._multi_objective(pred, grid=THREE_WAY_GRID, preset="energy")
        recs = strat.recommend(workload, context)
        assert recs[0].total_gpus == 2
        assert recs[0].metadata["predicted_power_watts"] == pytest.approx(100.0)
        assert recs[0].metadata["combined_score"] == pytest.approx(0.98)

    def test_custom_alpha_beta_override_preset_and_normalise(self, workload, context):
        """Custom alpha/beta override the preset and are rescaled to sum to 1."""
        pred = FakePredictor(THREE_WAY_TABLE)
        strat = self._multi_objective(pred, grid=THREE_WAY_GRID, preset="energy", alpha=0.2, beta=0.6)
        # 0.2:0.6 becomes 0.25/0.75, and the preset becomes "custom".
        assert strat.alpha == pytest.approx(0.25)
        assert strat.beta == pytest.approx(0.75)
        assert strat.preset == "custom"
        assert strat.get_name() == "multi_objective_custom"

    def test_alpha_beta_change_balanced_ranking(self, workload, context):
        """With both candidates on 2 GPUs, power-heavy weights pick the low-power one and
        throughput-heavy weights the faster one."""
        # P (batch 4): fast, high power. Q (batch 8): slow, low power.
        #   power_cost P = 380 x 2 = 760, Q = 40 x 2 = 80, so power_score P = 0.0, Q = 1.0
        #   time P = 1/900, Q = 1/200, so thr_score P = 1.0, Q = 0.0
        table = {
            (2, 4): (900.0, 380.0),  # P
            (2, 8): (200.0, 40.0),  # Q
        }
        grid = {"batch_sizes": [4, 8], "total_gpus": [2], "top_k": 3}

        # beta=0.9 on power: Q scores 0.9 and P 0.1.
        energy_heavy = self._multi_objective(FakePredictor(table), grid=grid, alpha=0.1, beta=0.9)
        recs_e = energy_heavy.recommend(workload, context)
        assert recs_e[0].metadata["batch_size"] == 8  # Q
        assert recs_e[0].metadata["combined_score"] == pytest.approx(0.9)

        # alpha=0.9 on throughput: P scores 0.9 and Q 0.1.
        perf_heavy = self._multi_objective(FakePredictor(table), grid=grid, alpha=0.9, beta=0.1)
        recs_p = perf_heavy.recommend(workload, context)
        assert recs_p[0].metadata["batch_size"] == 4  # P
        assert recs_p[0].metadata["combined_score"] == pytest.approx(0.9)

        # Both use the "custom" preset with the balanced selection policy.
        assert energy_heavy.preset == "custom"
        assert energy_heavy._pipeline.selection_policy == "balanced"

    def test_balanced_scores_match_the_worked_out_min_max_normalisation(self, workload, context):
        """The balanced scores for THREE_WAY match the min-max values below.

        For 1, 2 and 4 GPUs: power_cost = 200, 200, 1520, so power_score = (1520 - cost) / 1320 =
        1.0, 1.0, 0.0; thr_score = (1/100 - 1/thr) / (1/100 - 1/900) = 0.0, 0.9, 1.0; and
        combined = 0.5 * thr_score + 0.5 * power_score = 0.50, 0.95, 0.50.
        """
        pred = FakePredictor(THREE_WAY_TABLE)
        strat = self._multi_objective(pred, grid=THREE_WAY_GRID, preset="balanced")
        recs = strat.recommend(workload, context)
        by_gpus = {r.total_gpus: r.metadata for r in recs}

        # power_score
        assert by_gpus[1]["power_score"] == pytest.approx(1.0)
        assert by_gpus[2]["power_score"] == pytest.approx(1.0)
        assert by_gpus[4]["power_score"] == pytest.approx(0.0)
        # throughput_score
        assert by_gpus[1]["throughput_score"] == pytest.approx(0.0)
        assert by_gpus[2]["throughput_score"] == pytest.approx(0.9)
        assert by_gpus[4]["throughput_score"] == pytest.approx(1.0)
        # combined_score = 0.5 * throughput + 0.5 * power
        assert by_gpus[1]["combined_score"] == pytest.approx(0.50)
        assert by_gpus[2]["combined_score"] == pytest.approx(0.95)
        assert by_gpus[4]["combined_score"] == pytest.approx(0.50)

        # 2 GPUs score highest, and the list is in descending combined_score.
        assert recs[0].total_gpus == 2
        scores = [r.metadata["combined_score"] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_returns_at_most_top_k(self, workload, context):
        """top_k=2 returns the two best configs in order: 2 GPUs (0.95), then 1 or 4 GPUs (0.5)."""
        pred = FakePredictor(THREE_WAY_TABLE)
        grid = {"batch_sizes": [4], "total_gpus": [1, 2, 4], "top_k": 2}
        strat = self._multi_objective(pred, grid=grid, preset="balanced")
        recs = strat.recommend(workload, context)
        assert len(recs) == 2
        # rank metadata starts at 1.
        assert [r.metadata["rank"] for r in recs] == [1, 2]
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
        with pytest.raises(RuntimeError, match="no usable prediction"):
            strat.recommend(workload, context)
