"""Oracle-based tests for the multi-objective energy power-scoring path — the LIVE
GridWorkflowPipeline + normalize_candidates + rank_candidates.

Scoring model under test (all oracles below are hand-derived from these definitions,
written in a DIFFERENT form than the implementation):
    power_cost(c)  = c.power (per-GPU watts) × c.total_gpus          # TOTAL cluster watts
    time_cost(c)   = 1 / c.throughput                                # runtime proxy
    power_score    = (p_max − power_cost) / (p_max − p_min)          # min-max, higher=better
    throughput_score = (t_max − time_cost) / (t_max − t_min)         # min-max over 1/throughput
    combined_score = α·power_score + β·throughput_score              # α=power weight, β=throughput
Preset weights (α_power, β_throughput): energy (0.8, 0.2), balanced (0.5, 0.5),
performance (0.2, 0.8).

Guards a real, fixed bug: predictors report power PER-GPU (~constant across GPU count),
but total cluster power is per-GPU-watts × total_gpus. The pre-fix per-GPU fixed-cap
formula divided per-GPU power by a fixed TDP budget, so power_score spuriously climbed
toward 1.0 as total_gpus grew, making large clusters look almost free on energy.
EvaluatedCandidate.power (surfaced as predicted_power_watts) stays per-GPU for display.
Inputs are synthetic; no artifacts.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import pytest

from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.recommendation import Prediction
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.pipeline.feasibility import NoOpFeasibilityChecker
from coastline.sdk.pipeline.workflow import GridWorkflowPipeline


class _ScriptedPredictor:
    """Scripted (throughput, power) lookup keyed by (total_gpus, batch_size); used as
    both throughput and power predictor so per-candidate numbers stay coupled. ``power``
    is PER-GPU watts, exactly as the real power predictors report it."""

    def __init__(self, table: Dict[Tuple[int, int], Tuple[float, float]]) -> None:
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

    def get_name(self) -> str:
        return "scripted"


def _workload(batch_size: int = 4) -> WorkloadSpec:
    return WorkloadSpec(
        llm_model="mistral-7b-v0.1",
        fine_tuning_method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=batch_size,
    )


def _context() -> SystemContext:
    return SystemContext(
        available_gpu_models=["NVIDIA-A100-SXM4-80GB"],
        max_gpus=64,
        gpu_memory={"NVIDIA-A100-SXM4-80GB": 80},
        constraints=Constraints(max_gpus=64, gpus_per_node=8, max_nodes=8),
    )


def _pipeline(table, *, total_gpus, policy="balanced", alpha=0.5, beta=0.5):
    predictor = _ScriptedPredictor(table)
    return GridWorkflowPipeline.from_config(
        config={"grid": {"batch_sizes": [4], "total_gpus": total_gpus, "top_k": 3}},
        selection_policy=policy,
        strategy_name="test",
        throughput_predictor=predictor,
        power_predictor=predictor,
        feasibility_checker=NoOpFeasibilityChecker(),
        alpha=alpha,
        beta=beta,
    )


def test_energy_weighting_prefers_lower_total_power_regardless_of_gpu_count():
    """Regression: power_cost is TOTAL watts (per-GPU × count), not per-GPU. Equal
    throughput; 1 GPU @ 390 W/GPU = 390 W total vs 2 GPUs @ 100 W/GPU = 200 W total.
    Under energy weighting (α=0.9 power, β=0.1 thr) the lower-TOTAL-power 2-GPU config
    must win even though it uses MORE GPUs.

    By hand: power_cost 2-GPU=200, 1-GPU=390 → p_min=200, p_max=390.
      power_score(2GPU) = (390-200)/(390-200) = 190/190 = 1.0
      power_score(1GPU) = (390-390)/190       = 0.0
    Equal throughput → t_max==t_min → throughput_score = 1.0 for both (degenerate axis).
      combined(2GPU) = 0.9·1.0 + 0.1·1.0 = 1.0  →  combined(1GPU) = 0.9·0.0 + 0.1·1.0 = 0.1
    """
    table = {(1, 4): (500.0, 390.0), (2, 4): (500.0, 100.0)}
    recs = _pipeline(table, total_gpus=[1, 2], alpha=0.9, beta=0.1).recommend(_workload(), _context())
    by_gpus = {r.total_gpus: r.metadata for r in recs}
    assert recs[0].total_gpus == 2
    assert by_gpus[2]["power_score"] == pytest.approx(1.0)  # 200 W total -> lowest -> best
    assert by_gpus[1]["power_score"] == pytest.approx(0.0)  # 390 W total -> highest -> worst
    assert by_gpus[2]["combined_score"] == pytest.approx(1.0)
    assert by_gpus[1]["combined_score"] == pytest.approx(0.1)
    # Cross-check against the old per-GPU fixed-cap bug: 1 - 100/400 = 0.75 (A100 TDP=400 W).
    assert by_gpus[2]["power_score"] != pytest.approx(1.0 - 100.0 / 400.0)


def test_performance_weighting_prefers_higher_throughput_despite_higher_total_power():
    """Complement of the energy test on the SAME trade-off: with performance weighting
    (α=0.2 power, β=0.8 thr) the higher-throughput config wins even though it draws more
    TOTAL power — proving β actually weights throughput.

    LOW : 1 GPU @ 100 W = 100 W total, 300 tok/s.  HIGH: 2 GPU @ 250 W = 500 W total, 600 tok/s.
    power_cost: LOW=100, HIGH=500 → power_score LOW=1.0, HIGH=0.0.
    time_cost:  LOW=1/300, HIGH=1/600 → t_min=1/600, t_max=1/300
      throughput_score(HIGH) = (1/300 - 1/600)/(1/300 - 1/600) = 1.0 ; (LOW) = 0.0
      combined(HIGH) = 0.2·0.0 + 0.8·1.0 = 0.8  >  combined(LOW) = 0.2·1.0 + 0.8·0.0 = 0.2
    (Under energy weights the winner would flip — see the energy test.)
    """
    table = {(1, 4): (300.0, 100.0), (2, 4): (600.0, 250.0)}
    recs = _pipeline(table, total_gpus=[1, 2], policy="performance", alpha=0.2, beta=0.8).recommend(
        _workload(), _context()
    )
    by_gpus = {r.total_gpus: r.metadata for r in recs}
    assert recs[0].total_gpus == 2
    assert by_gpus[2]["throughput_score"] == pytest.approx(1.0)
    assert by_gpus[1]["throughput_score"] == pytest.approx(0.0)
    assert by_gpus[2]["combined_score"] == pytest.approx(0.8)
    assert by_gpus[1]["combined_score"] == pytest.approx(0.2)


def test_power_score_is_linear_minmax_of_total_power_with_interior_point():
    """power_score is a LINEAR min-max ramp over TOTAL power; an interior point pins the
    slope (endpoints alone can't). Equal throughput isolates the power axis.

    total_gpus 1/2/4 @ per-GPU 100/125/100 W → total power 100/250/400 W.
    p_min=100, p_max=400.
      power_score(100) = (400-100)/300 = 1.0
      power_score(250) = (400-250)/300 = 150/300 = 0.5   <- interior
      power_score(400) = (400-400)/300 = 0.0
    """
    table = {(1, 4): (500.0, 100.0), (2, 4): (500.0, 125.0), (4, 4): (500.0, 100.0)}
    recs = _pipeline(table, total_gpus=[1, 2, 4], policy="energy", alpha=0.8, beta=0.2).recommend(
        _workload(), _context()
    )
    by_gpus = {r.total_gpus: r.metadata for r in recs}
    assert by_gpus[1]["power_score"] == pytest.approx(1.0)
    assert by_gpus[2]["power_score"] == pytest.approx(0.5)
    assert by_gpus[4]["power_score"] == pytest.approx(0.0)
    # A "1 - cost/max" fixed-cap ramp would give 1 - 250/400 = 0.375 for the middle config.
    assert by_gpus[2]["power_score"] != pytest.approx(0.375)


def test_throughput_score_is_minmax_of_inverse_throughput_not_throughput():
    """throughput_score min-maxes 1/throughput (runtime), NOT throughput directly, so it
    is linear in runtime and nonlinear in throughput. Equal TOTAL power isolates the axis.

    total_gpus 1/2/4 @ per-GPU 300/150/75 W → total power 300/300/300 W (power axis degenerate → 1.0).
    throughputs 300/400/600 tok/s → time_cost 1/300, 1/400, 1/600. t_min=1/600, t_max=1/300.
      throughput_score(400) = (1/300 - 1/400)/(1/300 - 1/600)
                            = (1/1200)/(1/600) = 600/1200 = 0.5   <- interior
      throughput_score(300) = 0.0 ; throughput_score(600) = 1.0
    A DIRECT throughput min-max would instead give (400-300)/(600-300) = 1/3 ≈ 0.333.
    """
    table = {(1, 4): (300.0, 300.0), (2, 4): (400.0, 150.0), (4, 4): (600.0, 75.0)}
    recs = _pipeline(table, total_gpus=[1, 2, 4], policy="performance", alpha=0.2, beta=0.8).recommend(
        _workload(), _context()
    )
    by_gpus = {r.total_gpus: r.metadata for r in recs}
    assert by_gpus[1]["throughput_score"] == pytest.approx(0.0)
    assert by_gpus[2]["throughput_score"] == pytest.approx(0.5)
    assert by_gpus[4]["throughput_score"] == pytest.approx(1.0)
    # Reject the linear-in-throughput bug (would be 1/3 for the 400 tok/s config).
    assert by_gpus[2]["throughput_score"] != pytest.approx(1.0 / 3.0)
    # Power axis is degenerate (all 300 W total) so every power_score collapses to 1.0.
    assert by_gpus[2]["power_score"] == pytest.approx(1.0)


def test_lone_feasible_candidate_gets_degenerate_scores_of_one():
    """With a single feasible candidate p_max==p_min and t_max==t_min, so both min-max
    denominators are zero. The guard must yield 1.0 (best) on each axis, not NaN/crash."""
    table = {(2, 4): (500.0, 137.0)}
    recs = _pipeline(table, total_gpus=[2], policy="energy", alpha=0.8, beta=0.2).recommend(_workload(), _context())
    assert len(recs) == 1
    assert recs[0].metadata["power_score"] == pytest.approx(1.0)
    assert recs[0].metadata["throughput_score"] == pytest.approx(1.0)
    # combined = 0.8·1.0 + 0.2·1.0 = 1.0
    assert recs[0].metadata["combined_score"] == pytest.approx(1.0)


def test_predicted_power_watts_stays_per_gpu_and_tokens_per_watt_uses_it():
    """The stored/displayed power stays PER-GPU (the fix lives in the SCORE, not the
    surfaced power). tokens_per_watt is derived from the per-GPU value.

    2 GPUs @ 137 W/GPU, 500 tok/s.
      predicted_power_watts = 137 (per-GPU), NOT 137×2 = 274 (would be the total-power bug).
      tokens_per_watt = 500 / 137 = 3.6496 tok/W, NOT 500/274 = 1.825 (total-power bug).
    """
    table = {(2, 4): (500.0, 137.0)}
    recs = _pipeline(table, total_gpus=[2], policy="energy").recommend(_workload(), _context())
    assert recs[0].metadata["predicted_power_watts"] == pytest.approx(137.0)
    assert recs[0].metadata["predicted_power_watts"] != pytest.approx(274.0)
    assert recs[0].metadata["tokens_per_watt"] == pytest.approx(3.649635, abs=1e-4)
    assert recs[0].metadata["tokens_per_watt"] != pytest.approx(500.0 / 274.0)
