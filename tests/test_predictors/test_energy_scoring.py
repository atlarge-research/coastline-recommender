"""Tests for multi-objective power scoring in GridWorkflowPipeline (normalize_candidates and
rank_candidates), on synthetic inputs.

Scoring model:
    power_cost(c)    = c.power (watts per GPU) x c.total_gpus     # total cluster watts
    time_cost(c)     = 1 / c.throughput                           # runtime proxy
    power_score      = (p_max - power_cost) / (p_max - p_min)     # min-max, higher is better
    throughput_score = (t_max - time_cost) / (t_max - t_min)      # min-max over 1/throughput
    combined_score   = alpha * power_score + beta * throughput_score
alpha weights power and beta throughput. Preset weights (alpha, beta): energy (0.8, 0.2),
balanced (0.5, 0.5), performance (0.2, 0.8).

Predictors report power per GPU, which barely changes with GPU count, so the score uses total
cluster power. ``EvaluatedCandidate.power`` (shown as predicted_power_watts) stays per GPU.
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
    """Scripted (throughput, power) lookup keyed by (total_gpus, batch_size), used as both the
    throughput and the power predictor. ``power`` is watts per GPU, as the real power predictors
    report it."""

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
    """Power cost is total watts (watts per GPU x GPU count). With equal throughput, 2 GPUs at
    100 W (200 W total) beat 1 GPU at 390 W (390 W total) under alpha=0.9, beta=0.1.

        power_score: 2 GPUs (390 - 200) / 190 = 1.0, 1 GPU (390 - 390) / 190 = 0.0
        throughput_score: 1.0 for both (equal throughput)
        combined: 2 GPUs 0.9 x 1.0 + 0.1 x 1.0 = 1.0, 1 GPU 0.9 x 0.0 + 0.1 x 1.0 = 0.1
    """
    table = {(1, 4): (500.0, 390.0), (2, 4): (500.0, 100.0)}
    recs = _pipeline(table, total_gpus=[1, 2], alpha=0.9, beta=0.1).recommend(_workload(), _context())
    by_gpus = {r.total_gpus: r.metadata for r in recs}
    assert recs[0].total_gpus == 2
    assert by_gpus[2]["power_score"] == pytest.approx(1.0)  # 200 W total, the lowest
    assert by_gpus[1]["power_score"] == pytest.approx(0.0)  # 390 W total, the highest
    assert by_gpus[2]["combined_score"] == pytest.approx(1.0)
    assert by_gpus[1]["combined_score"] == pytest.approx(0.1)
    # A per-GPU fixed-cap score would give 1 - 100/400 = 0.75 (A100 TDP 400 W).
    assert by_gpus[2]["power_score"] != pytest.approx(1.0 - 100.0 / 400.0)


def test_performance_weighting_prefers_higher_throughput_despite_higher_total_power():
    """Under alpha=0.2, beta=0.8 the higher-throughput config wins although it draws more total
    power.

        LOW: 1 GPU at 100 W (100 W total), 300 tok/s. HIGH: 2 GPUs at 250 W (500 W total), 600 tok/s.
        power_score: LOW 1.0, HIGH 0.0. throughput_score: HIGH 1.0, LOW 0.0.
        combined: HIGH 0.2 x 0.0 + 0.8 x 1.0 = 0.8, LOW 0.2 x 1.0 + 0.8 x 0.0 = 0.2
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
    """power_score is linear min-max over total power; the middle point checks the slope. Equal
    throughput isolates the power axis.

        total_gpus 1/2/4 at 100/125/100 W per GPU: total power 100/250/400 W
        power_score: (400 - 100) / 300 = 1.0, (400 - 250) / 300 = 0.5, (400 - 400) / 300 = 0.0
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
    """throughput_score is min-max over 1/throughput (runtime), so it is linear in runtime. Equal
    total power isolates the axis.

        total_gpus 1/2/4 at 300/150/75 W per GPU: 300 W total each, so power_score is 1.0
        throughputs 300/400/600 tok/s: time_cost 1/300, 1/400, 1/600
        throughput_score(400) = (1/300 - 1/400) / (1/300 - 1/600) = 0.5; 300 gives 0.0, 600 gives 1.0
        Min-max over throughput itself would give (400 - 300) / (600 - 300) = 1/3.
    """
    table = {(1, 4): (300.0, 300.0), (2, 4): (400.0, 150.0), (4, 4): (600.0, 75.0)}
    recs = _pipeline(table, total_gpus=[1, 2, 4], policy="performance", alpha=0.2, beta=0.8).recommend(
        _workload(), _context()
    )
    by_gpus = {r.total_gpus: r.metadata for r in recs}
    assert by_gpus[1]["throughput_score"] == pytest.approx(0.0)
    assert by_gpus[2]["throughput_score"] == pytest.approx(0.5)
    assert by_gpus[4]["throughput_score"] == pytest.approx(1.0)
    # Min-max over throughput itself would give 1/3 for the 400 tok/s config.
    assert by_gpus[2]["throughput_score"] != pytest.approx(1.0 / 3.0)
    # Every config draws 300 W in total, so every power_score is 1.0.
    assert by_gpus[2]["power_score"] == pytest.approx(1.0)


def test_lone_feasible_candidate_gets_degenerate_scores_of_one():
    """With one feasible candidate both min-max denominators are zero, and each score is 1.0."""
    table = {(2, 4): (500.0, 137.0)}
    recs = _pipeline(table, total_gpus=[2], policy="energy", alpha=0.8, beta=0.2).recommend(_workload(), _context())
    assert len(recs) == 1
    assert recs[0].metadata["power_score"] == pytest.approx(1.0)
    assert recs[0].metadata["throughput_score"] == pytest.approx(1.0)
    # combined = 0.8 x 1.0 + 0.2 x 1.0 = 1.0
    assert recs[0].metadata["combined_score"] == pytest.approx(1.0)


def test_predicted_power_watts_stays_per_gpu_and_tokens_per_watt_uses_it():
    """The reported power stays per GPU, and tokens_per_watt uses it. For 2 GPUs at 137 W and
    500 tok/s, predicted_power_watts is 137 (the total is 274) and tokens_per_watt is
    500 / 137 = 3.6496.
    """
    table = {(2, 4): (500.0, 137.0)}
    recs = _pipeline(table, total_gpus=[2], policy="energy").recommend(_workload(), _context())
    assert recs[0].metadata["predicted_power_watts"] == pytest.approx(137.0)
    assert recs[0].metadata["predicted_power_watts"] != pytest.approx(274.0)
    assert recs[0].metadata["tokens_per_watt"] == pytest.approx(3.649635, abs=1e-4)
    assert recs[0].metadata["tokens_per_watt"] != pytest.approx(500.0 / 274.0)
