"""The min-GPU policy follows the thesis min-GPU algorithm, as in IBM AutoConf's min-GPU recommender.

    T = per-device batch x the job's GPUs
    g = 1
    while g <= C.maxGPUs:
        c = the workload with g GPUs in total and T / g per device
        if F(c): return c
        g = 2 * g
    return nothing

The job's GPUs are gpus_per_node x number_of_nodes, or 1 when the workload gives no layout. A g
that does not divide T, gives less than 1 per device, or needs more than max_nodes nodes is
skipped. The node layout is the grid's exact one. No simulation decides the pick: only the
returned candidates are simulated, to report their predictions. With top_k > 1 the first top_k
feasible candidates come back in doubling order. Without a top_k, min-GPU returns one.

A recording feasibility checker and a recording predictor show which candidates were checked and
which were simulated.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import pytest

import coastline
import coastline.sdk.policies as policies
from coastline.sdk.exceptions import NoFeasibleGPUCountError, NoPredictionError
from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.recommendation import Prediction
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.pipeline.feasibility import NoOpFeasibilityChecker
from coastline.sdk.pipeline.workflow import GridWorkflowPipeline
from coastline.sdk.policies import PolicyFactory
from coastline.sdk.policies.min_gpu import MinGPUStrategy

GPU = "NVIDIA-A100-SXM4-80GB"


class FeasibleFrom:
    """Feasible when the candidate has at least ``min_gpus`` GPUs. Records every check."""

    def __init__(self, min_gpus: int) -> None:
        self.min_gpus = min_gpus
        self.checked: list[tuple[int, int, int, int]] = []

    def is_feasible(self, workload: WorkloadSpec) -> tuple[bool, dict[str, Any]]:
        self.checked.append(
            (workload.total_gpus, workload.batch_size, workload.gpus_per_node or 1, workload.number_of_nodes or 1)
        )
        if workload.total_gpus >= self.min_gpus:
            return True, {"checked_by": "test"}
        return False, {"reason": f"needs {self.min_gpus} GPUs"}

    @property
    def checked_gpus(self) -> list[int]:
        return [gpus for gpus, _, _, _ in self.checked]

    @property
    def checked_pairs(self) -> list[tuple[int, int]]:
        """(GPUs, per-device batch) of every check."""
        return [(gpus, batch) for gpus, batch, _, _ in self.checked]


class RecordingPredictor:
    """Throughput that grows with the GPU count, so a choice by simulation would pick the largest
    candidate. ``missing`` GPU counts get no prediction. Used as both the throughput and the power
    predictor, so a simulated candidate is recorded twice; ``simulated`` lists each once."""

    def __init__(self, missing: tuple[int, ...] = ()) -> None:
        self.missing = missing
        self.calls: list[tuple[int, int]] = []

    def predict(self, workload: WorkloadSpec, context: SystemContext) -> Optional[Prediction]:
        self.calls.append((workload.total_gpus, workload.batch_size))
        if workload.total_gpus in self.missing:
            return None
        return Prediction(
            gpus_per_node=workload.gpus_per_node or 1,
            number_of_nodes=workload.number_of_nodes or 1,
            total_gpus=workload.total_gpus,
            predicted_throughput=100.0 * workload.total_gpus,
            predicted_runtime_seconds=1000.0 / workload.total_gpus,
            predicted_power=250.0,
        )

    def get_name(self) -> str:  # pragma: no cover - trivial
        return "recording"

    @property
    def simulated(self) -> list[tuple[int, int]]:
        return list(dict.fromkeys(self.calls))


def _context(max_gpus: int = 64, gpus_per_node: int = 8, max_nodes: int = 8) -> SystemContext:
    return SystemContext(
        available_gpu_models=[GPU],
        max_gpus=max_gpus,
        gpu_memory={GPU: 80},
        constraints=Constraints(max_gpus=max_gpus, gpus_per_node=gpus_per_node, max_nodes=max_nodes),
    )


def _workload(batch_size: int = 16, gpus_per_node: Optional[int] = None, nodes: Optional[int] = None) -> WorkloadSpec:
    """A job with ``batch_size`` per device; without a layout it is a 1-GPU job."""
    return WorkloadSpec(
        llm_model="mistral-7b-v0.1",
        fine_tuning_method="lora",
        gpu_model=GPU,
        tokens_per_sample=1024,
        batch_size=batch_size,
        gpus_per_node=gpus_per_node,
        number_of_nodes=nodes,
    )


def _min_gpu(
    checker: Any,
    predictor: Any = None,
    *,
    top_k: Optional[int] = 1,
    grid: Optional[dict] = None,
    strategy: Optional[dict] = None,
) -> MinGPUStrategy:
    predictor = predictor or RecordingPredictor()
    grid_config: dict = {"batch_sizes": [2, 8, 32], "total_gpus": [3, 6, 12], **(grid or {})}
    if top_k is not None:
        grid_config["top_k"] = top_k
    config: dict = {"grid": grid_config}
    if strategy:
        config["strategy"] = strategy
    pipeline = GridWorkflowPipeline.from_config(
        config=config,
        selection_policy="min_gpu",
        strategy_name="min_gpu",
        throughput_predictor=predictor,
        power_predictor=predictor,
        feasibility_checker=checker,
    )
    return MinGPUStrategy(pipeline=pipeline)


def test_checks_doubling_gpu_counts_and_returns_the_first_feasible():
    checker = FeasibleFrom(4)
    recs = _min_gpu(checker).recommend(_workload(), _context())

    assert checker.checked_gpus == [1, 2, 4]  # stops at the first feasible count
    assert [r.total_gpus for r in recs] == [4]
    assert recs[0].strategy == "min_gpu"


def test_the_total_batch_is_kept_and_split_over_the_gpus():
    # 4 per device on 8 GPUs is a total batch of 32; the grid's batch sizes and GPU counts are
    # not used.
    checker = FeasibleFrom(4)
    recs = _min_gpu(checker).recommend(_workload(batch_size=4, gpus_per_node=8, nodes=1), _context())

    assert checker.checked_pairs == [(1, 32), (2, 16), (4, 8)]
    assert (recs[0].total_gpus, recs[0].metadata["batch_size"]) == (4, 8)


def test_a_workload_without_a_layout_is_a_one_gpu_job():
    checker = FeasibleFrom(1000)
    with pytest.raises(NoFeasibleGPUCountError):
        _min_gpu(checker).recommend(_workload(batch_size=8), _context())

    assert checker.checked_pairs == [(1, 8), (2, 4), (4, 2), (8, 1)]


def test_the_job_gpus_count_every_node():
    checker = FeasibleFrom(1)
    recs = _min_gpu(checker).recommend(_workload(batch_size=2, gpus_per_node=8, nodes=2), _context())

    # 2 per device on 16 GPUs is a total batch of 32.
    assert checker.checked_pairs == [(1, 32)]
    assert (recs[0].total_gpus, recs[0].metadata["batch_size"]) == (1, 32)


def test_a_count_that_does_not_divide_the_total_is_skipped():
    checker = FeasibleFrom(1000)
    with pytest.raises(NoFeasibleGPUCountError):
        _min_gpu(checker).recommend(_workload(batch_size=12), _context())

    # 12 / 8 and 12 / 16 are not whole, and 32 GPUs would get less than 1 per device.
    assert checker.checked_pairs == [(1, 12), (2, 6), (4, 3)]


def test_the_loop_ends_at_the_contexts_maximum_gpus():
    checker = FeasibleFrom(1000)
    with pytest.raises(RuntimeError, match="no feasible GPU count") as excinfo:
        _min_gpu(checker).recommend(_workload(batch_size=64), _context(max_gpus=12, max_nodes=2))

    assert checker.checked_gpus == [1, 2, 4, 8]  # 16 > 12
    assert not isinstance(excinfo.value, NoPredictionError)


def test_the_layout_follows_the_exact_rule_within_node_and_node_count_limits():
    # 4 GPUs per node and at most 2 nodes: 16 GPUs would need 4 nodes, so it is not a candidate.
    checker = FeasibleFrom(1000)
    with pytest.raises(NoFeasibleGPUCountError):
        _min_gpu(checker).recommend(_workload(batch_size=64), _context(max_gpus=16, gpus_per_node=4, max_nodes=2))

    assert [(gpn, nodes) for _, _, gpn, nodes in checker.checked] == [(1, 1), (2, 1), (4, 1), (4, 2)]


def test_only_the_returned_candidate_is_simulated():
    predictor = RecordingPredictor()
    recs = _min_gpu(FeasibleFrom(2), predictor).recommend(_workload(), _context())

    # Throughput grows with the GPU count, yet the pick is the first feasible count.
    assert [r.total_gpus for r in recs] == [2]
    assert predictor.simulated == [(2, 8)]


def test_top_k_returns_the_first_feasible_counts_in_doubling_order_each_simulated():
    checker = FeasibleFrom(2)
    predictor = RecordingPredictor()
    recs = _min_gpu(checker, predictor, top_k=3).recommend(_workload(), _context())

    assert [(r.total_gpus, r.metadata["batch_size"]) for r in recs] == [(2, 8), (4, 4), (8, 2)]
    assert [r.metadata["rank"] for r in recs] == [1, 2, 3]
    assert checker.checked_gpus == [1, 2, 4, 8]  # stops once three are feasible
    assert predictor.simulated == [(2, 8), (4, 4), (8, 2)]


def test_without_a_top_k_min_gpu_returns_one():
    recs = _min_gpu(FeasibleFrom(1), top_k=None).recommend(_workload(), _context())
    assert [r.total_gpus for r in recs] == [1]


def test_the_returned_candidates_carry_their_predictions():
    recs = _min_gpu(FeasibleFrom(4)).recommend(_workload(), _context())
    rec = recs[0]

    assert (rec.gpus_per_node, rec.number_of_nodes) == (4, 1)
    assert rec.metadata["batch_size"] == 4
    assert rec.predicted_throughput == pytest.approx(400.0)
    assert rec.predicted_runtime_seconds == pytest.approx(250.0)
    assert rec.metadata["predicted_power_watts"] == pytest.approx(250.0)
    assert rec.metadata["tokens_per_watt"] == pytest.approx(400.0 / 250.0)
    assert rec.metadata["selection_policy"] == "min_gpu"
    assert rec.metadata["feasibility"] == {"checked_by": "test"}
    # min-GPU has no weighted score.
    assert rec.metadata["throughput_score"] is None
    assert rec.metadata["power_score"] is None


def test_no_feasible_count_raises_an_error_that_says_so():
    with pytest.raises(NoFeasibleGPUCountError) as excinfo:
        _min_gpu(FeasibleFrom(1000)).recommend(_workload(batch_size=64), _context(max_gpus=8, max_nodes=1))

    error = excinfo.value
    assert isinstance(error, RuntimeError)
    assert error.total_batch == 64
    assert error.gpu_counts == [1, 2, 4, 8]
    assert "Workflow (min_gpu): no feasible GPU count" in str(error)
    assert "total batch of 64" in str(error)
    # A worker process can send it back.
    copy = pickle.loads(pickle.dumps(error))
    assert (str(copy), copy.total_batch, copy.gpu_counts) == (str(error), 64, [1, 2, 4, 8])


@pytest.mark.parametrize("top_k", [1, 2, 5])
def test_an_unpredictable_pick_is_dropped_for_every_top_k(top_k):
    # 1 GPU is the first feasible count, but the predictor has no number for it.
    predictor = RecordingPredictor(missing=(1,))
    strategy = _min_gpu(NoOpFeasibilityChecker(), predictor, top_k=top_k)
    if top_k == 1:
        with pytest.raises(NoPredictionError, match="no usable prediction"):
            strategy.recommend(_workload(), _context())
        return
    recs = strategy.recommend(_workload(), _context())
    assert 1 not in [r.total_gpus for r in recs]
    assert recs[0].total_gpus == 2
    assert [r.metadata["rank"] for r in recs] == list(range(1, len(recs) + 1))
    assert all(r.predicted_throughput for r in recs)


def test_a_later_unpredictable_pick_is_dropped():
    predictor = RecordingPredictor(missing=(2,))
    recs = _min_gpu(NoOpFeasibilityChecker(), predictor, top_k=3).recommend(_workload(), _context())

    assert [r.total_gpus for r in recs] == [1, 4]
    assert [r.metadata["rank"] for r in recs] == [1, 2]


def test_no_predictable_pick_raises():
    predictor = RecordingPredictor(missing=(1, 2, 4))
    with pytest.raises(NoPredictionError, match="no usable prediction"):
        _min_gpu(NoOpFeasibilityChecker(), predictor, top_k=3).recommend(_workload(), _context())


def test_max_slowdown_does_not_change_the_pick():
    # The runtime guard needs every candidate simulated, which min-GPU does not do.
    recs = _min_gpu(FeasibleFrom(1), strategy={"runtime_guard_k": 1.0}).recommend(_workload(), _context())
    assert [r.total_gpus for r in recs] == [1]


# The tests below build min_gpu with PolicyFactory, so the configured feasibility check decides.


def _factory_config(predictors: dict, top_k: int = 1) -> dict:
    return {
        "strategy": {"name": "min_gpu"},
        "predictors": {"performance": "kavier", "energy": "kavier_power", **predictors},
        "grid": {"batch_sizes": [2, 64], "total_gpus": [3], "top_k": top_k},
    }


def test_the_factory_uses_the_configured_feasibility_check(monkeypatch):
    seen: list[dict] = []
    checker = FeasibleFrom(4)

    def factory(predictor_config: dict) -> FeasibleFrom:
        seen.append(dict(predictor_config))
        return checker

    monkeypatch.setattr(policies, "create_feasibility_checker", factory)
    strategy = PolicyFactory.create_strategy(config=_factory_config({"feasibility": "autoconf"}))
    recs = strategy.recommend(_workload(batch_size=16), _context())

    assert seen and seen[0]["feasibility"] == "autoconf"
    assert checker.checked_gpus == [1, 2, 4]
    assert [(r.total_gpus, r.metadata["batch_size"]) for r in recs] == [(4, 4)]
    assert recs[0].predicted_throughput and recs[0].predicted_throughput > 0


def test_rules_feasibility_admits_one_gpu_at_the_jobs_total_batch():
    strategy = PolicyFactory.create_strategy(config=_factory_config({"feasibility": "rules"}))
    recs = strategy.recommend(_workload(batch_size=4, gpus_per_node=4, nodes=1), _context())
    assert [(r.total_gpus, r.metadata["batch_size"]) for r in recs] == [(1, 16)]


def test_the_empirical_oom_guard_applies():
    # A total batch of 16 at 8192 tokens: 1 GPU holds 131,072 tokens and 2 GPUs 65,536, both over
    # the 60,224 budget; 4 GPUs hold 32,768.
    workload = WorkloadSpec(
        llm_model="mistral-7b-v0.1", fine_tuning_method="lora", gpu_model=GPU, tokens_per_sample=8192, batch_size=16
    )
    strategy = PolicyFactory.create_strategy(
        config=_factory_config({"feasibility": "rules", "empirical_oom_guard": True})
    )
    recs = strategy.recommend(workload, _context())
    assert [(r.total_gpus, r.metadata["batch_size"]) for r in recs] == [(4, 4)]


# Thesis E4, job 15: 4 per device on 8 GPUs, a total batch of 32.
E4_JOB_15 = {
    "llm_model": "granite-3.1-8b-instruct",
    "fine_tuning_method": "full",
    "gpu_model": GPU,
    "tokens_per_sample": 7621,
    "batch_size": 4,
    "gpus_per_node": 8,
    "number_of_nodes": 1,
}


def test_e4_job_15_moves_to_four_gpus_at_eight_per_device():
    # AutoConf rejects 1 GPU at 32 and 2 GPUs at 16 and accepts 4 GPUs at 8, which is the
    # configuration E4 ran.
    recs = coastline.Coastline("kavier", feasibility="autoconf").recommend(E4_JOB_15, goal="min_gpu")

    assert len(recs) == 1
    assert (recs[0].total_gpus, recs[0].metadata["batch_size"]) == (4, 8)


def test_the_docs_example_names_the_e4_workload():
    # The AutoConf verdicts in the example hold for this workload only, so the docs name it.
    docs = Path(__file__).resolve().parents[2] / "docs" / "recommendation.md"
    section = docs.read_text(encoding="utf-8").split("## Min-GPU", 1)[1]
    example = next(paragraph for paragraph in section.split("\n\n") if paragraph.startswith("For example"))
    example = " ".join(example.split())
    for value in ("granite-3.1-8b-instruct", "full fine-tuning", "7621 tokens per sample", GPU):
        assert value in example, value
