"""The parallel stages of GridWorkflowPipeline.recommend: chunking, gather order and when to fork.

recommend checks feasibility for every candidate, simulates every survivor and then ranks the whole
set. The first two stages may split their candidates across worker processes, and any worker count
must give the sequential result: rank_candidates keeps grid order for some equal scores, so chunks
are contiguous and gathered in chunk order. A stage forks only if it is expensive and large enough;
the cheap backends (rules, none, Kavier) and a checker that decides a whole chunk in one call run
inline. The tests check the path taken as well as the result.

No process is started: get_pool is stubbed to return None, so map_chunks runs the module-level
worker entry points in this process, and TestWorkerEntryPoints checks that they pickle.
os.cpu_count is fixed at 8 for the whole module.
"""

from __future__ import annotations

import pickle
from concurrent.futures.process import BrokenProcessPool
from typing import Any

import pytest

from coastline.sdk.constants import DEFAULT_AUTOCONF_MODEL_VERSION
from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.recommendation import Prediction
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.pipeline import parallel
from coastline.sdk.pipeline.feasibility import (
    FeasibilityChecker,
    GuardedFeasibilityChecker,
    NoOpFeasibilityChecker,
    RulesFeasibilityChecker,
    TokenBudgetFeasibilityChecker,
    _RulesThenAutoconfChecker,
    create_feasibility_checker,
    evaluate_chunk,
    is_expensive,
)
from coastline.sdk.pipeline.grid import GridConfig, generate_candidates
from coastline.sdk.pipeline.parallel import (
    MIN_ITEMS_PER_ROW_STAGE,
    MIN_ITEMS_TO_START_A_POOL,
    chunk,
    map_chunks,
    plan_chunks,
    resolve_workers,
    run_feasibility,
    run_simulation,
)
from coastline.sdk.pipeline.workflow import GridWorkflowPipeline, simulate_chunk, simulate_one
from coastline.sdk.predictors.base import BasePredictor

GPU = "NVIDIA-A100-SXM4-80GB"

#: resolve_workers clamps to os.cpu_count(), so the tests fix the core count.
FAKE_CPU_COUNT = 8


@pytest.fixture(autouse=True)
def fixed_cpu_count(monkeypatch):
    """Fix the core count so the worker clamping does not depend on the host."""
    monkeypatch.setattr("os.cpu_count", lambda: FAKE_CPU_COUNT)


@pytest.fixture(autouse=True)
def clean_parallel_state():
    """Leave no pool and no memoized worker state behind (the module keeps both as globals)."""
    parallel.shutdown_pool()
    parallel._WORKER_CHECKERS.clear()
    parallel._WORKER_PREDICTORS.clear()
    yield
    parallel.shutdown_pool()
    parallel._WORKER_CHECKERS.clear()
    parallel._WORKER_PREDICTORS.clear()


@pytest.fixture
def no_pool(monkeypatch):
    """Make ``get_pool`` return None, so ``map_chunks`` runs the workers in this process.

    Returns the worker counts it was asked for, so a test can check that a stage tried to fork.
    The pool reports warm; TestPlanChunks covers the cold-start threshold.
    """
    asked: list[int] = []

    def _fake_get_pool(workers: int):
        asked.append(workers)
        return None

    monkeypatch.setattr(parallel, "get_pool", _fake_get_pool)
    monkeypatch.setattr(parallel, "pool_is_warm", lambda: True)
    return asked


@pytest.fixture
def map_calls(monkeypatch):
    """Record every ``map_chunks`` call and pass it on to the real one.

    run_feasibility and run_simulation call map_chunks only when they fork, so an empty record
    means the stage ran inline.
    """
    calls: list[dict[str, Any]] = []
    real = parallel.map_chunks

    def _spy(worker, payloads, workers, *, stage):
        calls.append({"worker": worker, "payloads": payloads, "workers": workers, "stage": stage})
        return real(worker, payloads, workers, stage=stage)

    monkeypatch.setattr(parallel, "map_chunks", _spy)
    return calls


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


class _RecordingChecker:
    """A checker with is_feasible and no check_chunk.

    Rejects candidates whose total_gpus is in ``reject`` and records the order of the calls.
    """

    def __init__(self, reject: set[int] | None = None, expensive: bool = False):
        self.reject = reject or set()
        self.EXPENSIVE = expensive
        self.seen: list[int] = []

    def is_feasible(self, workload: WorkloadSpec) -> tuple[bool, dict[str, Any]]:
        self.seen.append(workload.total_gpus)
        if workload.total_gpus in self.reject:
            return False, {"error": "rejected by double", "total_gpus": workload.total_gpus}
        return True, {"total_gpus": workload.total_gpus}


class _ChunkingChecker:
    """A checker that offers check_chunk; ``n_verdicts`` lets a test return the wrong count."""

    def __init__(self, n_verdicts: int | None = None):
        self.n_verdicts = n_verdicts
        self.chunk_calls = 0
        self.single_calls = 0

    def is_feasible(self, workload: WorkloadSpec) -> tuple[bool, dict[str, Any]]:
        self.single_calls += 1
        return True, {"path": "single"}

    def check_chunk(self, workloads) -> list[tuple[bool, dict[str, Any]]]:
        self.chunk_calls += 1
        count = len(workloads) if self.n_verdicts is None else self.n_verdicts
        return [(True, {"path": "chunk", "position": i}) for i in range(count)]


class _ExpensiveRulesChecker(RulesFeasibilityChecker):
    """The rules backend marked expensive, so a stage may fork.

    A worker rebuilds a plain RulesFeasibilityChecker from ``feasibility: rules``, which gives the
    same verdicts, so forked and inline runs can be compared.
    """

    EXPENSIVE = True


class _BatchingRulesChecker(_ExpensiveRulesChecker):
    """Expensive and batched, like the AutoConf backend on the verified model version.

    One call decides a whole chunk, so the stage never forks. The verdicts are the rules backend's,
    so a run that did fork would still be comparable.
    """

    def batches(self) -> bool:
        return True

    def check_chunk(self, workloads) -> list[tuple[bool, dict[str, Any]]]:
        return [self.is_feasible(workload) for workload in workloads]


class _LinearPredictor(BasePredictor):
    """Throughput = 100 * total_gpus + batch_size and per-GPU power = 50 * total_gpus.

    Each (total_gpus, batch_size) gets its own throughput, so a reordered gather shows up.
    """

    EXPENSIVE = False

    def __init__(self, expensive: bool = False):
        self.EXPENSIVE = expensive

    def predict(self, workload: WorkloadSpec, context: SystemContext) -> Prediction:
        total = workload.total_gpus
        return Prediction(
            gpus_per_node=workload.gpus_per_node or 1,
            number_of_nodes=workload.number_of_nodes or 1,
            total_gpus=total,
            predicted_throughput=100.0 * total + workload.batch_size,
            predicted_power=50.0 * total,
            predicted_runtime_seconds=1000.0 / total,
        )

    def get_name(self) -> str:
        return "linear-stub"


class _DeadPool:
    """A pool whose map() raises BrokenProcessPool. Records whether it was shut down."""

    def __init__(self):
        self.shutdown_called = False

    def map(self, worker, payloads):
        raise BrokenProcessPool("a worker process terminated abruptly")

    def shutdown(self, **kwargs):
        self.shutdown_called = True


def _double_all(payload: list[int]) -> list[int]:
    """A module-level map_chunks worker (the pool pickles workers by name, so it cannot be local)."""
    return [item * 2 for item in payload]


def _candidates(context, batch_sizes=(8,), total_gpus=(1, 2, 4, 8)) -> list[WorkloadSpec]:
    """The grid a stage receives: batch_sizes x total_gpus, in generate_candidates' order."""
    base = WorkloadSpec(
        llm_model="mistral-7b-v0.1",
        fine_tuning_method="lora",
        gpu_model=GPU,
        tokens_per_sample=1024,
        batch_size=8,
    )
    return generate_candidates(base, context, GridConfig(batch_sizes=list(batch_sizes), total_gpus=list(total_gpus)))


def _config(workers: int, *, feasibility: str = "rules", total_gpus=(1, 2, 4, 8), batch_sizes=(4, 8)) -> dict:
    return {
        "grid": {"batch_sizes": list(batch_sizes), "total_gpus": list(total_gpus), "top_k": 16},
        "predictors": {"feasibility": feasibility},
        "runtime": {"parallel_workers": workers},
    }


def _pipeline(config: dict, **overrides) -> GridWorkflowPipeline:
    """Pipeline with stub predictors; the feasibility checker is built from the config."""
    kwargs: dict[str, Any] = {
        "throughput_predictor": _LinearPredictor(),
        "power_predictor": _LinearPredictor(),
    }
    kwargs.update(overrides)
    return GridWorkflowPipeline.from_config(
        config=config,
        selection_policy="performance",
        strategy_name="parallel-stages",
        **kwargs,
    )


def _dump(recs) -> list[dict]:
    """Recommendations as plain data, for a field-by-field comparison of two runs."""
    return [r.model_dump() for r in recs]


# chunk(): contiguous, in order, complete
class TestChunk:
    @pytest.mark.parametrize(
        "n_items, n_chunks, expected_sizes",
        [
            (8, 2, [4, 4]),  # exact split
            (8, 4, [2, 2, 2, 2]),
            (7, 2, [4, 3]),  # the remainder goes to the first chunks
            (7, 3, [3, 2, 2]),
            (10, 4, [3, 3, 2, 2]),
            (1, 1, [1]),
        ],
    )
    def test_sizes_are_balanced_with_the_remainder_first(self, n_items, n_chunks, expected_sizes):
        items = list(range(n_items))
        parts = chunk(items, n_chunks)
        assert [len(p) for p in parts] == expected_sizes
        assert sum(len(p) for p in parts) == n_items  # nothing lost, nothing duplicated

    @pytest.mark.parametrize("n_items", [0, 1, 2, 5, 9, 16])
    @pytest.mark.parametrize("n_chunks", [1, 2, 3, 4, 8])
    def test_chunks_are_contiguous_and_concatenate_back_to_the_input(self, n_items, n_chunks):
        items = list(range(n_items))
        parts = chunk(items, n_chunks)
        # The gather concatenates chunk results, so the chunks must concatenate back to the input.
        assert [item for part in parts for item in part] == items
        # Each chunk is a contiguous slice of the input.
        start = 0
        for part in parts:
            assert part == items[start : start + len(part)]
            start += len(part)

    @pytest.mark.parametrize("n_chunks", [1, 0, -3])
    def test_one_or_fewer_chunks_means_everything_in_one_piece(self, n_chunks):
        items = [1, 2, 3]
        assert chunk(items, n_chunks) == [[1, 2, 3]]

    def test_more_chunks_than_items_pads_with_empties_and_keeps_order(self):
        # plan_chunks never asks for more chunks than items, but chunk() still handles it.
        parts = chunk([1, 2], 5)
        assert parts == [[1], [2], [], [], []]
        assert [item for part in parts for item in part] == [1, 2]

    def test_empty_sequence_yields_empty_chunks(self):
        assert chunk([], 1) == [[]]
        assert chunk([], 3) == [[], [], []]

    def test_copies_rather_than_aliasing_the_input(self):
        items = [1, 2, 3, 4]
        parts = chunk(items, 2)
        parts[0][0] = 99
        assert items == [1, 2, 3, 4]


# plan_chunks() / resolve_workers()
class TestPlanChunks:
    @pytest.fixture(autouse=True)
    def warm_pool(self, monkeypatch):
        """These tables assume a running pool.

        A cold pool needs MIN_ITEMS_TO_START_A_POOL items, since the first stage to fork also pays
        each worker's model load; test_a_cold_pool_has_to_earn_its_start_up covers that case.
        """
        monkeypatch.setattr(parallel, "pool_is_warm", lambda: True)

    @pytest.mark.parametrize(
        "n_items, workers, expected",
        [
            # Per-candidate stage: the default floor is MIN_ITEMS_PER_ROW_STAGE = 8.
            (0, 4, 1),  # empty grid: nothing to fork
            (1, 4, 1),
            (7, 8, 1),  # one below the floor: inline at any worker count
            (8, 1, 1),  # one worker: inline
            (8, 0, 1),
            (8, 2, 2),  # at the floor: 8 // (8//2) = 2 chunks
            (8, 4, 4),  # 8 // (8//4) = 4
            (8, 8, 8),  # 8 // (8//8) = 8
            (9, 4, 4),
            (100, 4, 4),  # at most one chunk per worker
            (100, 2, 2),
        ],
    )
    def test_plan_table_for_a_per_candidate_stage(self, n_items, workers, expected):
        assert plan_chunks(n_items, workers) == expected
        assert plan_chunks(n_items, workers, MIN_ITEMS_PER_ROW_STAGE) == expected  # the default

    @pytest.mark.parametrize(
        "n_items, expected",
        [
            (8, 1),  # would fork with a warm pool; not worth starting one
            (256, 1),  # still below the cold-start threshold
            (MIN_ITEMS_TO_START_A_POOL - 1, 1),
            (MIN_ITEMS_TO_START_A_POOL, 4),
            (4 * MIN_ITEMS_TO_START_A_POOL, 4),
        ],
    )
    def test_a_cold_pool_has_to_earn_its_start_up(self, n_items, expected, monkeypatch):
        # Each new worker loads its models first, so the stage that starts the pool has to be big
        # enough to win that back. Once the pool is up the usual floors apply.
        monkeypatch.setattr(parallel, "pool_is_warm", lambda: False)
        assert plan_chunks(n_items, 4) == expected

    @pytest.mark.parametrize("workers", [1, 2, 3, 4, 8])
    @pytest.mark.parametrize("min_items", [MIN_ITEMS_PER_ROW_STAGE, 256])
    def test_a_planned_chunk_is_never_starved(self, workers, min_items):
        # For grid sizes around the floor: at most one chunk per worker, chunks that give back the
        # input, no fork below the floor, and no chunk smaller than a worker's share of the floor
        # (the share keeps a fork from costing more than the work).
        share = max(1, min_items // workers or 1)
        for n_items in list(range(0, 20)) + [min_items - 1, min_items, min_items + 1, 4 * min_items]:
            items = list(range(n_items))
            n_chunks = plan_chunks(n_items, workers, min_items)
            parts = chunk(items, n_chunks)
            assert 1 <= n_chunks <= max(1, workers), (n_items, workers, min_items)
            assert len(parts) == n_chunks, (n_items, workers, min_items)
            assert [item for part in parts for item in part] == items, (n_items, workers, min_items)
            if n_items < min_items:
                assert n_chunks == 1, (n_items, workers, min_items)
            if n_chunks > 1:
                assert min(len(p) for p in parts) >= share, (n_items, workers, min_items)


class TestResolveWorkers:
    @pytest.mark.parametrize("requested", [None, 0, -4, 1])
    def test_floor_is_one(self, requested):
        assert resolve_workers(requested) == 1

    @pytest.mark.parametrize("requested, expected", [(2, 2), (4, 4), (8, 8), (64, FAKE_CPU_COUNT)])
    def test_clamped_to_the_core_count(self, requested, expected):
        # os.cpu_count() is fixed at FAKE_CPU_COUNT by the autouse fixture.
        assert resolve_workers(requested) == expected


# the pool and the gather
class TestPoolAndGather:
    @pytest.mark.parametrize("workers", [1, 0, -1])
    def test_sequential_worker_counts_never_create_a_pool(self, workers):
        assert parallel.get_pool(workers) is None
        assert parallel._POOL is None

    def test_shutdown_is_idempotent(self):
        parallel.shutdown_pool()
        parallel.shutdown_pool()
        assert parallel._POOL is None
        assert parallel._POOL_WORKERS == 0

    def test_map_chunks_runs_inline_and_concatenates_in_chunk_order(self):
        # workers=1 uses no pool; the results come back in payload order.
        payloads = [[1, 2], [3], [4, 5, 6]]
        assert map_chunks(_double_all, payloads, 1, stage="feasibility") == [2, 4, 6, 8, 10, 12]

    def test_map_chunks_of_nothing_is_nothing(self):
        assert map_chunks(_double_all, [], 1, stage="feasibility") == []

    def test_a_dead_worker_is_recovered_in_process_not_turned_into_a_different_answer(self, monkeypatch):
        # Both callers catch Exception per row, so a raised pool error would read as "this
        # predictor could not handle the job" and the run would still write a plausible CSV.
        # The stage finishes in this process instead, the dead pool is dropped and nothing forks
        # again in this process.
        dead = _DeadPool()
        monkeypatch.setattr(parallel, "_POOL", dead)
        monkeypatch.setattr(parallel, "_POOL_WORKERS", 2)
        monkeypatch.setattr(parallel, "_FORKING_DISABLED", False)

        recovered = map_chunks(_double_all, [[1, 2], [3, 4]], 2, stage="feasibility")

        assert recovered == [2, 4, 6, 8]  # the sequential result
        assert dead.shutdown_called
        assert parallel._POOL is None and parallel._POOL_WORKERS == 0
        assert parallel._FORKING_DISABLED is True  # no new pool after a failure
        assert parallel.get_pool(4) is None  # and get_pool respects that

    def test_work_that_fails_in_a_worker_and_again_in_process_is_raised_with_its_own_type(self, monkeypatch):
        # Recovery covers a dead pool only. Work that also fails in this process raises
        # WorkerPoolFailure, a type the per-row error handling can let through.
        dead = _DeadPool()
        monkeypatch.setattr(parallel, "_POOL", dead)
        monkeypatch.setattr(parallel, "_POOL_WORKERS", 2)
        monkeypatch.setattr(parallel, "_FORKING_DISABLED", False)

        def _always_fails(payload):
            raise ValueError("the work itself is broken")

        with pytest.raises(parallel.WorkerPoolFailure, match="failed in a worker and again in-process"):
            map_chunks(_always_fails, [[1, 2], [3, 4]], 2, stage="feasibility")


# worker-side entry points: picklable by name, rebuilt from config, memoized
class TestWorkerEntryPoints:
    @pytest.mark.parametrize(
        "func",
        [
            parallel.feasibility_worker,
            parallel.simulation_worker,
            simulate_chunk,
            simulate_one,
        ],
        ids=["feasibility_worker", "simulation_worker", "simulate_chunk", "simulate_one"],
    )
    def test_stage_entry_points_are_module_level_and_pickle_by_name(self, func):
        # The pool spawns its workers, so a worker function must resolve by qualified name. A
        # closure or a method would fail to pickle only at run time, inside the pool.
        assert pickle.loads(pickle.dumps(func)) is func

    def test_stage_payloads_survive_a_round_trip(self, context):
        candidates = _candidates(context)
        feasibility_payload = ({"feasibility": "rules"}, candidates)
        simulation_payload = ({"performance": "kavier"}, context, candidates)

        assert pickle.loads(pickle.dumps(feasibility_payload)) == feasibility_payload
        assert pickle.loads(pickle.dumps(simulation_payload)) == simulation_payload

    def test_worker_checker_is_built_from_config_and_memoized_per_config(self):
        rules = parallel.worker_checker({"feasibility": "rules"})
        assert isinstance(rules, RulesFeasibilityChecker)
        # The cache key is the config as canonical JSON, so an equal dict gets the same checker
        # and a worker builds each checker once per config.
        assert parallel.worker_checker({"feasibility": "rules"}) is rules
        assert parallel.worker_checker({"feasibility": "rules", "empirical_oom_guard": False}) is not rules

        none_checker = parallel.worker_checker({"feasibility": "none"})
        assert isinstance(none_checker, NoOpFeasibilityChecker)
        assert none_checker is not rules

    def test_worker_predictors_are_built_from_config_and_memoized(self):
        config = {"performance": "kavier", "energy": "kavier_power"}
        throughput, power = parallel.worker_predictors(config)
        assert throughput is not None and power is not None
        # Key order does not matter (canonical JSON with sort_keys).
        again = parallel.worker_predictors({"energy": "kavier_power", "performance": "kavier"})
        assert again == (throughput, power)


class TestEvaluateChunk:
    def test_protocol_carries_no_chunk_default(self):
        # No checker inherits the FeasibilityChecker Protocol, so a check_chunk default on the
        # Protocol would reach none of them. evaluate_chunk is a module-level function for that reason.
        assert not hasattr(FeasibilityChecker, "check_chunk")
        for checker_type in (RulesFeasibilityChecker, NoOpFeasibilityChecker, GuardedFeasibilityChecker):
            assert FeasibilityChecker not in checker_type.__mro__

    @pytest.mark.parametrize(
        "checker_factory",
        [
            lambda: create_feasibility_checker({"feasibility": "rules"}),
            lambda: create_feasibility_checker({"feasibility": "none"}),
            lambda: _RecordingChecker(),
        ],
        ids=["rules_backend", "none_backend", "test_double"],
    )
    def test_falls_back_to_per_candidate_is_feasible(self, checker_factory, context):
        checker = checker_factory()
        candidates = _candidates(context)
        assert not hasattr(checker, "check_chunk"), "this test only means something without check_chunk"

        verdicts = evaluate_chunk(checker, candidates)

        assert len(verdicts) == len(candidates)
        # Compare with one is_feasible call per candidate.
        assert verdicts == [checker_factory().is_feasible(candidate) for candidate in candidates]

    def test_fallback_visits_every_candidate_once_in_input_order(self, context):
        checker = _RecordingChecker(reject={4})
        candidates = _candidates(context)

        verdicts = evaluate_chunk(checker, candidates)

        assert checker.seen == [c.total_gpus for c in candidates]  # once each, in order
        assert [ok for ok, _ in verdicts] == [c.total_gpus != 4 for c in candidates]
        assert verdicts[[c.total_gpus for c in candidates].index(4)][1]["error"] == "rejected by double"

    def test_check_chunk_is_preferred_when_offered(self, context):
        checker = _ChunkingChecker()
        candidates = _candidates(context)

        verdicts = evaluate_chunk(checker, candidates)

        assert checker.chunk_calls == 1  # one call for the whole run
        assert checker.single_calls == 0  # and none per candidate
        assert [meta["position"] for _, meta in verdicts] == list(range(len(candidates)))

    @pytest.mark.parametrize("n_verdicts", [0, 3, 9], ids=["none", "too_few", "too_many"])
    def test_a_miscounted_chunk_raises_instead_of_silently_misaligning(self, n_verdicts, context):
        # Verdicts are matched to candidates by position, so a wrong count would pair a candidate
        # with another candidate's verdict.
        candidates = _candidates(context, batch_sizes=(4, 8))
        assert len(candidates) == 8
        checker = _ChunkingChecker(n_verdicts=n_verdicts)

        with pytest.raises(RuntimeError, match=rf"returned {n_verdicts} verdicts for 8 candidates"):
            evaluate_chunk(checker, candidates)

    @pytest.mark.parametrize(
        "checker_factory",
        [lambda: RulesFeasibilityChecker(), lambda: _ChunkingChecker()],
        ids=["per_candidate", "chunking"],
    )
    def test_an_empty_run_is_empty(self, checker_factory):
        assert evaluate_chunk(checker_factory(), []) == []

    def test_guarded_chunk_matches_the_per_candidate_verdicts_and_metadata(self, context):
        # When enabled, the empirical OOM guard is the outermost checker, so it offers check_chunk
        # to keep the backend's batching. Its chunked verdicts and metadata must equal the
        # per-candidate ones, in input order. With a budget of 16,384 tokens per device and
        # tokens_per_sample=1024, batch 8 passes and batch 32 is rejected.
        config = {"feasibility": "rules", "empirical_oom_guard": True, "empirical_oom_token_budget": 16384}
        checker = create_feasibility_checker(config)
        assert isinstance(checker, GuardedFeasibilityChecker)
        candidates = _candidates(context, batch_sizes=(8, 32))

        chunked = evaluate_chunk(checker, candidates)
        per_candidate = [create_feasibility_checker(config).is_feasible(c) for c in candidates]

        assert chunked == per_candidate
        # Both verdicts occur, so both cases are compared.
        assert {ok for ok, _ in chunked} == {True, False}
        assert [ok for ok, _ in chunked] == [c.batch_size == 8 for c in candidates]


# is_expensive: what is worth sending to a worker
class TestIsExpensive:
    def test_the_autoconf_chain_is_expensive(self):
        # Built directly because create_feasibility_checker returns the rules checker when AutoConf
        # is not installed. Building the chain loads no model (the predictor is lazy).
        chain = _RulesThenAutoconfChecker(model_version=DEFAULT_AUTOCONF_MODEL_VERSION)
        assert is_expensive(chain) is True

    def test_the_guard_inherits_the_cost_of_what_it_wraps(self):
        guard = TokenBudgetFeasibilityChecker(60224)
        chain = _RulesThenAutoconfChecker(model_version=DEFAULT_AUTOCONF_MODEL_VERSION)
        # Around AutoConf the guard counts as expensive, since the backend is a model.
        assert is_expensive(GuardedFeasibilityChecker(guard, chain)) is True
        # Around rules it is cheap.
        assert is_expensive(GuardedFeasibilityChecker(guard, RulesFeasibilityChecker())) is False

    @pytest.mark.parametrize(
        "checker_factory",
        [
            lambda: RulesFeasibilityChecker(),
            lambda: NoOpFeasibilityChecker(),
            lambda: TokenBudgetFeasibilityChecker(60224),
            lambda: create_feasibility_checker({"feasibility": "rules"}),
            lambda: create_feasibility_checker({"feasibility": "none"}),
            lambda: create_feasibility_checker({"feasibility": "rules", "empirical_oom_guard": True}),
        ],
        ids=["rules", "none", "token_budget", "rules_backend", "none_backend", "guarded_rules_backend"],
    )
    def test_cheap_checkers_are_cheap(self, checker_factory):
        assert is_expensive(checker_factory()) is False

    def test_an_unflagged_checker_defaults_to_cheap(self):
        # An unknown checker runs inline, since a dispatch may cost more than the work.
        assert is_expensive(_RecordingChecker()) is False
        assert is_expensive(object()) is False


# _batches: whether a checker decides a whole chunk in one call
class TestBatchesGate:
    @pytest.mark.parametrize(
        "checker_factory",
        [
            lambda: RulesFeasibilityChecker(),
            lambda: NoOpFeasibilityChecker(),
            lambda: _RecordingChecker(),
            lambda: object(),
        ],
        ids=["rules", "none", "test_double", "bare_object"],
    )
    def test_a_checker_that_says_nothing_is_treated_as_per_candidate(self, checker_factory):
        # Without a batches() method a checker counts as one call per candidate.
        assert parallel._batches(checker_factory()) is False

    def test_the_autoconf_chain_batches_only_on_the_verified_model(self, monkeypatch):
        # Batched and per-row verdicts were measured to agree on one model version; other versions
        # use one call per candidate and the per-candidate fork threshold.
        monkeypatch.delenv("COASTLINE_NO_AUTOCONF_BATCH", raising=False)
        verified = _RulesThenAutoconfChecker(model_version=DEFAULT_AUTOCONF_MODEL_VERSION)
        other = _RulesThenAutoconfChecker(model_version="0.0.0-unverified")

        assert parallel._batches(verified) is True
        assert parallel._batches(other) is False

    def test_the_opt_out_environment_variable_turns_batching_off(self, monkeypatch):
        monkeypatch.setenv("COASTLINE_NO_AUTOCONF_BATCH", "1")
        chain = _RulesThenAutoconfChecker(model_version=DEFAULT_AUTOCONF_MODEL_VERSION)
        assert parallel._batches(chain) is False

    def test_the_guard_reports_the_regime_of_the_backend_it_wraps(self, monkeypatch):
        # When enabled, the guard is the outermost checker, so it reports whether its backend
        # batches.
        monkeypatch.delenv("COASTLINE_NO_AUTOCONF_BATCH", raising=False)
        guard = TokenBudgetFeasibilityChecker(60224)
        chain = _RulesThenAutoconfChecker(model_version=DEFAULT_AUTOCONF_MODEL_VERSION)

        assert parallel._batches(GuardedFeasibilityChecker(guard, chain)) is True
        assert parallel._batches(GuardedFeasibilityChecker(guard, RulesFeasibilityChecker())) is False


# run_feasibility and run_simulation: fork only when it pays, gather in order
class TestRunFeasibility:
    def test_a_cheap_backend_runs_inline(self, context, map_calls, no_pool):
        checker = RulesFeasibilityChecker()
        candidates = _candidates(context, batch_sizes=(4, 8))
        assert len(candidates) == 8  # large enough to fork

        verdicts = run_feasibility(checker, {"feasibility": "rules"}, candidates, workers=4)

        assert verdicts == [checker.is_feasible(c) for c in candidates]
        assert map_calls == [], "the rules backend is two integer comparisons; forking it would cost more than the work"
        assert no_pool == []

    def test_no_predictor_config_means_no_fork(self, context, map_calls, no_pool):
        # Workers rebuild the checker from the predictor config, so without one the stage cannot
        # fork, however expensive the backend.
        checker = _ExpensiveRulesChecker()
        candidates = _candidates(context, batch_sizes=(4, 8))

        verdicts = run_feasibility(checker, None, candidates, workers=4)

        assert verdicts == [checker.is_feasible(c) for c in candidates]
        assert map_calls == []

    def test_one_worker_means_no_fork(self, context, map_calls):
        checker = _ExpensiveRulesChecker()
        candidates = _candidates(context, batch_sizes=(4, 8))

        verdicts = run_feasibility(checker, {"feasibility": "rules"}, candidates, workers=1)

        assert verdicts == [checker.is_feasible(c) for c in candidates]
        assert map_calls == []

    def test_forked_verdicts_are_the_sequential_verdicts_in_the_same_order(self, context, map_calls, no_pool):
        # Workers rebuild a rules checker from the config, which decides like the checker marked
        # expensive, so any difference in the output comes from the chunking.
        candidates = _candidates(context, batch_sizes=(4, 8))
        sequential = run_feasibility(RulesFeasibilityChecker(), {"feasibility": "rules"}, candidates, workers=1)

        forked = run_feasibility(_ExpensiveRulesChecker(), {"feasibility": "rules"}, candidates, workers=4)

        assert forked == sequential
        assert len(map_calls) == 1
        call = map_calls[0]
        assert call["stage"] == "feasibility"
        assert call["worker"] is parallel.feasibility_worker
        assert len(call["payloads"]) == plan_chunks(len(candidates), 4) == 4
        # The chunks are contiguous and give back the candidate list in order.
        assert [c for _, part in call["payloads"] for c in part] == candidates
        assert all(cfg == {"feasibility": "rules"} for cfg, _ in call["payloads"])
        assert no_pool == [4]  # it tried to fork with 4 workers

    @pytest.mark.parametrize("size", [8, 256, 4096])
    def test_a_batched_stage_never_forks_at_any_size(self, context, map_calls, no_pool, size):
        # One call decides a whole chunk, and splitting it pays that call's fixed cost once per
        # chunk, so forking gains nothing at any size.
        grid = _candidates(context, batch_sizes=(4, 8))
        candidates = (grid * (size // len(grid) + 1))[:size]

        verdicts = run_feasibility(_BatchingRulesChecker(), {"feasibility": "rules"}, candidates, workers=4)

        assert verdicts == [RulesFeasibilityChecker().is_feasible(c) for c in candidates]
        assert map_calls == []  # did not fork

    def test_the_same_checker_without_batching_does_fork(self, context, map_calls, no_pool):
        # Without batching (another AutoConf version, or COASTLINE_NO_AUTOCONF_BATCH=1) the cost
        # is per candidate, so forking pays.
        grid = _candidates(context, batch_sizes=(4, 8))
        candidates = (grid * 64)[:512]
        sequential = run_feasibility(RulesFeasibilityChecker(), {"feasibility": "rules"}, candidates, workers=1)

        forked = run_feasibility(_ExpensiveRulesChecker(), {"feasibility": "rules"}, candidates, workers=4)

        assert forked == sequential
        assert len(map_calls) == 1
        assert [c for _, part in map_calls[0]["payloads"] for c in part] == candidates

    @pytest.mark.parametrize("workers", [1, 4])
    def test_an_empty_candidate_list_is_empty_verdicts(self, workers, map_calls):
        assert run_feasibility(_ExpensiveRulesChecker(), {"feasibility": "rules"}, [], workers) == []
        assert map_calls == []  # nothing to fork


class TestRunSimulation:
    def test_a_cheap_predictor_runs_inline(self, context, map_calls, no_pool):
        throughput, power = _LinearPredictor(), _LinearPredictor()
        candidates = _candidates(context, batch_sizes=(4, 8))

        results = run_simulation(throughput, power, {"performance": "kavier"}, candidates, context, workers=4)

        assert results == [simulate_one(throughput, power, c, context) for c in candidates]
        assert map_calls == []

    def test_forked_predictions_are_the_sequential_predictions_in_the_same_order(
        self, context, map_calls, no_pool, monkeypatch
    ):
        # Workers get the same stubs, so only the chunking and the gather are tested. Each
        # candidate has its own output, so a reordered gather fails.
        throughput, power = _LinearPredictor(expensive=True), _LinearPredictor()
        monkeypatch.setattr(parallel, "worker_predictors", lambda config: (throughput, power))
        candidates = _candidates(context, batch_sizes=(4, 8))
        sequential = [simulate_one(throughput, power, c, context) for c in candidates]

        forked = run_simulation(throughput, power, {"performance": "stub"}, candidates, context, workers=4)

        assert forked == sequential
        assert len(set(forked)) == len(forked), "the oracle must be order-sensitive"
        assert len(map_calls) == 1
        assert map_calls[0]["stage"] == "simulation"
        assert map_calls[0]["worker"] is parallel.simulation_worker
        assert [c for _, _, part in map_calls[0]["payloads"] for c in part] == candidates

    def test_only_the_throughput_predictor_decides_the_fork(self, context, map_calls, no_pool):
        # Kavier-style predictors return the power with the throughput, so an expensive power
        # predictor behind a cheap throughput predictor does not make the stage fork.
        throughput, power = _LinearPredictor(expensive=False), _LinearPredictor(expensive=True)
        candidates = _candidates(context, batch_sizes=(4, 8))

        run_simulation(throughput, power, {"performance": "stub"}, candidates, context, workers=4)

        assert map_calls == []

    @pytest.mark.parametrize("workers", [1, 4])
    def test_nothing_to_simulate_is_no_predictions(self, workers, context, map_calls):
        throughput, power = _LinearPredictor(expensive=True), _LinearPredictor()
        assert run_simulation(throughput, power, {"performance": "stub"}, [], context, workers) == []
        assert map_calls == []


# GridWorkflowPipeline.recommend end to end
class TestStagedPipeline:
    def test_from_config_reads_runtime_parallel_workers(self):
        assert _pipeline(_config(workers=4)).workers == 4
        # Without the section or the key the default is 1, so a library call stays in the
        # caller's process.
        assert _pipeline({"grid": {"batch_sizes": [8], "total_gpus": [1]}}).workers == 1
        assert _pipeline({"grid": {"batch_sizes": [8], "total_gpus": [1]}, "runtime": {}}).workers == 1
        # Clamped to the core count (8 here).
        assert _pipeline(_config(workers=64)).workers == FAKE_CPU_COUNT

    def test_from_config_keeps_the_predictor_config_when_it_built_every_component(self):
        # Workers rebuild the checker from this config; without it no stage could fork.
        pipeline = GridWorkflowPipeline.from_config(
            config=_config(workers=4), selection_policy="performance", strategy_name="parallel-stages"
        )
        assert pipeline.predictor_config == {"feasibility": "rules"}

    @pytest.mark.parametrize(
        "injected",
        ["throughput_predictor", "power_predictor", "feasibility_checker"],
    )
    def test_an_injected_component_withholds_the_predictor_config_so_nothing_forks(self, injected):
        # A worker can rebuild only what the config names, so with a passed-in object a fork would
        # run a different one. The whole pipeline then stays in this process.
        component = _ExpensiveRulesChecker() if injected == "feasibility_checker" else _LinearPredictor()
        pipeline = _pipeline(_config(workers=4), **{injected: component})
        assert pipeline.predictor_config is None

    def test_one_and_two_workers_return_identical_recommendations(self, workload, context, map_calls):
        # The rules backend needs no AutoConf install and is cheap, so it also stays inline at
        # workers=2.
        one = _pipeline(_config(workers=1)).recommend(workload, context)
        two = _pipeline(_config(workers=2)).recommend(workload, context)

        assert len(one) == 8  # 2 batch sizes x 4 GPU counts, all feasible under rules
        assert _dump(two) == _dump(one)
        assert map_calls == [], "a cheap feasibility backend and a cheap predictor never fork"

    def test_a_forked_feasibility_stage_returns_the_sequential_recommendations(
        self, workload, context, map_calls, no_pool, monkeypatch
    ):
        # The rules backend is marked expensive so the stage chunks and gathers through the real
        # feasibility_worker, which rebuilds a rules checker from the config. The flag is set on
        # the class because an injected instance would withhold the predictor config and stop the
        # fork. Both runs build every component from the config (Kavier is cheap and
        # deterministic), so the fork is the only difference.
        def from_config(workers: int):
            config = _config(workers=workers)
            config["predictors"]["performance"] = "kavier"
            return GridWorkflowPipeline.from_config(
                config=config, selection_policy="performance", strategy_name="parallel-stages"
            )

        sequential = from_config(1).recommend(workload, context)
        monkeypatch.setattr(RulesFeasibilityChecker, "EXPENSIVE", True)

        forked = from_config(4).recommend(workload, context)

        assert _dump(forked) == _dump(sequential)
        assert len(map_calls) == 1 and map_calls[0]["stage"] == "feasibility"
        assert len(map_calls[0]["payloads"]) == 4
        # Ranking runs over the whole set and is never forked.
        assert [r.total_gpus for r in forked] == [r.total_gpus for r in sequential]

    @pytest.mark.parametrize("workers", [1, 2])
    def test_an_empty_grid_raises_the_normal_error_at_any_worker_count(self, workload, context, workers):
        # generate_candidates skips non-positive GPU counts, so both stages get an empty list and
        # the usual "no feasible candidates" RuntimeError is raised.
        pipeline = _pipeline(_config(workers=workers, total_gpus=(0,), batch_sizes=(8,)))

        with pytest.raises(RuntimeError, match="no feasible candidates found in grid of 0 configurations"):
            pipeline.recommend(workload, context)

    @pytest.mark.parametrize("workers", [1, 4])
    def test_every_candidate_infeasible_raises_at_any_worker_count(self, workload, context, workers):
        # Feasibility rejects every candidate, so simulation gets an empty list and the usual error
        # is raised. The double is cheap and runs inline, so it is the checker that decides (a
        # forked stage would rebuild its checker from the config).
        pipeline = _pipeline(
            _config(workers=workers),
            feasibility_checker=_RecordingChecker(reject={1, 2, 4, 8}),
        )

        with pytest.raises(RuntimeError, match="no feasible candidates found in grid of 8 configurations"):
            pipeline.recommend(workload, context)
