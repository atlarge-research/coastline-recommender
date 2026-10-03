"""Per-stage parallel execution over grid candidates.

The pipeline runs its stages in sequence. A stage may split its candidates across worker
processes and joins them before the next stage starts. Ranking (min-max over the feasible set,
then a sort) runs in one process, since splitting it would need a merge that costs more than
the sort.

Two stages fork: simulation when the predictor is a data-driven model, and feasibility when its
classifier cannot be batched. A batched classifier judges a whole chunk in one call, and forking
on top of that gains nothing.

Chunks are contiguous and results are joined in chunk order, so every candidate keeps its
sequential position. ``rank_candidates`` can fall back to grid order on ties, so a reordering
could change which config wins.

A stage forks only when its per-candidate work outweighs sending the candidate to a worker. The
AutoConf classifier does. The ``rules`` backend (two integer comparisons) and the analytical
Kavier predictor do not and always run in this process.

Workers are processes because threads gain nothing for the native predictors under the GIL, and
the ML backends cannot safely be loaded together in one interpreter.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable, Optional, Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Config key for the worker count (``runtime.parallel_workers`` in experiment.yaml).
RUNTIME_SECTION = "runtime"
WORKERS_KEY = "parallel_workers"

#: Default worker count of CLI commands. The SDK default is 1: a library call keeps the caller's
#: work in its process, where the caller's monkeypatches and globals apply.
DEFAULT_CLI_WORKERS = 4

# Fewest candidates for which a stage is split across processes. A stage that pays per candidate
# (an ML predictor, or AutoConf without batching) gains from forking after a few candidates. A
# stage batched into one call never gains, since each chunk pays the call's fixed cost again.
MIN_ITEMS_PER_ROW_STAGE = 8

# A new worker loads its model before it can answer. The pool is reused for the whole run, but the
# first stage to fork pays that start-up alone, so starting a pool needs a larger stage.
MIN_ITEMS_TO_START_A_POOL = 512


def resolve_workers(requested: Optional[int]) -> int:
    """Clamp a requested worker count to something this machine can honour (>= 1)."""
    if not requested or requested < 1:
        return 1
    return max(1, min(int(requested), os.cpu_count() or 1))


def pool_is_warm() -> bool:
    """Whether a pool already exists, so a stage would not pay to start one."""
    return _POOL is not None


def plan_chunks(n_items: int, workers: int, min_items: int = MIN_ITEMS_PER_ROW_STAGE) -> int:
    """Number of chunks to split ``n_items`` into; 1 means run in this process.

    Below ``min_items`` the dispatch costs more than the work. Before the pool exists the bar is
    MIN_ITEMS_TO_START_A_POOL, since this stage would also pay for every worker's model load.
    """
    if not pool_is_warm():
        min_items = max(min_items, MIN_ITEMS_TO_START_A_POOL)
    if workers <= 1 or n_items < min_items:
        return 1
    return max(1, min(workers, n_items // max(1, min_items // workers or 1)))


def chunk(items: Sequence[T], n_chunks: int) -> list[list[T]]:
    """Split into ``n_chunks`` contiguous, near-equal chunks, preserving order."""
    if n_chunks <= 1:
        return [list(items)]
    size, remainder = divmod(len(items), n_chunks)
    chunks: list[list[T]] = []
    start = 0
    for i in range(n_chunks):
        end = start + size + (1 if i < remainder else 0)
        chunks.append(list(items[start:end]))
        start = end
    return chunks


# --------------------------------------------------------------------------------------------
# The pool: one per process, shared by every job and stage.
#
# Each worker loads the AutoGluon feasibility model once. A pool per recommendation would pay that
# load for every job, which costs more than forking saves. The pool is replaced only when the
# worker count changes; otherwise the interpreter cleans it up at exit.
# --------------------------------------------------------------------------------------------


class WorkerPoolFailure(RuntimeError):
    """A stage failed in a worker and again in this process.

    Its own type, so the per-row error handling of ``batch_api.recommend`` and
    ``trace.recommend._recommend_row`` can tell a broken pool from a bad workload.
    """


_POOL: Optional[ProcessPoolExecutor] = None
_POOL_WORKERS = 0
# Guards the check, shutdown and create steps below. The pool is a module global used from
# FastAPI's threadpool as well as from a CLI run, so two callers could otherwise each create one,
# or one could shut down the pool the other is still using.
_POOL_LOCK = threading.Lock()
# Set when a pool dies. The failed stage is rerun in this process and nothing forks again, since
# a new pool would likely die the same way.
_FORKING_DISABLED = False


def get_pool(workers: int) -> Optional[ProcessPoolExecutor]:
    """The shared pool for ``workers`` processes, or None when running sequentially."""
    global _POOL, _POOL_WORKERS
    if workers <= 1 or _FORKING_DISABLED:
        return None
    with _POOL_LOCK:
        if _POOL is not None and _POOL_WORKERS == workers:
            return _POOL
        _shutdown_locked()
        logger.info("starting a pool of %d worker processes for per-stage parallelism", workers)
        _POOL = ProcessPoolExecutor(max_workers=workers)
        _POOL_WORKERS = workers
        return _POOL


def _shutdown_locked() -> None:
    global _POOL, _POOL_WORKERS
    if _POOL is not None:
        _POOL.shutdown(wait=False, cancel_futures=True)
    _POOL = None
    _POOL_WORKERS = 0


def shutdown_pool() -> None:
    """Tear the shared pool down (idempotent)."""
    with _POOL_LOCK:
        _shutdown_locked()


def _retire_pool(dead: ProcessPoolExecutor) -> None:
    """Drop a pool that died, without touching a replacement someone else may have installed."""
    global _POOL, _POOL_WORKERS, _FORKING_DISABLED
    with _POOL_LOCK:
        _FORKING_DISABLED = True
        if _POOL is dead:
            _POOL = None
            _POOL_WORKERS = 0
    dead.shutdown(wait=False, cancel_futures=True)


def map_chunks(
    worker: Callable[[Any], list[Any]],
    payloads: list[Any],
    workers: int,
    *,
    stage: str,
) -> list[Any]:
    """Run ``worker`` over ``payloads`` in the shared pool and concatenate results in order.

    ``worker`` must be a module-level function (the pool spawns, so it is pickled by name) taking
    one payload and returning a list. If a worker dies, the chunks are rerun in this process;
    WorkerPoolFailure is raised if that fails too.
    """
    pool = get_pool(workers)
    if pool is None:
        return [item for payload in payloads for item in worker(payload)]
    try:
        chunk_results = list(pool.map(worker, payloads))
    except BrokenProcessPool:
        # Redo the work here. Both callers catch Exception per row, so a raised error would be
        # reported as "this predictor could not handle this job". Running the same chunks in
        # this process gives the sequential result.
        _retire_pool(pool)
        logger.warning(
            "a worker process died during the %s stage; finishing this stage in-process and "
            "running sequentially from here on (results are unaffected)",
            stage,
        )
        try:
            return [item for payload in payloads for item in worker(payload)]
        except Exception as inline_exc:  # the work fails in this process too
            raise WorkerPoolFailure(
                f"the {stage} stage failed in a worker and again in-process: {inline_exc}"
            ) from inline_exc
    return [item for chunk_result in chunk_results for item in chunk_result]


# --------------------------------------------------------------------------------------------
# Worker-side state. A loaded AutoGluon model cannot be pickled, so a worker rebuilds the checker
# and predictors from the config it receives and keeps them for its lifetime, keyed by config.
# --------------------------------------------------------------------------------------------

_WORKER_CHECKERS: dict[str, Any] = {}
_WORKER_PREDICTORS: dict[str, Any] = {}


def _config_key(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, default=repr)


def worker_checker(predictor_config: dict[str, Any]) -> Any:
    """This worker's feasibility checker for ``predictor_config``, built once."""
    key = _config_key(predictor_config)
    checker = _WORKER_CHECKERS.get(key)
    if checker is None:
        from coastline.sdk.pipeline.feasibility import create_feasibility_checker

        checker = create_feasibility_checker(predictor_config)
        _WORKER_CHECKERS[key] = checker
    return checker


def worker_predictors(predictor_config: dict[str, Any]) -> tuple[Any, Any]:
    """This worker's (throughput, power) predictors for ``predictor_config``, built once."""
    key = _config_key(predictor_config)
    pair = _WORKER_PREDICTORS.get(key)
    if pair is None:
        from coastline.sdk.policies import PolicyFactory

        pair = (
            PolicyFactory.throughput_predictor(predictor_config),
            PolicyFactory.power_predictor(predictor_config),
        )
        _WORKER_PREDICTORS[key] = pair
    return pair


def _batches(checker: Any) -> bool:
    """Whether this checker decides a whole chunk in one call rather than one call per candidate."""
    can_batch = getattr(checker, "batches", None)
    if can_batch is not None:
        return bool(can_batch())
    return False


def feasibility_worker(payload: tuple[dict[str, Any], list[Any]]) -> list[Any]:
    """Pool entry point for the feasibility stage: one chunk of candidates in, verdicts out."""
    predictor_config, workloads = payload
    from coastline.sdk.pipeline.feasibility import evaluate_chunk

    return evaluate_chunk(worker_checker(predictor_config), workloads)


def run_feasibility(
    checker: Any,
    predictor_config: Optional[dict[str, Any]],
    workloads: Sequence[Any],
    workers: int,
) -> list[Any]:
    """Feasibility verdicts for every candidate, in order, forked when that pays off.

    Runs in this process for a cheap or batching checker, too few candidates, one worker, or no
    predictor config for the workers to rebuild the checker from.
    """
    from coastline.sdk.pipeline.feasibility import evaluate_chunk, is_expensive

    # A batching checker gained nothing from forking at any measured grid size (see
    # MIN_ITEMS_PER_ROW_STAGE), so it never forks; the same checker without batching does.
    worth_forking = bool(predictor_config) and is_expensive(checker) and not _batches(checker)
    n_chunks = plan_chunks(len(workloads), workers, MIN_ITEMS_PER_ROW_STAGE) if worth_forking else 1
    if n_chunks <= 1:
        return evaluate_chunk(checker, workloads)
    payloads = [(predictor_config, part) for part in chunk(workloads, n_chunks)]
    logger.debug("feasibility: %d candidates over %d workers", len(workloads), n_chunks)
    return map_chunks(feasibility_worker, payloads, workers, stage="feasibility")


def simulation_worker(payload: tuple[dict[str, Any], Any, list[Any]]) -> list[Any]:
    """Pool entry point for the simulation stage: one chunk of candidates in, predictions out."""
    predictor_config, context, workloads = payload
    from coastline.sdk.pipeline.workflow import simulate_chunk

    throughput_predictor, power_predictor = worker_predictors(predictor_config)
    return simulate_chunk(throughput_predictor, power_predictor, workloads, context)


def run_simulation(
    throughput_predictor: Any,
    power_predictor: Any,
    predictor_config: Optional[dict[str, Any]],
    workloads: Sequence[Any],
    context: Any,
    workers: int,
) -> list[Any]:
    """Predictions for every feasible candidate, in order, forked when that pays off.

    Only the data-driven models fork, since a prediction costs far more than its dispatch. Kavier
    and the cache cost far less than a dispatch and always run in this process.
    """
    from coastline.sdk.pipeline.workflow import simulate_chunk

    expensive = getattr(throughput_predictor, "EXPENSIVE", False) is True
    n_chunks = plan_chunks(len(workloads), workers, MIN_ITEMS_PER_ROW_STAGE) if (predictor_config and expensive) else 1
    if n_chunks <= 1:
        return simulate_chunk(throughput_predictor, power_predictor, list(workloads), context)
    payloads = [(predictor_config, context, part) for part in chunk(workloads, n_chunks)]
    logger.debug("simulation: %d candidates over %d workers", len(workloads), n_chunks)
    return map_chunks(simulation_worker, payloads, workers, stage="simulation")
