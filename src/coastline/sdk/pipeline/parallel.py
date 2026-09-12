"""Per-stage parallel execution over grid candidates.

The pipeline runs its stages one after another with a barrier between them: feasibility forks
its candidates across worker processes and joins, then simulation forks the survivors and joins.
Ranking is a whole-set reduction (min-max over the feasible set, then a sort) and stays
sequential — splitting it would need a merge costing more than the sort it replaces.

Two properties are load-bearing:

* **Order.** Chunks are contiguous and results are concatenated in chunk order, so a candidate's
  position is identical to the sequential path. ``rank_candidates`` breaks exact score ties by
  grid insertion order, so any reordering would silently change which config wins.
* **Worth it.** A stage forks only when its per-candidate work dominates the cost of shipping the
  candidate to a worker. The AutoConf feasibility classifier (~3.3 ms/candidate) qualifies; the
  divisibility ``rules`` backend and the analytical Kavier predictor (~2.6 us) do not — for those
  the dispatch would cost more than the work, so they always run inline.

Processes, not threads: the GIL makes threads useless for the native predictors, and the ML
backends are not safe to co-load in one interpreter anyway.
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

#: The CLI's default. The SDK default is 1: a library call must not silently move a caller's
#: work into subprocesses, where monkeypatched modules and in-process globals do not follow.
DEFAULT_CLI_WORKERS = 4

# How many candidates a stage needs before splitting it across processes wins. These are two
# very different regimes, both measured on this machine:
#
# * A stage that costs per CANDIDATE -- the ML predictors (2-77 ms each), or the AutoConf
#   classifier when batching is unavailable (~4.9 ms each) -- pays for a fork almost immediately:
#   a handful of candidates already outweighs the ~1 ms of pickling a chunk.
# * A stage that has already been BATCHED into one vectorised call costs ~4.5 ms fixed plus
#   ~0.2 ms per candidate (the per-row rule classifier and frame build, which cannot batch).
#   Splitting it pays k times that fixed cost, so it only wins once the linear part dominates.
#   Measured on a 405-row trace with an 18-candidate grid: 16.1 s sequential against 20.4 s at
#   2 workers and 24.8 s at 8 -- forking a batched stage this small is pure loss.
MIN_ITEMS_PER_ROW_STAGE = 8
MIN_ITEMS_BATCHED_STAGE = 256


def resolve_workers(requested: Optional[int]) -> int:
    """Clamp a requested worker count to something this machine can honour (>= 1)."""
    if not requested or requested < 1:
        return 1
    return max(1, min(int(requested), os.cpu_count() or 1))


def plan_chunks(n_items: int, workers: int, min_items: int = MIN_ITEMS_PER_ROW_STAGE) -> int:
    """How many chunks to split ``n_items`` into — 1 means "run inline".

    ``min_items`` is the point where forking starts to pay for this stage; below it the dispatch
    costs more than the work, so the stage runs in the calling process.
    """
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
# The pool: one per process, reused across every job and every stage.
#
# Each worker re-pays the AutoGluon feasibility model load (~1.65 s) once. A pool created per
# recommendation would re-pay it per job, which on a 400-row trace costs far more than the fork
# saves — so the pool is a module-level singleton, replaced only when the worker count changes
# and otherwise left to the interpreter to reap.
# --------------------------------------------------------------------------------------------

class WorkerPoolFailure(RuntimeError):
    """A worker process died. Its own type so the per-row isolation layers can let it through.

    ``batch_api.recommend`` and ``trace.recommend._recommend_row`` both wrap a row in
    ``except Exception`` to stop one bad workload sinking a batch. A dead pool is not a bad
    workload, and laundering it into "this predictor could not handle this job" would write a
    plausible CSV that quietly is not the answer.
    """


_POOL: Optional[ProcessPoolExecutor] = None
_POOL_WORKERS = 0
# Guards the check-shutdown-create sequence below. The pool is a module global reached from
# FastAPI's threadpool as well as from a plain CLI run, so two callers could otherwise each
# create one, or one could tear down the pool the other is still feeding.
_POOL_LOCK = threading.Lock()
# Set when a pool dies: the stage that died is recovered inline, and nothing forks again in this
# process. Re-creating a pool that has just died tends to buy the same death at more cost.
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
    one payload and returning a list. A dead worker is a hard failure: the caller is told to re-run
    sequentially rather than silently getting a different answer.
    """
    pool = get_pool(workers)
    if pool is None:
        return [item for payload in payloads for item in worker(payload)]
    try:
        chunk_results = list(pool.map(worker, payloads))
    except BrokenProcessPool:
        # Recover by doing the work here. Raising instead would be honest but useless: both
        # callers wrap a row in `except Exception`, so the failure would be laundered into
        # "this predictor could not handle this job" and the run would finish with a plausible
        # CSV that is not the answer. Running the same chunks in this process gives exactly the
        # sequential result, which is the guarantee the fork exists to preserve.
        _retire_pool(pool)
        logger.warning(
            "a worker process died during the %s stage; finishing this stage in-process and "
            "running sequentially from here on (results are unaffected)",
            stage,
        )
        try:
            return [item for payload in payloads for item in worker(payload)]
        except Exception as inline_exc:  # the work itself is broken, not just the pool
            raise WorkerPoolFailure(
                f"the {stage} stage failed in a worker and again in-process: {inline_exc}"
            ) from inline_exc
    return [item for chunk_result in chunk_results for item in chunk_result]


# --------------------------------------------------------------------------------------------
# Worker-side state. A spawned worker cannot receive the checker or the predictors themselves
# (a loaded AutoGluon model is not picklable), so it rebuilds them from the config it is sent and
# keeps them for the life of the process, keyed by that config.
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
    """Feasibility verdicts for every candidate, in order, forked when that is worth it.

    Falls back to the inline path whenever forking cannot pay: a cheap backend, too few
    candidates, one worker, or no predictor config to rebuild the checker from in the worker.
    """
    from coastline.sdk.pipeline.feasibility import evaluate_chunk, is_expensive

    # A checker that batches has already collapsed its per-candidate cost, so it needs a much
    # bigger grid before splitting it is worth k batched calls instead of one.
    min_items = MIN_ITEMS_BATCHED_STAGE if _batches(checker) else MIN_ITEMS_PER_ROW_STAGE
    n_chunks = (
        plan_chunks(len(workloads), workers, min_items) if (predictor_config and is_expensive(checker)) else 1
    )
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
    """Predictions for every feasible candidate, in order, forked when that is worth it.

    Only the data-driven models earn a fork: at 2-77 ms per prediction the dispatch is noise
    beside the work. The analytical Kavier predictor (2.6 us) and the cache (4.6 us) are orders
    of magnitude cheaper than the dispatch itself, so they always run inline.
    """
    from coastline.sdk.pipeline.workflow import simulate_chunk

    expensive = getattr(throughput_predictor, "EXPENSIVE", False) is True
    n_chunks = (
        plan_chunks(len(workloads), workers, MIN_ITEMS_PER_ROW_STAGE) if (predictor_config and expensive) else 1
    )
    if n_chunks <= 1:
        return simulate_chunk(throughput_predictor, power_predictor, list(workloads), context)
    payloads = [(predictor_config, context, part) for part in chunk(workloads, n_chunks)]
    logger.debug("simulation: %d candidates over %d workers", len(workloads), n_chunks)
    return map_chunks(simulation_worker, payloads, workers, stage="simulation")
