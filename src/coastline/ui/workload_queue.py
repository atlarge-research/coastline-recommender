"""In-memory workload queue + FIFO cluster simulation (the Exp2/Exp4 operational view).

The cluster **simulation** — FIFO scheduling of the queued jobs onto a fixed GPU cluster, with
per-job wait/runtime and per-cluster makespan/utilisation/energy — is done by Kavier
(``kavier.sdk.cluster.schedule``). This module owns the in-memory queue, CSV import, and the
adaptation of Kavier's result into the UI's SimulationResult/ClusterTimeline. Coastline consumes;
Kavier simulates.
"""

from __future__ import annotations

import csv
import io
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from kavier.sdk.cluster import schedule as cluster_schedule
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Fallback per-GPU draw for the energy summary when a job carries no Kavier per-GPU power. Passed to
# the simulator as its default; a job's own ``predicted_power_watts_per_gpu`` takes precedence.
_AVG_WATTS_PER_GPU = 350.0


class QueueJob(BaseModel):
    """Scheduler record: four fields are consumed by the FIFO sim; the rest is metadata."""

    request_id: str = Field(..., description="Stable identifier")
    arrival_time: float = Field(..., ge=0.0, description="Arrival timestamp (seconds, relative or epoch)")
    num_gpus: int = Field(..., ge=1, description="GPUs requested")
    predicted_duration_s: float = Field(..., gt=0.0, description="Predicted runtime in seconds")
    predicted_power_watts_per_gpu: Optional[float] = Field(
        default=None,
        gt=0.0,
        description=(
            "Per-GPU power (W) captured at add-time from Kavier when the workload config is known; "
            "the simulator falls back to a cluster-average constant when this is None"
        ),
    )
    llm_model: Optional[str] = Field(None, description="Display only — scheduler does not consume this")
    fine_tuning_method: Optional[str] = None
    gpu_model: Optional[str] = Field(
        None, description="Kavier input — only consumed by the import handler's power/duration lookup"
    )
    tokens_per_sample: Optional[int] = Field(default=None, gt=0, description="Kavier input")
    batch_size: Optional[int] = Field(None, description="Display + Kavier input")
    dataset_size: Optional[int] = Field(default=None, gt=0, description="Kavier-duration input")
    training_epochs: Optional[int] = Field(default=None, gt=0, description="Kavier-duration input")
    gpus_per_node: Optional[int] = Field(default=None, ge=1, description="Kavier-input layout hint")
    number_of_nodes: Optional[int] = Field(default=None, ge=1, description="Kavier-input layout hint")


# In-memory store (process-local singleton).
_lock = threading.Lock()
_jobs: List[QueueJob] = []
_id_counter: int = 0


def add_job(job: QueueJob) -> QueueJob:
    with _lock:
        _jobs.append(job)
    return job


def remove_job(request_id: str) -> bool:
    with _lock:
        for i, j in enumerate(_jobs):
            if j.request_id == request_id:
                del _jobs[i]
                return True
    return False


def list_jobs() -> List[QueueJob]:
    with _lock:
        return list(_jobs)


def clear_jobs() -> int:
    with _lock:
        n = len(_jobs)
        _jobs.clear()
    return n


def generate_id() -> str:
    """Monotonic 3-digit job IDs (001, 002, ...); process-local, resets on restart."""
    global _id_counter
    with _lock:
        _id_counter += 1
        return f"{_id_counter:03d}"


# Cluster simulation (delegated to kavier.sdk.cluster) + result adaptation.


@dataclass
class JobResult:
    request_id: str
    num_gpus: int
    predicted_duration_s: float
    arrival_time: float
    start_time: float
    end_time: float
    wait_time_s: float
    completion_time_s: float
    energy_kwh: float
    llm_model: Optional[str] = None
    fine_tuning_method: Optional[str] = None
    batch_size: Optional[int] = None
    training_epochs: Optional[int] = None


@dataclass
class SimulationResult:
    makespan_s: float
    avg_resource_occupation: float
    goodput_jobs_per_s: float
    avg_waiting_time_s: float
    avg_job_completion_time_s: float
    total_energy_kwh: float
    n_jobs: int
    jobs: List[JobResult]


@dataclass
class ClusterTimeline:
    """Step-series for the cluster figure (the Exp2/Exp4 plot): GPUs allocated
    and queue depth over time, both derived from a completed FIFO simulation.

    Times are normalised so ``t == 0`` is the first job arrival; the series runs
    to ``t == makespan_s`` where the cluster drains back to empty. Each triple
    ``(t[i], gpus_used[i], queue_depth[i])`` is the cluster state across the
    half-open interval ``[t[i], t[i + 1])`` — i.e. a step-after staircase, the
    natural shape for "allocated GPUs" and "jobs waiting", which only change at
    discrete arrival / start / end events.

    ``cluster_gpus`` is the capacity ceiling (the GPU chart's y-max and the
    dashed reference rule). ``peak_gpus`` / ``peak_queue`` are the high-water
    marks, handy for an at-a-glance caption."""

    t: List[float]
    gpus_used: List[int]
    queue_depth: List[int]
    cluster_gpus: int
    makespan_s: float
    peak_gpus: int
    peak_queue: int


def _empty_result() -> SimulationResult:
    """Zeroed simulation result for an empty (or fully-skipped) queue."""
    return SimulationResult(
        makespan_s=0.0,
        avg_resource_occupation=0.0,
        goodput_jobs_per_s=0.0,
        avg_waiting_time_s=0.0,
        avg_job_completion_time_s=0.0,
        total_energy_kwh=0.0,
        n_jobs=0,
        jobs=[],
    )


def simulate_fifo(jobs: List[QueueJob], n_gpus_cluster: int) -> SimulationResult:
    """FIFO simulation of a list of QueueJobs on an n-GPU cluster.

    The scheduling + per-cluster metrics + energy are computed by Kavier
    (``kavier.sdk.cluster.schedule``, strict-FIFO flat pool with head-of-line blocking); this
    function only adapts the inputs/outputs to the UI's models. Oversized jobs (``num_gpus >
    n_gpus_cluster``) are dropped — under strict FIFO they would block the queue head forever — so
    the behaviour matches the previous in-process scheduler.
    """
    if not jobs:
        return _empty_result()
    if n_gpus_cluster <= 0:
        raise ValueError(f"n_gpus_cluster must be > 0 (got {n_gpus_cluster})")

    # Key the jobs by POSITION, not request_id: request_ids are normally unique but a client can set
    # a duplicate (or a CSV import can carry one), and the scheduler must keep each result's own
    # display metadata rather than collapsing duplicates onto the last job's.
    rows = [
        {
            "job_id": i,
            "submit_s": j.arrival_time,
            "gpus": j.num_gpus,
            "duration_s": j.predicted_duration_s,
            "power_w_per_gpu": j.predicted_power_watts_per_gpu,
        }
        for i, j in enumerate(jobs)
    ]
    # kavier models the cluster as num_nodes x node_gpus; the UI's flat pool of
    # n_gpus_cluster GPUs is one node holding them all (any job up to the total fits).
    result = cluster_schedule(
        rows,
        policy="distributed-fcfs",
        num_nodes=1,
        node_gpus=n_gpus_cluster,
        oversized="drop",
        default_watts_per_gpu=_AVG_WATTS_PER_GPU,
    )
    if result.dropped:
        logger.warning(
            "simulate_fifo: skipping %d job(s) requiring more than %d GPUs",
            len(result.dropped),
            n_gpus_cluster,
        )
    if not result.jobs:
        return _empty_result()

    completed: List[JobResult] = []
    for record in result.jobs:
        source = jobs[record.job_id]  # record.job_id is the positional index passed in above
        completed.append(
            JobResult(
                request_id=source.request_id,
                num_gpus=record.gpus,
                predicted_duration_s=record.runtime_s,
                arrival_time=record.submit_s,
                start_time=record.start_s,
                end_time=record.end_s,
                wait_time_s=record.wait_s,
                completion_time_s=record.turnaround_s,  # job completion time = turnaround (end - arrival)
                energy_kwh=record.energy_kwh if record.energy_kwh is not None else 0.0,
                llm_model=source.llm_model,
                fine_tuning_method=source.fine_tuning_method,
                batch_size=source.batch_size,
                training_epochs=source.training_epochs,
            )
        )
    completed.sort(key=lambda r: (r.start_time, r.request_id))

    cluster = result.cluster
    return SimulationResult(
        makespan_s=cluster.makespan_s,
        avg_resource_occupation=cluster.utilization,
        goodput_jobs_per_s=cluster.goodput_jobs_per_s,
        avg_waiting_time_s=cluster.avg_wait_s,
        avg_job_completion_time_s=cluster.avg_turnaround_s,
        total_energy_kwh=cluster.total_energy_kwh if cluster.total_energy_kwh is not None else 0.0,
        n_jobs=cluster.n_jobs,
        jobs=completed,
    )


def build_cluster_timeline(jobs: List[JobResult], n_gpus_cluster: int) -> ClusterTimeline:
    """Turn a finished FIFO run into the two step-series the cluster figure draws:
    GPUs allocated over time and jobs-in-queue over time.

    Pure and side-effect-free — it reads only the four scheduler-relevant fields
    on each ``JobResult`` (arrival / start / end / num_gpus), so it is trivially
    unit-testable and never re-runs the simulation. Pass the ``jobs`` from a
    :class:`SimulationResult` (already FIFO-scheduled with start/end times set).

    The series are built by a linear sweep over arrival / start / end events, not
    an O(jobs × breakpoints) scan, so a few-thousand-job trace stays interactive:

    * a job adds ``num_gpus`` at its ``start_time`` and releases them at
      ``end_time`` → GPUs allocated;
    * a job joins the queue at ``arrival_time`` and leaves it at ``start_time``
      → queue depth.

    Each event time lands on exactly one breakpoint, and the cumulative value
    *after* applying every delta at that time is the state for the interval
    starting there. Times are normalised to the first arrival."""
    runnable = [j for j in jobs if j.num_gpus <= n_gpus_cluster]
    if not runnable:
        return ClusterTimeline(
            t=[],
            gpus_used=[],
            queue_depth=[],
            cluster_gpus=max(int(n_gpus_cluster), 0),
            makespan_s=0.0,
            peak_gpus=0,
            peak_queue=0,
        )

    t0 = min(j.arrival_time for j in runnable)

    # Signed deltas keyed by event time. Kavier's schedule uses exact times, so a release and the
    # matching dispatch coincide on one breakpoint; the capacity clamp in the sweep below is a cheap
    # defensive guard against any float sliver and never distorts a well-formed run.
    gpu_delta: Dict[float, int] = defaultdict(int)
    queue_delta: Dict[float, int] = defaultdict(int)
    for j in runnable:
        start = j.start_time - t0
        end = j.end_time - t0
        arrival = j.arrival_time - t0
        gpu_delta[start] += j.num_gpus
        gpu_delta[end] -= j.num_gpus
        queue_delta[arrival] += 1
        queue_delta[start] -= 1

    # Always anchor the series at t=0 (the first arrival) so the chart starts on
    # the axis even if nothing is running yet at that instant.
    breakpoints = sorted(set(gpu_delta) | set(queue_delta) | {0.0})

    cap = int(n_gpus_cluster)
    t_list: List[float] = []
    gpus_list: List[int] = []
    queue_list: List[int] = []
    cum_gpus = 0
    cum_queue = 0
    peak_gpus = 0
    peak_queue = 0
    for e in breakpoints:
        cum_gpus += gpu_delta.get(e, 0)
        cum_queue += queue_delta.get(e, 0)
        # Keep the running accumulator exact (so a later release still nets back
        # to zero) but never publish more than the cluster can hold: the only way
        # ``cum_gpus`` exceeds ``cap`` is the sub-eps double-count above, since
        # the FIFO scheduler itself never over-allocates.
        shown_gpus = min(cum_gpus, cap)
        t_list.append(e)
        gpus_list.append(shown_gpus)
        queue_list.append(cum_queue)
        peak_gpus = max(peak_gpus, shown_gpus)
        peak_queue = max(peak_queue, cum_queue)

    makespan_s = max(j.end_time - t0 for j in runnable)

    return ClusterTimeline(
        t=t_list,
        gpus_used=gpus_list,
        queue_depth=queue_list,
        cluster_gpus=int(n_gpus_cluster),
        makespan_s=makespan_s,
        peak_gpus=peak_gpus,
        peak_queue=peak_queue,
    )


# CSV import (multiple trace schemas tolerated via column-name aliases).


def parse_csv(text: str) -> List[QueueJob]:
    """Parse a workload-trace CSV into QueueJobs; accepts flexible column-name aliases.

    Required columns: arrival (submission_time / arrival_time), num_gpus, duration
    (duration_ms / duration_s / predicted_duration_s). Standard trace headers import straight through."""
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV has no header row")
    cols = {c.lower(): c for c in reader.fieldnames}

    def pick(*aliases: str) -> Optional[str]:
        for alias in aliases:
            key = alias.lower()
            if key in cols:
                return cols[key]
        return None

    col_arrival = pick("arrival_time", "submission_time", "arrival", "submit_time")
    col_gpus = pick("num_gpus", "number_gpus", "gpus")
    col_dur_s = pick("predicted_duration_s", "duration_s", "duration_seconds")
    col_dur_ms = pick("duration_ms")
    col_id = pick("request_id", "id", "job_id")
    col_model = pick("llm_model", "model_name", "model")
    col_method = pick("fine_tuning_method", "method")
    col_gpu_model = pick("gpu_model")
    col_tokens = pick("tokens_per_sample")
    col_batch = pick("batch_size", "batch")
    col_dataset = pick("dataset_size")
    col_epochs = pick("training_epochs", "num_train_epochs", "epochs")
    col_nodes = pick("number_of_nodes", "number_nodes", "num_nodes")
    col_gpn = pick("num_gpus_per_node", "gpus_per_node")
    col_power = pick("predicted_power_watts_per_gpu", "power_watts_per_gpu", "power_watts", "power")

    missing = []
    if not col_arrival:
        missing.append("arrival/submission_time")
    if not col_gpus:
        missing.append("num_gpus/number_gpus")
    if not (col_dur_s or col_dur_ms):
        missing.append("duration_s/duration_ms/predicted_duration_s")
    if missing:
        raise ValueError(f"CSV missing required column(s): {', '.join(missing)}")

    def _maybe_int(row: Dict[str, Any], col: Optional[str]) -> Optional[int]:
        """None on missing/blank/unparseable/non-positive cells (QueueJob fields are gt=0)."""
        if not col or not row.get(col):
            return None
        try:
            v = int(float(row[col]))
        except ValueError:
            return None
        return v if v > 0 else None

    def _maybe_float_positive(row: Dict[str, Any], col: Optional[str]) -> Optional[float]:
        if not col or not row.get(col):
            return None
        try:
            v = float(row[col])
            return v if v > 0 else None
        except ValueError:
            return None

    jobs: List[QueueJob] = []
    for row in reader:
        # Skip rows with non-positive / unparseable duration (jobs that never completed).
        try:
            if col_dur_s and row.get(col_dur_s):
                duration = float(row[col_dur_s])
            elif col_dur_ms and row.get(col_dur_ms):
                duration = float(row[col_dur_ms]) / 1000.0
            else:
                continue
        except ValueError:
            continue
        if duration <= 0:
            continue  # silently skip impossibly-short rows

        # Drop rows with missing/unparseable arrival or GPU count rather than failing the whole import.
        try:
            arrival_time = float(row[col_arrival]) if row.get(col_arrival) else None
        except ValueError:
            arrival_time = None
        if arrival_time is None:
            continue
        try:
            num_gpus = int(float(row[col_gpus])) if row.get(col_gpus) else 0
        except ValueError:
            num_gpus = 0
        if num_gpus < 1:
            continue

        jobs.append(
            QueueJob(
                request_id=str(row[col_id]) if col_id and row.get(col_id) else generate_id(),
                arrival_time=arrival_time,
                num_gpus=num_gpus,
                predicted_duration_s=duration,
                predicted_power_watts_per_gpu=_maybe_float_positive(row, col_power),
                llm_model=str(row[col_model]) if col_model and row.get(col_model) else None,
                fine_tuning_method=str(row[col_method]) if col_method and row.get(col_method) else None,
                gpu_model=str(row[col_gpu_model]) if col_gpu_model and row.get(col_gpu_model) else None,
                tokens_per_sample=_maybe_int(row, col_tokens),
                batch_size=_maybe_int(row, col_batch),
                dataset_size=_maybe_int(row, col_dataset),
                training_epochs=_maybe_int(row, col_epochs),
                number_of_nodes=_maybe_int(row, col_nodes),
                gpus_per_node=_maybe_int(row, col_gpn),
            )
        )
    return jobs
