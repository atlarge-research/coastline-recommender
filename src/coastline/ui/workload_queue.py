"""In-memory workload queue and FIFO cluster simulation (the Exp2/Exp4 operational view).

Kavier (``kavier.sdk.cluster.schedule``) runs the simulation: FIFO scheduling of the queued jobs
on a fixed GPU cluster, with per-job wait and runtime and the cluster's makespan, utilisation and
energy. This module holds the queue, the CSV import and the conversion of Kavier's result into
the UI's SimulationResult and ClusterTimeline.
"""

from __future__ import annotations

import csv
import io
import logging
import math
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from kavier.sdk.cluster import schedule as cluster_schedule
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Per-GPU power [W] for the energy summary when a job has none; passed to the simulator as its
# default. A job's own predicted_power_watts_per_gpu takes precedence.
_AVG_WATTS_PER_GPU = 350.0


class QueueJob(BaseModel):
    """Queue entry: the FIFO simulation reads four fields; the rest is metadata."""

    request_id: str = Field(..., description="Stable identifier")
    arrival_time: float = Field(
        ..., ge=0.0, allow_inf_nan=False, description="Arrival timestamp (seconds, relative or epoch)"
    )
    num_gpus: int = Field(..., ge=1, description="GPUs requested")
    predicted_duration_s: float = Field(..., gt=0.0, allow_inf_nan=False, description="Predicted runtime in seconds")
    predicted_power_watts_per_gpu: Optional[float] = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "Per-GPU power (W) captured at add-time from Kavier when the workload config is known; "
            "the simulator falls back to a cluster-average constant when this is None"
        ),
    )
    llm_model: Optional[str] = Field(None, description="Display only - scheduler does not consume this")
    fine_tuning_method: Optional[str] = None
    gpu_model: Optional[str] = Field(
        None, description="Kavier input - only consumed by the import handler's power/duration lookup"
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


def next_arrival_time() -> float:
    """Arrival time for a job added without one: the latest arrival already queued, else 0.

    Keeps the job on the queue's own time base (CSV imports carry relative offsets) and puts it
    after the queued jobs in FIFO order. The earliest row of a CSV with ISO timestamps arrives
    here too.
    """
    with _lock:
        return max((j.arrival_time for j in _jobs), default=0.0)


def generate_id() -> str:
    """Monotonic 3-digit job IDs (001, 002, ...); process-local, resets on restart."""
    global _id_counter
    with _lock:
        _id_counter += 1
        return f"{_id_counter:03d}"


# Cluster simulation (in kavier.sdk.cluster) and conversion of its result.


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
    """GPUs allocated and queue depth over time from a finished FIFO simulation (the Exp2/Exp4
    cluster plot).

    ``t == 0`` is the first arrival and the series ends at ``t == makespan_s``, when the cluster
    is empty again. ``(t[i], gpus_used[i], queue_depth[i])`` holds over ``[t[i], t[i + 1])``, a
    step-after series, since both values change only at arrival, start and end events.
    ``cluster_gpus`` is the capacity (the y-max and dashed line of the GPU chart);
    ``peak_gpus`` and ``peak_queue`` are the maxima.
    """

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
    """FIFO simulation of QueueJobs on a cluster of ``n_gpus_cluster`` GPUs.

    Kavier (``kavier.sdk.cluster.schedule``) computes the schedule, the cluster metrics and the
    energy, as a strict-FIFO flat pool with head-of-line blocking; this function converts the
    inputs and outputs. Jobs with ``num_gpus > n_gpus_cluster`` are dropped, since under strict
    FIFO they would block the head of the queue forever.
    """
    if not jobs:
        return _empty_result()
    if n_gpus_cluster <= 0:
        raise ValueError(f"n_gpus_cluster must be > 0 (got {n_gpus_cluster})")

    # Jobs are keyed by position: a client or a CSV import can repeat a request_id, and each
    # result keeps the metadata of its own job.
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
    """Build the GPUs-allocated and queue-depth step series from a finished FIFO run.

    Pass the ``jobs`` of a :class:`SimulationResult`; only arrival, start, end and num_gpus are
    read. One sweep over the events builds both series: a job holds ``num_gpus`` from
    ``start_time`` to ``end_time`` and waits in the queue from ``arrival_time`` to
    ``start_time``. The value after all deltas at a time is the state from that time on.
    Times start at the first arrival.
    """
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

    # Signed deltas keyed by event time. Kavier's times are exact, so a release and the start it
    # allows share one breakpoint; the clamp in the sweep below only guards against float error.
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

    # Start the series at t=0 (the first arrival), even when nothing runs yet at that time.
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
        # Clamp only the shown value, so later releases still bring cum_gpus back to zero.
        # cum_gpus can exceed cap only through float error; the scheduler never over-allocates.
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


# CSV import; column-name aliases cover several trace schemas.


def _number(cell: Optional[str]) -> Optional[float]:
    """A cell as a float, or None when it is blank or not a number."""
    if not cell:
        return None
    try:
        return float(cell)
    except ValueError:
        return None


def _timestamp(cell: Optional[str]) -> Optional[datetime]:
    """A cell as an ISO 8601 timestamp (one without a zone is UTC), or None when it is not one."""
    if not cell:
        return None
    try:
        stamp = datetime.fromisoformat(cell.strip())
    except ValueError:
        return None
    return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=timezone.utc)


def _arrival_seconds(cells: List[Optional[str]], iso_start: float = 0.0) -> List[Optional[float]]:
    """The arrival of each row in seconds, None where a cell is blank or unreadable.

    Numbers are kept as they are. In a column with no number but ISO timestamps, the earliest
    timestamp arrives at ``iso_start`` and the others keep their spacing from it.
    """
    numbers = [_number(cell) for cell in cells]
    if any(number is not None for number in numbers):
        return numbers
    stamps = [_timestamp(cell) for cell in cells]
    known = [stamp for stamp in stamps if stamp is not None]
    if not known:
        return numbers
    start = min(known)
    return [None if stamp is None else iso_start + (stamp - start).total_seconds() for stamp in stamps]


def parse_csv(text: str, iso_start: float = 0.0) -> List[QueueJob]:
    """Parse a workload-trace CSV into QueueJobs, accepting several names per column.

    Required: arrival (submission_time or arrival_time), num_gpus, and duration (duration_ms,
    duration_s or predicted_duration_s). Arrivals are seconds, kept as given, or ISO timestamps:
    the earliest arrives at ``iso_start`` (the import passes the queue's next arrival time) and the
    others keep their spacing from it.
    """
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
        except (ValueError, OverflowError):  # OverflowError: an infinite cell
            return None
        return v if v > 0 else None

    def _maybe_float_positive(row: Dict[str, Any], col: Optional[str]) -> Optional[float]:
        if not col or not row.get(col):
            return None
        try:
            v = float(row[col])
            return v if math.isfinite(v) and v > 0 else None
        except ValueError:
            return None

    rows = list(reader)
    arrivals = _arrival_seconds([row.get(col_arrival) for row in rows], iso_start)
    jobs: List[QueueJob] = []
    for row, arrival_time in zip(rows, arrivals):
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
            continue  # skip rows without a positive duration

        # Drop rows with missing/unparseable arrival or GPU count rather than failing the whole import.
        if arrival_time is None:
            continue
        try:
            num_gpus = int(float(row[col_gpus])) if row.get(col_gpus) else 0
        except (ValueError, OverflowError):  # OverflowError: an infinite cell
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
