"""simulate_fifo must keep each result's OWN display metadata, even when two queued jobs share a
request_id (a client can set request_id via the API, and a CSV import can carry duplicate ids).

Regression guard for the Kavier-delegation refactor: an earlier version keyed the display fields on
request_id, so duplicates collapsed onto the last job's metadata. The scheduler keys on position.
"""

from __future__ import annotations

from coastline.ui.workload_queue import QueueJob, simulate_fifo


def test_simulate_fifo_keeps_per_job_metadata_with_duplicate_request_ids() -> None:
    jobs = [
        QueueJob(request_id="x", arrival_time=0, num_gpus=2, predicted_duration_s=10, llm_model="A"),
        QueueJob(request_id="x", arrival_time=0, num_gpus=2, predicted_duration_s=10, llm_model="B"),
    ]
    result = simulate_fifo(jobs, n_gpus_cluster=8)
    # Both jobs are scheduled; each JobResult keeps its own model, not two copies of the last one.
    assert sorted(j.llm_model for j in result.jobs) == ["A", "B"]
