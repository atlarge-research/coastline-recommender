"""simulate_fifo keeps each result's own display metadata when two queued jobs share a request_id.

A client can set request_id through the API, and a CSV import can carry duplicate ids. The
scheduler keys results on position; keyed on request_id, duplicates would all get the last job's
metadata.
"""

from __future__ import annotations

from coastline.ui.workload_queue import QueueJob, simulate_fifo


def test_simulate_fifo_keeps_per_job_metadata_with_duplicate_request_ids() -> None:
    jobs = [
        QueueJob(request_id="x", arrival_time=0, num_gpus=2, predicted_duration_s=10, llm_model="A"),
        QueueJob(request_id="x", arrival_time=0, num_gpus=2, predicted_duration_s=10, llm_model="B"),
    ]
    result = simulate_fifo(jobs, n_gpus_cluster=8)
    # Both jobs are scheduled and each JobResult keeps its own model.
    assert sorted(j.llm_model for j in result.jobs) == ["A", "B"]
