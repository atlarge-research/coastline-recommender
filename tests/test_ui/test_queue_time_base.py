"""The workload queue keeps one time base and refuses numbers the scheduler cannot use.

A job added from the dashboard without an arrival time arrives with the latest job already queued
(at 0 in an empty queue), so it shares the relative time base of CSV imports and keeps its place
in FIFO order. Infinite or NaN durations, arrivals and powers are rejected when added or imported,
so the queue never holds a job that breaks GET /api/queue or POST /api/admin/run.

Expected run, on the 32-GPU repo cluster: CSV jobs j1 (t=0) and j2 (t=60 s), 8 GPUs and 1 h
each, plus a dashboard job of 8 GPUs and 1 h. All three fit at once, so the makespan is
60 s + 3600 s = 3660 s.

A CSV whose arrivals are ISO timestamps keeps their spacing, and its earliest timestamp arrives
with the latest job already queued (at 0 in an empty queue), as a dashboard job does. A later
import therefore queues after the jobs already there.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import coastline.ui.app as app_module
from coastline.ui import workload_queue
from coastline.ui.workload_queue import parse_csv

_CSV = "request_id,arrival_time,num_gpus,duration_s\nj1,0,8,3600\nj2,60,8,3600\n"


@pytest.fixture
def client():
    workload_queue.clear_jobs()
    with TestClient(app_module.app) as test_client:
        yield test_client
    workload_queue.clear_jobs()


def test_a_dashboard_job_joins_the_time_base_of_imported_jobs(client) -> None:
    assert client.post("/api/admin/import", json={"csv": _CSV}).json()["imported"] == 2

    added = client.post("/api/queue", json={"num_gpus": 8, "predicted_duration_s": 3600}).json()["job"]
    assert added["arrival_time"] == 60.0  # the arrival of the latest queued job (j2)

    run = client.post("/api/admin/run").json()
    assert run["totals"]["makespan_s"] == pytest.approx(3660.0)
    # FIFO keeps the add order: j1, then j2, then the dashboard job.
    starts = {job["request_id"]: job["start_time"] for job in run["jobs"]}
    assert starts["j1"] <= starts["j2"] <= starts[added["request_id"]]


def test_a_job_added_to_an_empty_queue_arrives_at_zero(client) -> None:
    job = client.post("/api/queue", json={"num_gpus": 1, "predicted_duration_s": 60}).json()["job"]

    assert job["arrival_time"] == 0.0


def test_an_explicit_arrival_time_is_kept(client) -> None:
    job = client.post("/api/queue", json={"num_gpus": 1, "predicted_duration_s": 60, "arrival_time": 42.5}).json()

    assert job["job"]["arrival_time"] == 42.5


@pytest.mark.parametrize("field", ["predicted_duration_s", "arrival_time", "predicted_power_watts_per_gpu"])
@pytest.mark.parametrize("value", ["1e999", "NaN"])
def test_a_non_finite_number_is_rejected_when_added(client, field, value) -> None:
    body = {"num_gpus": 1, "predicted_duration_s": 60, field: None}
    raw = str(body).replace("'", '"').replace("None", value)

    resp = client.post("/api/queue", content=raw, headers={"content-type": "application/json"})

    assert resp.status_code == 422
    assert client.get("/api/queue").json()["jobs"] == []


@pytest.mark.parametrize(
    "row",
    ["j1,0,8,inf", "j1,inf,8,3600", "j1,0,8,nan"],
    ids=["infinite duration", "infinite arrival", "NaN duration"],
)
def test_a_non_finite_number_is_rejected_when_imported(client, row) -> None:
    resp = client.post("/api/admin/import", json={"csv": f"request_id,arrival_time,num_gpus,duration_s\n{row}\n"})

    assert resp.status_code == 400
    assert client.get("/api/queue").status_code == 200
    assert client.get("/api/queue").json()["jobs"] == []


def test_non_finite_optional_cells_are_dropped_and_the_queue_still_runs(client) -> None:
    csv_text = (
        "request_id,arrival_time,num_gpus,duration_s,power_watts_per_gpu,tokens_per_sample\nj1,0,8,3600,inf,inf\n"
    )
    resp = client.post("/api/admin/import", json={"csv": csv_text})

    assert resp.status_code == 200 and resp.json()["imported"] == 1
    (job,) = client.get("/api/queue").json()["jobs"]
    assert job["predicted_power_watts_per_gpu"] is None
    assert job["tokens_per_sample"] is None
    assert client.post("/api/admin/run").status_code == 200


def test_an_infinite_gpu_count_skips_the_row(client) -> None:
    csv_text = "request_id,arrival_time,num_gpus,duration_s\nj1,0,inf,3600\nj2,0,8,3600\n"
    resp = client.post("/api/admin/import", json={"csv": csv_text})

    assert resp.status_code == 200
    assert [job["request_id"] for job in client.get("/api/queue").json()["jobs"]] == ["j2"]


_ISO_CSV = (
    "request_id,submission_time,num_gpus,duration_s\nj1,2026-03-02T09:00:00Z,8,3600\nj2,2026-03-02T09:01:00Z,8,3600\n"
)


def test_iso_timestamp_arrivals_are_imported_on_the_queue_time_base(client) -> None:
    resp = client.post("/api/admin/import", json={"csv": _ISO_CSV})

    assert resp.status_code == 200 and resp.json()["imported"] == 2
    arrivals = {job["request_id"]: job["arrival_time"] for job in client.get("/api/queue").json()["jobs"]}
    assert arrivals == {"j1": 0.0, "j2": 60.0}
    added = client.post("/api/queue", json={"num_gpus": 8, "predicted_duration_s": 3600}).json()["job"]
    assert added["arrival_time"] == 60.0
    assert client.post("/api/admin/run").json()["totals"]["makespan_s"] == pytest.approx(3660.0)


def test_a_later_iso_import_queues_after_the_jobs_already_queued(client) -> None:
    """A second ISO file starts at the latest queued arrival (60 s) and keeps its own spacing.

    On the 32-GPU cluster every job takes all 32 GPUs for 1 h, so they run one after another in
    arrival order: d1a 0-3600 s, d1b 3600-7200 s, d2a 7200-10800 s, d2b 10800-14400 s.
    """
    header = "request_id,submission_time,num_gpus,duration_s\n"
    day1 = header + "d1a,2026-03-02T09:00:00Z,32,3600\nd1b,2026-03-02T09:01:00Z,32,3600\n"
    day2 = header + "d2a,2026-03-03T09:00:00Z,32,3600\nd2b,2026-03-03T09:02:00Z,32,3600\n"
    assert client.post("/api/admin/import", json={"csv": day1}).json()["imported"] == 2
    assert client.post("/api/admin/import", json={"csv": day2}).json()["imported"] == 2

    arrivals = {job["request_id"]: job["arrival_time"] for job in client.get("/api/queue").json()["jobs"]}
    assert arrivals == {"d1a": 0.0, "d1b": 60.0, "d2a": 60.0, "d2b": 180.0}
    run = client.post("/api/admin/run").json()
    starts = {job["request_id"]: job["start_time"] for job in run["jobs"]}
    assert starts == {"d1a": 0.0, "d1b": 3600.0, "d2a": 7200.0, "d2b": 10800.0}
    assert run["totals"]["makespan_s"] == pytest.approx(14400.0)


def test_an_iso_import_after_a_dashboard_job_starts_at_its_arrival(client) -> None:
    client.post("/api/queue", json={"num_gpus": 1, "predicted_duration_s": 60, "arrival_time": 500.0})

    assert client.post("/api/admin/import", json={"csv": _ISO_CSV}).json()["imported"] == 2
    arrivals = {job["request_id"]: job["arrival_time"] for job in client.get("/api/queue").json()["jobs"]}
    assert (arrivals["j1"], arrivals["j2"]) == (500.0, 560.0)


def test_iso_arrivals_with_zone_offsets_and_without_a_zone() -> None:
    # 10:00+01:00 is 09:00 UTC; a timestamp without a zone is read as UTC.
    jobs = parse_csv(
        "submission_time,num_gpus,duration_s\n"
        "2026-03-02T10:00:00+01:00,1,60\n"
        "2026-03-02 09:00:30,1,60\n"
        "2026-03-02T09:02:00Z,1,60\n"
    )

    assert [job.arrival_time for job in jobs] == [0.0, 30.0, 120.0]


def test_an_unreadable_arrival_still_skips_only_its_row() -> None:
    jobs = parse_csv(
        "request_id,submission_time,num_gpus,duration_s\n"
        "j1,2026-03-02T09:00:00Z,1,60\n"
        "j2,not a time,1,60\n"
        "j3,,1,60\n"
        "j4,2026-03-02T09:00:10Z,1,60\n"
    )

    assert [(job.request_id, job.arrival_time) for job in jobs] == [("j1", 0.0), ("j4", 10.0)]


def test_numeric_arrivals_are_kept_as_given() -> None:
    jobs = parse_csv("submission_time,num_gpus,duration_s\n100,1,60\n160,1,60\n")

    assert [job.arrival_time for job in jobs] == [100.0, 160.0]
