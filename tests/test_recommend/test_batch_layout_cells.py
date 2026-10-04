"""Integer cells of a batch row, and the job's layout (gpus_per_node, number_of_nodes).

Only min_gpu reads the layout, for the job's total batch, so a layout cell cannot fail a row under
the weighted goals. A whole number written as a float, such as '8.0' from a pandas CSV export of a
column with blanks, is read as an int. A blank cell is missing. 0, a negative or a fraction fails
the row with a message that names the column.
"""

from __future__ import annotations

import csv
import io
from typing import Any

import pandas as pd
import pytest

import coastline
import coastline.sdk.policies as policies
from coastline.sdk.models.workload import WorkloadSpec

GPU = "NVIDIA-A100-SXM4-80GB"
JOB = {
    "llm_model": "mistral-7b-v0.1",
    "fine_tuning_method": "lora",
    "gpu_model": GPU,
    "tokens_per_sample": 1024,
    "batch_size": 4,
}
WEIGHTED = ("performance", "balanced", "energy")


class FeasibleFrom4:
    """Feasible from 4 GPUs on; records the GPU counts and batch sizes it checked."""

    def __init__(self) -> None:
        self.checked: list[tuple[int, int]] = []

    def is_feasible(self, workload: WorkloadSpec) -> tuple[bool, dict[str, Any]]:
        self.checked.append((workload.total_gpus, workload.batch_size))
        return workload.total_gpus >= 4, {}


@pytest.fixture
def checker(monkeypatch) -> FeasibleFrom4:
    """Every min_gpu strategy built through PolicyFactory gets this checker."""
    fake = FeasibleFrom4()
    monkeypatch.setattr(policies, "create_feasibility_checker", lambda predictor_config: fake)
    return fake


def _text_rows(*rows: dict[str, Any]) -> list[dict[str, str]]:
    """The rows as csv.DictReader reads them back from a CSV file: every cell is text."""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return list(csv.DictReader(io.StringIO(buffer.getvalue())))


def _recommend(rows: list[dict[str, Any]], **kw: Any) -> pd.DataFrame:
    kw.setdefault("predictor", "kavier")
    kw.setdefault("feasibility", "rules")
    kw.setdefault("max_gpus", 8)
    return coastline.recommend(rows, **kw)


def _pick(row: pd.Series) -> tuple[int, int]:
    return int(row["total_gpus"]), int(row["recommended_batch_size"])


@pytest.mark.parametrize("goal", WEIGHTED)
@pytest.mark.parametrize(
    "layout",
    [
        {"gpus_per_node": "8.0", "number_of_nodes": "1.0"},
        {"gpus_per_node": "0", "number_of_nodes": "-1"},
        {"gpus_per_node": "eight", "number_of_nodes": "1.5"},
    ],
)
def test_the_weighted_goals_ignore_the_layout_cells(goal, layout):
    plain = _recommend(_text_rows(JOB), goal=goal).iloc[0]
    with_layout = _recommend(_text_rows({**JOB, **layout}), goal=goal).iloc[0]

    assert bool(with_layout["feasible"]), with_layout["error"]
    assert _pick(with_layout) == _pick(plain)


def test_min_gpu_reads_a_layout_written_as_floats(checker):
    out = _recommend(_text_rows({**JOB, "gpus_per_node": "8.0", "number_of_nodes": "1.0"}), goal="min_gpu", max_gpus=16)
    row = out.iloc[0]

    assert bool(row["feasible"]), row["error"]
    # 4 per device on 8 GPUs is a total batch of 32.
    assert checker.checked[:3] == [(1, 32), (2, 16), (4, 8)]
    assert _pick(row) == (4, 8)


def test_min_gpu_reads_float_cells_of_a_dataframe(checker):
    # A column with a blank becomes float in a DataFrame, so 8 is 8.0 and the blank is NaN.
    jobs = pd.DataFrame([{**JOB, "gpus_per_node": 8, "number_of_nodes": 1}, {**JOB, "number_of_nodes": 1}])
    assert jobs["gpus_per_node"].dtype == float
    out = _recommend(jobs, goal="min_gpu", max_gpus=16)

    assert out["feasible"].tolist() == [True, True], out["error"].tolist()
    assert _pick(out.iloc[0]) == (4, 8)  # total batch 32
    assert _pick(out.iloc[1]) == (4, 1)  # 1 GPU per node: total batch 4


@pytest.mark.parametrize("blank", ["", " "])
def test_min_gpu_takes_a_blank_layout_cell_as_missing(checker, blank):
    out = _recommend(_text_rows({**JOB, "gpus_per_node": blank, "number_of_nodes": blank}), goal="min_gpu", max_gpus=16)
    row = out.iloc[0]

    assert bool(row["feasible"]), row["error"]
    # No layout: a 1-GPU job with a total batch of 4.
    assert checker.checked[:3] == [(1, 4), (2, 2), (4, 1)]
    assert _pick(row) == (4, 1)


@pytest.mark.parametrize(
    "cells,error",
    [
        ({"gpus_per_node": "0"}, "gpus_per_node must be >= 1, got '0'"),
        ({"gpus_per_node": "-8"}, "gpus_per_node must be >= 1, got '-8'"),
        ({"number_of_nodes": "0.0"}, "number_of_nodes must be >= 1, got '0.0'"),
        ({"gpus_per_node": "8.5"}, "gpus_per_node must be a whole number, got '8.5'"),
        ({"number_of_nodes": "two"}, "number_of_nodes must be a whole number, got 'two'"),
    ],
)
def test_min_gpu_fails_a_row_with_a_bad_layout_cell(checker, cells, error):
    out = _recommend(_text_rows({**JOB, **cells}, JOB), goal="min_gpu", max_gpus=16)

    assert not bool(out.iloc[0]["feasible"])
    assert out.iloc[0]["error"] == error
    # The other row still runs.
    assert bool(out.iloc[1]["feasible"]), out.iloc[1]["error"]


def test_whole_floats_in_the_other_integer_columns_are_ints():
    as_ints = _recommend(_text_rows({**JOB, "batch_size": 16, "max_gpus": 4})).iloc[0]
    as_floats = _recommend(
        _text_rows({**JOB, "batch_size": "16.0", "tokens_per_sample": "1024.0", "max_gpus": "4.0"})
    ).iloc[0]

    assert bool(as_floats["feasible"]), as_floats["error"]
    assert _pick(as_floats) == _pick(as_ints)


@pytest.mark.parametrize(
    "cells,error",
    [
        ({"batch_size": "0"}, "batch_size must be >= 1, got '0'"),
        ({"tokens_per_sample": "-1024"}, "tokens_per_sample must be >= 1, got '-1024'"),
        ({"tokens_per_sample": "1024.5"}, "tokens_per_sample must be a whole number, got '1024.5'"),
        ({"max_gpus": "4.5"}, "max_gpus must be a whole number, got '4.5'"),
    ],
)
def test_bad_integer_cells_fail_the_row_with_their_name(cells, error):
    out = _recommend(_text_rows({**JOB, **cells}))
    assert not bool(out.iloc[0]["feasible"])
    assert out.iloc[0]["error"] == error


def _csv_text(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    pd.DataFrame(rows).to_csv(buffer, index=False)
    return buffer.getvalue()


def test_the_csv_endpoint_reads_float_and_blank_layout_cells():
    from fastapi.testclient import TestClient

    from coastline.ui.app import app

    # pandas writes a column with a blank as floats: 8.0 and 1.0, and an empty cell.
    text = _csv_text([{**JOB, "gpus_per_node": 8, "number_of_nodes": 1}, JOB])
    assert "8.0,1.0" in text and text.endswith(",,\n")
    with TestClient(app) as client:
        resp = client.post("/api/recommend/csv", json={"csv": text, "feasibility": "rules"})

    assert resp.status_code == 200, resp.text
    out = pd.read_csv(io.StringIO(resp.json()["csv"]))
    assert out["feasible"].tolist() == [True, True], out["error"].tolist()


def test_the_csv_endpoint_gives_min_gpu_the_float_layout(checker):
    from fastapi.testclient import TestClient

    from coastline.ui.app import app

    text = _csv_text([{**JOB, "gpus_per_node": 8, "number_of_nodes": 1}, JOB])
    with TestClient(app) as client:
        resp = client.post("/api/recommend/csv", json={"csv": text, "goal": "min_gpu", "max_gpus": 16})

    assert resp.status_code == 200, resp.text
    out = pd.read_csv(io.StringIO(resp.json()["csv"]))
    assert out["feasible"].tolist() == [True, True], out["error"].tolist()
    assert list(zip(out["total_gpus"], out["recommended_batch_size"])) == [(4, 8), (4, 1)]


def test_the_batch_endpoint_ignores_a_zero_layout_under_the_default_goal():
    from fastapi.testclient import TestClient

    from coastline.ui.app import app

    workload = {**JOB, "gpus_per_node": "8.0", "number_of_nodes": 0}
    with TestClient(app) as client:
        resp = client.post("/api/recommend/batch", json={"workloads": [workload], "feasibility": "rules"})

    assert resp.status_code == 200, resp.text
    result = resp.json()["results"][0]
    assert result["feasible"] is True, result["error"]
