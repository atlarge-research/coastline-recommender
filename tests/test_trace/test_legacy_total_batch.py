"""In a legacy trace (no per-device batch column), metadata.batch_size is the job's total batch for
every goal.

The weighted goals seed Kavier with the per-device batch, total / (gpus_per_node x nodes), and a
row whose total does not split evenly over its GPUs is kept unchanged with a note. min_gpu takes
the total as it is. Every goal writes back the recommended per-device batch times the recommended
GPUs. The bundled sample trace follows the same rule: each total splits evenly over its GPUs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from coastline.sdk.trace import recommend as trace_recommend
from coastline.sdk.trace.recommend import recommend_trace

_SAMPLE = Path(__file__).resolve().parents[2] / "config" / "coastline_functionality" / "sample_trace.csv"

# A job on 8 GPUs x 2 nodes with a total batch of 64: 4 per device.
_ROW = {
    "metadata.model_name": "mistral-7b-v0.1",
    "metadata.method": "lora",
    "resources.gpu_model": "NVIDIA-A100-SXM4-80GB",
    "metadata.tokens_per_sample": 1024,
    "metadata.batch_size": 64,
    "resources.num_gpus_per_node": 8,
    "resources.num_nodes": 2,
    "metadata.output.train_tokens_per_second": 15000.0,
    "metadata.train_runtime": 3600.0,
}


@pytest.fixture
def engine_calls(monkeypatch) -> list[dict[str, Any]]:
    """Replace coastline.recommend with a fake that records each workload and picks 4 GPUs on one
    node at 2 per device."""
    calls: list[dict[str, Any]] = []

    def fake_recommend(workloads, **kw):
        calls.append({**workloads[0], "goal": kw["goal"]})
        pick = {"feasible": True, "number_of_nodes": 1, "gpus_per_node": 4, "batch_size": 2, "throughput_tok_s": 9e3}
        return pd.DataFrame([pick])

    monkeypatch.setattr(trace_recommend.coastline, "recommend", fake_recommend)
    return calls


def _run(tmp_path, rows: list[dict[str, Any]], goal: str) -> pd.DataFrame:
    trace = tmp_path / "trace.csv"
    pd.DataFrame(rows).to_csv(trace, index=False)
    return recommend_trace(str(trace), str(tmp_path / "out.csv"), method="kavier", goal=goal, cluster_gpus=32)


@pytest.mark.parametrize("goal", ["performance", "balanced", "energy"])
def test_a_weighted_goal_seeds_with_the_per_device_batch(tmp_path, engine_calls, goal):
    row = _run(tmp_path, [_ROW], goal).iloc[0]

    assert engine_calls[0]["batch_size"] == 4  # 64 over 16 GPUs
    assert pd.isna(row["metadata.recommendation_note"]), row["metadata.recommendation_note"]
    # The total batch written back: 2 per device on 4 GPUs.
    assert int(row["metadata.batch_size"]) == 8
    assert (int(row["resources.num_gpus_per_node"]), int(row["resources.num_nodes"])) == (4, 1)


def test_min_gpu_seeds_with_the_total_and_writes_back_a_total(tmp_path, engine_calls):
    row = _run(tmp_path, [_ROW], "min_gpu").iloc[0]

    assert engine_calls[0]["batch_size"] == 64
    assert int(row["metadata.batch_size"]) == 8


@pytest.mark.parametrize("goal", ["performance", "balanced", "energy"])
def test_a_total_that_does_not_split_over_the_gpus_keeps_the_row(tmp_path, engine_calls, goal):
    odd = {**_ROW, "metadata.batch_size": 12}
    row = _run(tmp_path, [odd, _ROW], goal).iloc[0]

    # Only the second row reached the engine.
    assert [call["batch_size"] for call in engine_calls] == [4]
    note = str(row["metadata.recommendation_note"])
    assert "the total batch 12 does not split evenly over 16 GPUs" in note
    assert "unchanged" in note
    assert int(row["metadata.batch_size"]) == 12
    assert (int(row["resources.num_gpus_per_node"]), int(row["resources.num_nodes"])) == (8, 2)


def test_min_gpu_takes_a_total_that_does_not_split_over_the_jobs_gpus(tmp_path, engine_calls):
    row = _run(tmp_path, [{**_ROW, "metadata.batch_size": 12}], "min_gpu").iloc[0]

    assert engine_calls[0]["batch_size"] == 12
    assert pd.isna(row["metadata.recommendation_note"]), row["metadata.recommendation_note"]


def test_every_total_in_the_sample_trace_splits_over_its_gpus():
    trace = pd.read_csv(_SAMPLE)
    gpus = trace["resources.num_gpus_per_node"] * trace["resources.num_nodes"]
    assert (trace["metadata.batch_size"] % gpus == 0).all(), trace[["metadata.batch_size"]].assign(gpus=gpus)


@pytest.mark.parametrize("goal", ["performance", "min_gpu"])
def test_the_sample_trace_writes_totals_for_every_goal(tmp_path, goal):
    # The real engine with the rules check: each recommended row gets a total that splits over
    # its recommended GPUs, and no row is kept for its batch.
    out = recommend_trace(
        str(_SAMPLE), str(tmp_path / "out.csv"), method="kavier", goal=goal, feasibility="rules", cluster_gpus=32
    )
    notes = out["metadata.recommendation_note"].dropna().astype(str)
    assert not notes.str.contains("does not split").any(), notes.tolist()
    gpus = out["resources.num_gpus_per_node"] * out["resources.num_nodes"]
    assert (out["metadata.batch_size"] % gpus == 0).all()
