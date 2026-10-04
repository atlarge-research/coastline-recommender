"""Every entry point that reaches min_gpu runs the thesis algorithm.

The feasibility checker is replaced by one that admits 4 GPUs or more, so the thesis algorithm
picks 4 GPUs (1 and 2 fail) and splits the job's total batch over them, whatever batch sizes and
GPU counts the caller's grid lists. JOB gives no layout, so it is a 1-GPU job with a total batch
of 16, and the pick is 4 GPUs at 4 per device. Kavier predicts the pick. Without a top_k, every
entry point returns one configuration for min_gpu.
"""

from __future__ import annotations

import csv
import json
from typing import Any

import pandas as pd
import pytest
import yaml

import coastline
import coastline.sdk.policies as policies
from coastline.sdk.models.workload import WorkloadSpec

GPU = "NVIDIA-A100-SXM4-80GB"
JOB = {
    "llm_model": "mistral-7b-v0.1",
    "fine_tuning_method": "lora",
    "gpu_model": GPU,
    "tokens_per_sample": 1024,
    "batch_size": 16,
}


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


def _assert_thesis_order(checker: FeasibleFrom4, total: int) -> None:
    assert checker.checked[:3] == [(1, total), (2, total // 2), (4, total // 4)]


def test_batch_api(checker):
    out = coastline.recommend(
        [JOB], goal="min_gpu", predictor="kavier", feasibility="rules", max_gpus=16, batch_sizes=[2, 64]
    )
    row = out.iloc[0]
    assert bool(row["feasible"]), row["error"]
    assert (int(row["total_gpus"]), int(row["recommended_batch_size"])) == (4, 4)
    assert row["throughput_tok_s"] > 0
    _assert_thesis_order(checker, 16)


def test_batch_api_reads_the_jobs_layout(checker):
    # 4 per device on 8 GPUs is a total batch of 32.
    job = {**JOB, "batch_size": 4, "gpus_per_node": 8, "number_of_nodes": 1}
    out = coastline.recommend([job], goal="min_gpu", predictor="kavier", feasibility="rules", max_gpus=16)
    row = out.iloc[0]
    assert bool(row["feasible"]), row["error"]
    assert (int(row["total_gpus"]), int(row["recommended_batch_size"])) == (4, 8)
    _assert_thesis_order(checker, 32)


def test_batch_api_goal_alias(checker):
    out = coastline.recommend([JOB], goal="Min-GPU", predictor="kavier", feasibility="rules", max_gpus=16)
    assert int(out.iloc[0]["total_gpus"]) == 4


def test_facade_goal(checker):
    recs = coastline.Coastline("kavier", feasibility="rules").recommend(
        JOB, goal="min_gpu", batch_sizes=[2, 64], total_gpus=[1, 3], max_gpus=16, top_k=5
    )
    # The first feasible counts in doubling order, up to max_gpus.
    assert [(r.total_gpus, r.metadata["batch_size"]) for r in recs] == [(4, 4), (8, 2), (16, 1)]
    assert all(r.predicted_throughput and r.predicted_throughput > 0 for r in recs)


def test_facade_returns_one_without_a_top_k(checker):
    recs = coastline.Coastline("kavier", feasibility="rules").recommend(JOB, goal="min_gpu", max_gpus=16)
    assert [(r.total_gpus, r.metadata["batch_size"]) for r in recs] == [(4, 4)]


def test_facade_strategy_argument(checker):
    recs = coastline.Coastline("kavier", feasibility="rules").recommend(
        JOB, strategy="min_gpu", preset="performance", max_gpus=16, top_k=1
    )
    assert [(r.total_gpus, r.metadata["batch_size"]) for r in recs] == [(4, 4)]


def test_repl_engine_returns_one_without_a_top_k(checker):
    from coastline.sdk.recommend import engine
    from coastline.sdk.recommend._goals import goal_to_label

    answers = {
        **engine.defaults(engine.resolve_options()),
        **JOB,
        "goal_label": goal_to_label("min_gpu"),
        "predictor": "kavier",
        "feasibility": "rules",
    }
    recs, _ = engine.run_pipeline(answers, top_k=None)
    assert [(r.total_gpus, r.metadata["batch_size"]) for r in recs] == [(4, 4)]


def _min_gpu_config(**extra: Any) -> dict:
    return {
        "strategy": {"name": "min_gpu"},
        "predictors": {"performance": "kavier", "energy": "kavier_power", "feasibility": "autoconf"},
        "grid": {"batch_sizes": [2, 64], "total_gpus": [1, 3], "top_k": 1},
        **extra,
    }


def test_recommend_csv(checker, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(_min_gpu_config()), encoding="utf-8")
    jobs = tmp_path / "jobs.csv"
    pd.DataFrame([JOB]).to_csv(jobs, index=False)
    out = tmp_path / "out.csv"

    coastline.recommend_csv(config, jobs, out, cluster_gpus=16)

    row = next(csv.DictReader(out.open(newline="", encoding="utf-8")))
    assert row["feasible"] == "True", row["error"]
    assert (int(row["recommended_total_gpus"]), int(row["recommended_batch_size"])) == (4, 4)
    _assert_thesis_order(checker, 16)


def test_recommend_job_config(checker, tmp_path, capsys):
    from coastline.cli import main

    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump(_min_gpu_config(workload=dict(JOB))), encoding="utf-8")

    main(["recommend-job", "--config", str(config), "--cluster-gpus", "16"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["configuration"]["total_gpus"] == 4
    assert payload["metadata"]["batch_size"] == 4
    _assert_thesis_order(checker, 16)


def _trace_row() -> dict:
    return {
        "metadata.model_name": JOB["llm_model"],
        "metadata.method": JOB["fine_tuning_method"],
        "resources.gpu_model": GPU,
        "metadata.tokens_per_sample": JOB["tokens_per_sample"],
        "metadata.batch_size": 128,
        "resources.num_gpus_per_node": 8,
        "resources.num_nodes": 1,
        "per_device_train_batch_size": 16,
    }


def _run_trace(tmp_path, rows: list[dict]) -> pd.DataFrame:
    from coastline.cli import main

    trace = tmp_path / "trace.csv"
    pd.DataFrame(rows).to_csv(trace, index=False)
    out = tmp_path / "out.csv"
    main(
        ["recommend-trace", "--input", str(trace), "--output", str(out), "--goal", "min_gpu"]
        + ["--cluster-gpus", "16", "--workers", "1"]
    )
    return pd.read_csv(out)


def test_recommend_trace(checker, tmp_path):
    row = _run_trace(tmp_path, [_trace_row()]).iloc[0]

    assert pd.isna(row["metadata.recommendation_note"]), row["metadata.recommendation_note"]
    assert int(row["resources.num_gpus_per_node"]) * int(row["resources.num_nodes"]) == 4
    # 16 per device on 8 GPUs: the total batch of 128 is kept and split over 4 GPUs.
    assert int(row["per_device_train_batch_size"]) == 32
    assert int(row["metadata.batch_size"]) == 128
    _assert_thesis_order(checker, 128)


def test_recommend_trace_legacy_mode_keeps_the_total_batch(checker, tmp_path):
    # No per-device column: metadata.batch_size is the job's total batch.
    legacy = {k: v for k, v in _trace_row().items() if k != "per_device_train_batch_size"}
    row = _run_trace(tmp_path, [legacy]).iloc[0]

    assert pd.isna(row["metadata.recommendation_note"]), row["metadata.recommendation_note"]
    assert int(row["resources.num_gpus_per_node"]) * int(row["resources.num_nodes"]) == 4
    assert int(row["metadata.batch_size"]) == 128
    _assert_thesis_order(checker, 128)


def test_explain(checker, capsys):
    from coastline.cli import main

    main(
        [
            "explain",
            "--model",
            JOB["llm_model"],
            "--method",
            JOB["fine_tuning_method"],
            "--gpu-model",
            GPU,
            "--tokens",
            str(JOB["tokens_per_sample"]),
            "--batch-size",
            "16",
            "--strategy",
            "min_gpu",
            "--max-gpus",
            "16",
        ]
    )
    out = capsys.readouterr().out
    assert "winner    4 GPU(s)" in out
    # Without --top-k, min_gpu shows one configuration.
    assert "\n   2  " not in out
    _assert_thesis_order(checker, 16)


def test_dashboard(checker):
    from fastapi.testclient import TestClient

    from coastline.ui.app import app

    body = {**JOB, "prediction_model": "kavier", "strategy": "min_gpu", "total_gpus": 16}
    with TestClient(app) as client:
        resp = client.post("/api/recommend", json=body)

    assert resp.status_code == 200, resp.text
    top = resp.json()["recommendation"]
    assert (top["total_gpus"], top["batch_size"]) == (4, 4)
    # The dashboard has no top_k field, so min_gpu returns one configuration.
    assert len(resp.json()["candidates"]) == 1
    _assert_thesis_order(checker, 16)


def test_dashboard_names_min_gpu_when_no_gpu_count_is_feasible(monkeypatch):
    from fastapi.testclient import TestClient

    from coastline.ui.app import app

    class Never:
        def is_feasible(self, workload: WorkloadSpec) -> tuple[bool, dict[str, Any]]:
            return False, {}

    monkeypatch.setattr(policies, "create_feasibility_checker", lambda predictor_config: Never())
    body = {**JOB, "batch_size": 64, "prediction_model": "kavier", "strategy": "min_gpu", "total_gpus": 16}
    with TestClient(app) as client:
        resp = client.post("/api/recommend", json=body)

    assert resp.status_code == 404, resp.text
    detail = resp.json()["detail"]
    assert "Minimum GPUs found no feasible GPU count" in detail
    assert "total batch of 64" in detail
    assert "1, 2, 4, 8 and 16 GPUs" in detail
    assert "unsupported" not in detail


def test_legacy_trace_never_gets_more_gpus_than_the_job_had(tmp_path):
    # Real AutoConf. Each job ran on 8 GPUs; metadata.batch_size is its total batch.
    def job(method: str, tokens: int, total: int) -> dict:
        return {
            "metadata.model_name": "mistral-7b-v0.1",
            "metadata.method": method,
            "resources.gpu_model": GPU,
            "metadata.tokens_per_sample": tokens,
            "metadata.batch_size": total,
            "resources.num_gpus_per_node": 8,
            "resources.num_nodes": 1,
        }

    jobs = [job("lora", 4096, 16), job("lora", 2048, 32), job("full", 2048, 16)]
    out = _run_trace(tmp_path, jobs)

    for original, (_, row) in zip(jobs, out.iterrows()):
        assert pd.isna(row["metadata.recommendation_note"]), row["metadata.recommendation_note"]
        gpus = int(row["resources.num_gpus_per_node"]) * int(row["resources.num_nodes"])
        assert gpus <= 8
        assert int(row["metadata.batch_size"]) == original["metadata.batch_size"]
        assert int(row["metadata.batch_size"]) % gpus == 0
