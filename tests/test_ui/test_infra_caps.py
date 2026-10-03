"""The dashboard keeps every recommended layout within infrastructure.yaml's node width.

Nodes mode rejects a request for wider nodes than the cluster has; total mode and the batch
endpoints do not produce one either. The cluster here is 16 GPUs on four 4-GPU nodes, and the
policy pins Kavier and the `rules` checker, so no AutoConf model is needed.
"""

from __future__ import annotations

import csv
import io

import pytest
import yaml
from fastapi.testclient import TestClient

import coastline.ui.app as app_module
from coastline.sdk.io.infrastructure import Infrastructure

_FOUR_GPU_NODES = Infrastructure(total_gpus=16, max_nodes=4, max_gpus_per_node=4, gpu_models=["NVIDIA-A100-SXM4-80GB"])

_WORKLOAD = {
    "llm_model": "mistral-7b-v0.1",
    "fine_tuning_method": "lora",
    "gpu_model": "NVIDIA-A100-SXM4-80GB",
    "tokens_per_sample": 1024,
    "batch_size": 16,
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    policy = tmp_path / "experiment.yaml"
    policy.write_text(
        yaml.safe_dump(
            {
                "strategy": {"name": "multi_objective", "preset": "performance"},
                "predictors": {"performance": "kavier", "energy": "kavier_power", "feasibility": "rules"},
                "grid": {"batch_sizes": [8, 16], "top_k": 5},
            }
        )
    )
    monkeypatch.setenv("EXPERIMENT_CONFIG", str(policy))
    monkeypatch.setattr(app_module, "load_infrastructure", lambda: _FOUR_GPU_NODES)
    with TestClient(app_module.app) as test_client:
        yield test_client


def _check_layouts(rows: list[dict]) -> None:
    assert rows
    for row in rows:
        assert row["gpus_per_node"] <= 4, row
        assert row["gpus_per_node"] * row["number_of_nodes"] == row["total_gpus"] <= 16, row


def test_total_mode_never_offers_a_node_wider_than_the_cluster(client) -> None:
    body = {**_WORKLOAD, "hardware_mode": "total", "total_gpus": 16, "prediction_model": "kavier"}
    resp = client.post("/api/recommend", json={**body, "strategy": "multi_objective", "preset": "performance"})

    assert resp.status_code == 200, resp.text
    _check_layouts(resp.json()["candidates"])


def test_the_batch_endpoint_never_offers_a_node_wider_than_the_cluster(client) -> None:
    resp = client.post(
        "/api/recommend/batch",
        json={"workloads": [_WORKLOAD], "goal": "performance", "max_gpus": 16, "top_k": 5, "feasibility": "rules"},
    )

    assert resp.status_code == 200, resp.text
    results = resp.json()["results"]
    assert all(row["feasible"] for row in results)
    _check_layouts(results)


def test_the_csv_endpoint_never_offers_a_node_wider_than_the_cluster(client) -> None:
    header = ",".join(_WORKLOAD)
    values = ",".join(str(v) for v in _WORKLOAD.values())
    resp = client.post(
        "/api/recommend/csv",
        json={"csv": f"{header}\n{values}\n", "goal": "performance", "max_gpus": 16, "feasibility": "rules"},
    )

    assert resp.status_code == 200, resp.text
    (row,) = csv.DictReader(io.StringIO(resp.json()["csv"]))
    assert row["feasible"] == "True"
    assert int(float(row["gpus_per_node"])) <= 4
