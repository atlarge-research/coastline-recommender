"""Integration tests for the FastAPI web application: the `/` HTML dashboard and the
`/api/predict` playground (one config across selected predictors).

The app loads the options and strategy config in its lifespan, so the TestClient is entered as a
context manager to run startup. Echoed and derived fields are checked against the request,
runtime and energy against the reported throughput, and the unsupported-GPU and model-cap cases
against the documented behaviour. /api/health and /api/recommend are covered in test_service.py.
"""

import pytest
from fastapi.testclient import TestClient

from coastline.ui.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _predict_body(**overrides):
    body = {
        "llm_model": "mistral-7b-v0.1",
        "fine_tuning_method": "full",
        "gpu_model": "NVIDIA-A100-SXM4-80GB",
        "tokens_per_sample": 2048,
        "batch_size": 8,
        "gpus_per_node": 8,
        "number_of_nodes": 1,
        "dataset_size": 10000,
        "training_epochs": 3,
        "models": ["kavier"],
    }
    body.update(overrides)
    return body


def test_index_dashboard_shows_the_tabpfn_attribution(client):
    """The Prior Labs License (section 10) asks every user interface to show this line."""
    page = client.get("/")
    assert page.status_code == 200
    assert "Built with PriorLabs-TabPFN" in page.text


def test_index_dashboard_lists_the_same_gpu_catalog_as_the_options_api(client):
    """Every GPU that /api/options lists is an option in the dashboard dropdown."""
    page = client.get("/")
    assert page.status_code == 200
    assert "COASTLINE" in page.text  # the page is the dashboard

    catalog = client.get("/api/options").json()["gpus"]
    assert catalog, "options API must advertise at least one GPU"
    for gpu in catalog:
        # Each catalog GPU is rendered as a selectable <option value="...">.
        assert f'value="{gpu}"' in page.text, f"{gpu} missing from dashboard dropdown"


def test_predict_echoes_config_and_derives_total_gpus(client):
    """The playground echoes the submitted config and reports total_gpus = gpus_per_node x nodes
    (4 x 2 = 8; a sum would give 6)."""
    resp = client.post("/api/predict", json=_predict_body(gpus_per_node=4, number_of_nodes=2))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["success"] is True
    cfg = data["config"]
    assert cfg["llm_model"] == "mistral-7b-v0.1"  # echoed as sent
    assert cfg["gpus_per_node"] == 4 and cfg["number_of_nodes"] == 2
    # 4 GPUs/node across 2 nodes = 8 GPUs.
    assert cfg["total_gpus"] == 8


def test_predict_kavier_runtime_and_energy_are_consistent_with_reported_throughput(client):
    """Runtime and energy follow from the throughput and power that Kavier reports:

    total_tokens = dataset_size * epochs * tokens_per_sample
                 = 10000 * 3 * 2048 = 61_440_000
    runtime_s    = total_tokens / throughput
    energy_kwh   = power_per_gpu * total_gpus * runtime_s / 3600 / 1000
    """
    resp = client.post("/api/predict", json=_predict_body(gpus_per_node=8, number_of_nodes=1))
    assert resp.status_code == 200, resp.text
    result = resp.json()["results"][0]
    assert result["available"] is True

    throughput = result["predicted_throughput"]
    power = result["power_watts"]

    # 10000 samples * 3 epochs * 2048 tokens/sample = 61,440,000 tokens.
    total_tokens = 10000 * 3 * 2048
    assert total_tokens == 61_440_000
    expected_runtime = total_tokens / throughput
    assert result["predicted_runtime_seconds"] == pytest.approx(expected_runtime)

    # 8 GPUs at power_per_gpu watts each, for expected_runtime seconds.
    # Wh = W * s / 3600; kWh = Wh / 1000.
    expected_energy_kwh = power * 8 * expected_runtime / 3600 / 1000
    assert result["energy_kwh"] == pytest.approx(expected_energy_kwh)


def test_predict_kavier_power_is_per_gpu_within_the_a100_envelope(client):
    """The reported power is per GPU, so it lies in (0, 400] W; 400 W is the A100-SXM4-80GB TDP.

    Cluster power for 8 GPUs would be about 1.7 kW. The throughput is finite and positive for this
    supported model, GPU and method."""
    resp = client.post("/api/predict", json=_predict_body(gpus_per_node=8, number_of_nodes=1))
    result = resp.json()["results"][0]
    assert result["available"] is True

    throughput = result["predicted_throughput"]
    assert throughput > 0 and throughput == throughput  # finite, positive (NaN != NaN)

    power = result["power_watts"]
    # A100-SXM4-80GB TDP = 400 W; a per-GPU figure cannot exceed it.
    assert 0 < power <= 400


def test_predict_unsupported_gpu_marks_model_unavailable_without_failing_batch(client):
    """An unknown GPU marks that model available=False while the request still succeeds (200,
    success=True)."""
    resp = client.post("/api/predict", json=_predict_body(gpu_model="NOT-A-REAL-GPU"))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["success"] is True
    result = data["results"][0]
    assert result["model"] == "kavier"
    assert result["available"] is False
    assert "predicted_throughput" not in result  # no numbers when unavailable


def test_predict_rejects_more_models_than_the_per_request_cap(client):
    """Each model runs in its own subprocess, so a request is capped at 6 models by default
    (COASTLINE_MAX_PREDICT_MODELS); 7 are rejected with 400 before any subprocess starts."""
    resp = client.post("/api/predict", json=_predict_body(models=["kavier"] * 7))
    assert resp.status_code == 400, resp.text
    assert resp.json()["success"] is False
