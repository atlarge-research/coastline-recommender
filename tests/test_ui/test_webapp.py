"""Integration tests for the FastAPI web application.

Covers the endpoints unique to the webapp surface: the `/` HTML dashboard and
the `/api/predict` playground (one config across selected predictors). The app
uses a lifespan to load options/strategy config, so the TestClient is entered
as a context manager to trigger startup.

Every assertion carries an independent oracle: config-echo / derived fields are
hand-derived from the request; runtime and energy are cross-checked against the
*reported* throughput (so the derivation is in a different form than the impl);
the unsupported-GPU and model-cap paths assert the documented contract.

NOTE: /api/health and /api/recommend (kavier candidates, min_gpu, 422) are
covered by api/tests/test_api.py; this file keeps only the `/` page and the
`/api/predict` playground.
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


def test_index_dashboard_lists_the_same_gpu_catalog_as_the_options_api(client):
    """The rendered dashboard must offer exactly the GPUs advertised by
    /api/options — the dropdown is a view of that catalog, not a separate list.
    Oracle: cross-surface consistency invariant (HTML options == API catalog),
    independent of any hard-coded GPU name. A bug that drops a GPU from the
    template, or renders a stale/different set, breaks this."""
    page = client.get("/")
    assert page.status_code == 200
    assert "COASTLINE" in page.text  # guard: real dashboard, not an error page

    catalog = client.get("/api/options").json()["gpus"]
    assert catalog, "options API must advertise at least one GPU"
    for gpu in catalog:
        # Each catalog GPU is rendered as a selectable <option value="...">.
        assert f'value="{gpu}"' in page.text, f"{gpu} missing from dashboard dropdown"


def test_predict_echoes_config_and_derives_total_gpus(client):
    """The playground echoes the submitted config and reports total_gpus as the
    product of the node layout. Oracle: total_gpus = gpus_per_node * nodes,
    hand-computed 4 * 2 = 8 (a different, non-echo field). A bug using a sum, or
    reading the wrong field, would not land on 8."""
    resp = client.post("/api/predict", json=_predict_body(gpus_per_node=4, number_of_nodes=2))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["success"] is True
    cfg = data["config"]
    assert cfg["llm_model"] == "mistral-7b-v0.1"  # echoed verbatim
    assert cfg["gpus_per_node"] == 4 and cfg["number_of_nodes"] == 2
    # 4 GPUs/node across 2 nodes = 8 GPUs.
    assert cfg["total_gpus"] == 8


def test_predict_kavier_runtime_and_energy_are_consistent_with_reported_throughput(client):
    """runtime and energy must satisfy the physics identities against the values
    Kavier reports, so a wrong total-tokens product or a mangled unit factor is
    caught. Oracles (different form than the impl, chained off the reported
    throughput/power):
        total_tokens = dataset_size * epochs * tokens_per_sample
                     = 10000 * 3 * 2048 = 61_440_000
        runtime_s    = total_tokens / throughput
        energy_kwh   = power_per_gpu * total_gpus * runtime_s / 3600 / 1000
    A /3600-vs-/1000 slip, or dropping the total_gpus multiplier, goes red."""
    resp = client.post("/api/predict", json=_predict_body(gpus_per_node=8, number_of_nodes=1))
    assert resp.status_code == 200, resp.text
    result = resp.json()["results"][0]
    assert result["available"] is True

    throughput = result["predicted_throughput"]
    power = result["power_watts"]

    # 10000 samples * 3 epochs * 2048 tokens/sample = 61,440,000 tokens.
    total_tokens = 10000 * 3 * 2048
    assert total_tokens == 61_440_000  # oracle pinned by hand
    expected_runtime = total_tokens / throughput
    assert result["predicted_runtime_seconds"] == pytest.approx(expected_runtime)

    # 8 GPUs at power_per_gpu watts each, for expected_runtime seconds.
    # Wh = W * s / 3600; kWh = Wh / 1000.
    expected_energy_kwh = power * 8 * expected_runtime / 3600 / 1000
    assert result["energy_kwh"] == pytest.approx(expected_energy_kwh)


def test_predict_kavier_power_is_per_gpu_within_the_a100_envelope(client):
    """The reported power is per-GPU, so it must sit inside a single A100's power
    envelope. Oracle: an A100-SXM4-80GB draws >0 and <= its 400 W TDP (analytic
    hardware spec). This rejects the classic bug of reporting *cluster* power
    (8 x per-GPU ~ 1.7 kW, far above 400) or a unit blow-up, and pins throughput
    as a finite positive quantity for a supported (model, GPU, method)."""
    resp = client.post("/api/predict", json=_predict_body(gpus_per_node=8, number_of_nodes=1))
    result = resp.json()["results"][0]
    assert result["available"] is True

    throughput = result["predicted_throughput"]
    assert throughput > 0 and throughput == throughput  # finite, positive (NaN != NaN)

    power = result["power_watts"]
    # A100-SXM4-80GB TDP = 400 W; a per-GPU figure cannot exceed it.
    assert 0 < power <= 400


def test_predict_unsupported_gpu_marks_model_unavailable_without_failing_batch(client):
    """An uncalibrated/typo'd GPU is per-model isolated: that model comes back
    available=False but the request still succeeds (200, success=True). Oracle:
    the documented playground contract — one bad predictor never sinks the batch.
    A regression that 500s on an unknown GPU, or flips available to True, fails."""
    resp = client.post("/api/predict", json=_predict_body(gpu_model="NOT-A-REAL-GPU"))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["success"] is True
    result = data["results"][0]
    assert result["model"] == "kavier"
    assert result["available"] is False
    assert "predicted_throughput" not in result  # no numbers when unavailable


def test_predict_rejects_more_models_than_the_per_request_cap(client):
    """Each model spawns its own subprocess, so the request is capped. Oracle:
    the default cap is 6 (COASTLINE_MAX_PREDICT_MODELS), so 7 requested models
    must be rejected with 400 before any subprocess launches. A raised/removed
    cap, or an off-by-one that admits 7, fails this."""
    resp = client.post("/api/predict", json=_predict_body(models=["kavier"] * 7))
    assert resp.status_code == 400, resp.text
    assert resp.json()["success"] is False
