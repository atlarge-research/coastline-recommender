"""Tests for the Coastline FastAPI service (``coastline.ui.app``).

Run from the repo root with::

    PYTHONPATH=coastline:coastline/common:kavier/src \
    DATA_DIR=./trace-archive .venv/bin/python -m pytest coastline/api/tests -q

These tests exercise the HTTP surface via ``fastapi.testclient.TestClient`` and the
strategy-config loader in isolation. They never load the trained ML pickles:
``/api/recommend`` is always invoked with ``prediction_model="kavier"`` (the analytical
predictor), which is also the model's default. ``PolicyFactory.throughput_predictor``
maps ``"kavier"`` to ``create_physics_driven()`` with no unpickling, so
the data-driven artifacts (which segfault on this host) are never touched.

Production code in ``coastline/api/main.py`` is not modified.
"""

from __future__ import annotations

import shutil

import pytest
from fastapi.testclient import TestClient

import coastline.ui.app as main
from coastline.ui.app import _DEFAULT_STRATEGY_CONFIG, _load_strategy_config, app

# A minimal Kavier-supported workload. mistral-7b-v0.1 + A100-SXM4-80GB are both in
# the calibrated Kavier libraries, so the analytical predictor returns a result.
_KAVIER_BODY = {
    "llm_model": "mistral-7b-v0.1",
    "fine_tuning_method": "full",
    "gpu_model": "NVIDIA-A100-SXM4-80GB",
    "tokens_per_sample": 1024,
    "batch_size": 8,
    "prediction_model": "kavier",
    "strategy": "multi_objective",
    "preset": "balanced",
    "total_gpus": 8,
}

# Env vars that override the strategy-config search path; cleared so tests are
# deterministic regardless of the caller's shell environment.
_STRATEGY_ENV_KEYS = ("STRATEGY_CONFIG", "EXPERIMENT_CONFIG", "CONFIG_FILE")


@pytest.fixture(scope="module")
def client():
    """A TestClient with the app lifespan run (loads OPTIONS + STRATEGY_CONFIG)."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def clean_strategy_env(monkeypatch):
    """Remove strategy-config env overrides for the duration of a test."""
    for key in _STRATEGY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


# --------------------------------------------------------------------------- #
# /api/health
# --------------------------------------------------------------------------- #


def test_health_ok(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["status"] == "healthy"
    # The lifespan ran, so the strategy config must be populated.
    assert body["strategy_config_loaded"] is True
    # These keys are part of the documented health contract.
    assert set(body) >= {
        "success",
        "status",
        "options_loaded",
        "config_loaded",
        "strategy_config_loaded",
    }


def test_options_endpoint_lists_available_inputs(client):
    resp = client.get("/api/options")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    # A consumer discovers valid inputs here instead of scraping the dashboard HTML.
    assert set(body) >= {"models", "methods", "gpus", "predictors", "recommendation_policies", "presets"}
    assert any(p["id"] == "kavier" for p in body["predictors"])


def test_version_endpoint_reports_a_version(client):
    resp = client.get("/api/version")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True and body["name"] == "coastline"
    assert isinstance(body["version"], str) and body["version"]


# --------------------------------------------------------------------------- #
# RecommendRequest validation (pydantic model, no network)
# --------------------------------------------------------------------------- #


def test_unknown_predictor_is_rejected(client):
    resp = client.post("/api/recommend", json={**_KAVIER_BODY, "prediction_model": "xgbost"})
    assert resp.status_code == 422
    assert "unknown predictor" in resp.json()["error"]


def test_batch_recommend_returns_records_with_rationale(client):
    body = {
        "workloads": [
            {
                "model": "mistral-7b-v0.1",
                "method": "full",
                "gpu_model": "NVIDIA-A100-SXM4-80GB",
                "tokens_per_sample": 1024,
                "batch_size": 8,
            },
            {
                "model": "mistral-7b-v0.1",
                "method": "full",
                "gpu_model": "NVIDIA-A100-SXM4-80GB",
                "tokens_per_sample": 2048,
                "batch_size": 8,
            },
        ],
        "predictor": "kavier",
        "max_gpus": 8,
    }
    resp = client.post("/api/recommend/batch", json=body)
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["success"] is True and payload["count"] == 2
    rows = payload["results"]
    assert len(rows) == 2 and all(r["feasible"] for r in rows)
    assert "rationale" in rows[0]  # facade parity: the same 'why' column as coastline.recommend


def test_batch_recommend_empty_workloads_is_422(client):
    resp = client.post("/api/recommend/batch", json={"workloads": []})
    assert resp.status_code == 422


def test_batch_recommend_default_predictor_is_kavier(client):
    """Omitting ``predictor`` from a batch request must default to 'kavier', not 'intelligent'.

    This guards the surface-alignment requirement: facade, batch API, and CSV endpoint
    must all default to the same predictor so identical default calls return identical results.
    """
    body_no_predictor = {
        "workloads": [
            {
                "model": "mistral-7b-v0.1",
                "method": "full",
                "gpu_model": "NVIDIA-A100-SXM4-80GB",
                "tokens_per_sample": 1024,
                "batch_size": 8,
            },
        ],
        # predictor intentionally omitted — must pick up the kavier default
        "max_gpus": 8,
    }
    body_explicit_kavier = {**body_no_predictor, "predictor": "kavier"}

    resp_default = client.post("/api/recommend/batch", json=body_no_predictor)
    resp_explicit = client.post("/api/recommend/batch", json=body_explicit_kavier)
    assert resp_default.status_code == 200, resp_default.text
    assert resp_explicit.status_code == 200, resp_explicit.text

    default_throughput = resp_default.json()["results"][0]["throughput_tok_s"]
    explicit_throughput = resp_explicit.json()["results"][0]["throughput_tok_s"]
    assert default_throughput == pytest.approx(explicit_throughput, rel=1e-9), (
        "Default predictor is not 'kavier': throughput differed from explicit kavier call"
    )


def test_csv_endpoint_default_predictor_is_kavier(client):
    """Omitting ``predictor`` from a CSV request must default to 'kavier'."""
    csv_in = "model,method,gpu_model,tokens_per_sample,batch_size\nmistral-7b-v0.1,full,NVIDIA-A100-SXM4-80GB,1024,8\n"
    resp_default = client.post("/api/recommend/csv", json={"csv": csv_in, "max_gpus": 8})
    resp_kavier = client.post("/api/recommend/csv", json={"csv": csv_in, "predictor": "kavier", "max_gpus": 8})
    assert resp_default.status_code == 200, resp_default.text
    assert resp_kavier.status_code == 200, resp_kavier.text

    import csv
    import io

    default_rows = list(csv.DictReader(io.StringIO(resp_default.json()["csv"])))
    kavier_rows = list(csv.DictReader(io.StringIO(resp_kavier.json()["csv"])))
    assert len(default_rows) == 1 and len(kavier_rows) == 1
    assert default_rows[0]["throughput_tok_s"] == kavier_rows[0]["throughput_tok_s"], (
        "Default CSV predictor is not 'kavier': throughput differed from explicit kavier call"
    )


def test_recommend_csv_endpoint_returns_csv(client):
    csv_in = "model,method,gpu_model,tokens_per_sample,batch_size\nmistral-7b-v0.1,full,NVIDIA-A100-SXM4-80GB,1024,8\n"
    resp = client.post("/api/recommend/csv", json={"csv": csv_in, "predictor": "kavier", "max_gpus": 8})
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["success"] is True and payload["count"] == 1
    # the output CSV carries the recommendation + the rationale column
    assert "rationale" in payload["csv"] and "total_gpus" in payload["csv"]


def test_recommend_csv_endpoint_empty_is_400(client):
    resp = client.post("/api/recommend/csv", json={"csv": "model,method\n"})  # header only
    assert resp.status_code == 400


def test_recommend_runtime_is_dataset_scaled_and_has_rationale(client):
    resp = client.post("/api/recommend", json=_KAVIER_BODY)
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    top = payload["recommendation"]
    # Runtime now follows the engine/facade convention (total_tokens / throughput), so the
    # web /api/recommend agrees with coastline.recommend instead of diverging.
    total_tokens = 10000 * 3 * _KAVIER_BODY["tokens_per_sample"]  # dataset_size * epochs * tokens (defaults)
    assert top["predicted_runtime_seconds"] == pytest.approx(total_tokens / top["predicted_throughput"], rel=1e-6)
    # Facade-parity: the response now carries the one-line rationale.
    assert isinstance(payload["rationale"], str) and "GPU" in payload["rationale"]


def test_async_job_submit_and_poll(client):
    import time

    body = {
        "workloads": [
            {
                "model": "mistral-7b-v0.1",
                "method": "full",
                "gpu_model": "NVIDIA-A100-SXM4-80GB",
                "tokens_per_sample": 1024,
                "batch_size": 8,
            }
        ],
        "predictor": "kavier",
        "max_gpus": 8,
    }
    submit = client.post("/api/jobs", json=body)
    assert submit.status_code == 202
    job_id = submit.json()["job_id"]
    for _ in range(100):  # kavier finishes in ~1-2s; poll up to 10s
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["status"] == "done":
            assert state["result"]["count"] == 1
            assert state["result"]["results"][0]["feasible"]
            return
        assert state["status"] != "error", state.get("error")
        time.sleep(0.1)
    raise AssertionError("async job did not complete in time")


def test_async_job_unknown_id_is_404(client):
    assert client.get("/api/jobs/does-not-exist").status_code == 404


def test_async_job_records_error_when_workload_raises(client, monkeypatch):
    """A batch whose recommend raises (not a per-row feasible=False, but a hard
    failure) lands the job in status='error' with the message captured — the worker
    records the failure on the job rather than crashing the background thread."""
    import time

    import coastline

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic recommend failure")

    # The worker does `import coastline; coastline.recommend(...)`, so patching the
    # module attribute is seen by the background thread too.
    monkeypatch.setattr(coastline, "recommend", _boom)

    body = {
        "workloads": [
            {
                "model": "mistral-7b-v0.1",
                "method": "full",
                "gpu_model": "NVIDIA-A100-SXM4-80GB",
                "tokens_per_sample": 1024,
                "batch_size": 8,
            }
        ],
        "predictor": "kavier",
        "max_gpus": 8,
    }
    submit = client.post("/api/jobs", json=body)
    assert submit.status_code == 202
    job_id = submit.json()["job_id"]
    for _ in range(100):  # the worker fails immediately; poll briefly
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["status"] == "error":
            assert "synthetic recommend failure" in (state["error"] or "")
            assert state["result"] is None
            return
        assert state["status"] != "done", "expected an error, not a completed job"
        time.sleep(0.05)
    raise AssertionError("job never reached the error state")


# --------------------------------------------------------------------------- #
# /api/recommend/csv — row-cap (413) and invalid-goal (422) branches
# --------------------------------------------------------------------------- #


def test_recommend_csv_over_row_cap_is_413(client, monkeypatch):
    """A CSV with more data rows than COASTLINE_MAX_BATCH_WORKLOADS is rejected with
    413 before any recommendation runs."""
    monkeypatch.setattr(main, "_MAX_BATCH_WORKLOADS", 3)
    header = "model,method,gpu_model,tokens_per_sample,batch_size\n"
    row = "mistral-7b-v0.1,full,NVIDIA-A100-SXM4-80GB,1024,8\n"
    csv_in = header + row * 4  # 4 data rows > cap of 3
    resp = client.post("/api/recommend/csv", json={"csv": csv_in, "predictor": "kavier"})
    assert resp.status_code == 413
    assert "too many rows" in resp.json()["detail"].lower()


def test_recommend_csv_value_error_is_422(client, monkeypatch):
    """When ``coastline.recommend`` raises a ValueError/TypeError at the batch level
    (bad shape / unknown knob), the CSV endpoint translates it to 422. Patched to
    raise so the 422 branch is covered deterministically without ML loading."""
    import coastline

    def _raise(*args, **kwargs):
        raise ValueError("bad batch input")

    monkeypatch.setattr(coastline, "recommend", _raise)
    csv_in = "model,method,gpu_model,tokens_per_sample,batch_size\nmistral-7b-v0.1,full,NVIDIA-A100-SXM4-80GB,1024,8\n"
    resp = client.post("/api/recommend/csv", json={"csv": csv_in, "predictor": "kavier"})
    assert resp.status_code == 422
    assert "bad batch input" in resp.json()["detail"]


def test_recommend_csv_invalid_goal_is_isolated_not_500(client):
    """An unknown goal is isolated per-row by the facade (feasible=False + the goal
    error in the row), so the endpoint returns 200 with a failed row — it does NOT
    500 or silently drop the row. This pins the documented isolation contract."""
    csv_in = "model,method,gpu_model,tokens_per_sample,batch_size\nmistral-7b-v0.1,full,NVIDIA-A100-SXM4-80GB,1024,8\n"
    resp = client.post(
        "/api/recommend/csv",
        json={"csv": csv_in, "predictor": "kavier", "goal": "no-such-goal"},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["count"] == 1
    # The failed row is preserved in the output CSV, carrying the goal error.
    assert "unknown goal" in payload["csv"]
    assert "False" in payload["csv"]  # feasible=False column


# --------------------------------------------------------------------------- #
# /api/recommend/batch — over the 200-workload cap (422)
# --------------------------------------------------------------------------- #


def test_recommend_batch_over_workload_cap_is_422(client):
    """More than COASTLINE_MAX_BATCH_WORKLOADS (default 200) workloads trips the
    pydantic max_length on the request body -> 422 (never reaching the recommender)."""
    one = {
        "model": "mistral-7b-v0.1",
        "method": "full",
        "gpu_model": "NVIDIA-A100-SXM4-80GB",
        "tokens_per_sample": 1024,
        "batch_size": 8,
    }
    body = {"workloads": [dict(one) for _ in range(main._MAX_BATCH_WORKLOADS + 1)], "predictor": "kavier"}
    resp = client.post("/api/recommend/batch", json=body)
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# /api/recommend happy path (Kavier-only: no ML pickle loading)
# --------------------------------------------------------------------------- #


def test_recommend_kavier_returns_candidates(client):
    resp = client.post("/api/recommend", json=_KAVIER_BODY)
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["success"] is True
    assert payload["strategy"] == "multi_objective"
    assert payload["preset"] == "balanced"

    candidates = payload["candidates"]
    assert isinstance(candidates, list) and len(candidates) >= 1
    # The top recommendation is the first candidate.
    assert payload["recommendation"] == candidates[0]

    top = payload["recommendation"]
    # Serialized candidate contract from _serialize_candidate.
    assert set(top) >= {
        "rank",
        "total_gpus",
        "predicted_throughput",
    }
    assert top["rank"] == 1
    # Kavier produced a positive throughput for this supported workload.
    assert top["predicted_throughput"] is not None
    assert top["predicted_throughput"] > 0
    # The recommended GPU count respects the requested budget.
    assert 1 <= top["total_gpus"] <= _KAVIER_BODY["total_gpus"]

    # workload_summary echoes the request inputs.
    summary = payload["workload_summary"]
    assert summary["llm_model"] == _KAVIER_BODY["llm_model"]
    assert summary["gpu_model"] == _KAVIER_BODY["gpu_model"]
    assert summary["tokens_per_sample"] == _KAVIER_BODY["tokens_per_sample"]


def test_recommend_unknown_gpu_returns_404_not_500(client):
    """POST /api/recommend with an unknown gpu_model must return 404, not 500.

    Regression for UnsupportedGPUError (a RecommenderSystemError subclass) not
    being caught by the handler's ValueError/RuntimeError arms, so it fell through
    to the generic except-Exception->500 branch instead of the intended 404."""
    body = {**_KAVIER_BODY, "gpu_model": "NOT-A-REAL-GPU-XYZ"}
    resp = client.post("/api/recommend", json=body)
    assert resp.status_code == 404, f"Unknown GPU should be 404, got {resp.status_code}: {resp.text}"


def test_recommend_min_gpu_strategy_kavier(client):
    """The min_gpu strategy also runs Kavier-only end-to-end (preset is dropped)."""
    body = {**_KAVIER_BODY, "strategy": "min_gpu"}
    resp = client.post("/api/recommend", json=body)
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["success"] is True
    assert payload["strategy"] == "min_gpu"
    # min_gpu does not carry a preset in the response.
    assert payload["preset"] is None
    assert len(payload["candidates"]) >= 1


# --------------------------------------------------------------------------- #
# _load_strategy_config fallback order
# --------------------------------------------------------------------------- #


def test_load_strategy_config_uses_repo_experiment_yaml(clean_strategy_env):
    """With the real repo layout and no env overrides, config/coastline_functionality/experiment.yaml wins.

    This guards the recent move of the strategy config to
    config/coastline_functionality/experiment.yaml (ahead of
    config/coastline_functionality/default.yaml in the candidate list).
    """
    experiment = main._REPO_ROOT / "config" / "coastline_functionality" / "experiment.yaml"
    default = main._REPO_ROOT / "config" / "coastline_functionality" / "default.yaml"
    assert experiment.is_file(), f"expected {experiment} to exist"
    assert default.is_file(), f"expected {default} to exist"

    config = _load_strategy_config()
    # experiment.yaml declares multi_objective/balanced; default.yaml declares min_gpu.
    assert config["strategy"]["name"] == "multi_objective"
    assert config["strategy"]["preset"] == "balanced"
    # experiment.yaml has an explicit predictors block (not the orchestrator shim).
    assert config["predictors"]["performance"] == "intelligent"


def test_load_strategy_config_falls_back_to_default_yaml(tmp_path, monkeypatch, clean_strategy_env):
    """When experiment.yaml is absent, the loader uses config/coastline_functionality/default.yaml.

    default.yaml uses the new ``predictors:`` schema directly; the legacy
    ``orchestrator:`` translation is covered by
    coastline_recommender/tests/test_run_config.py.
    """
    cfg_dir = tmp_path / "config" / "coastline_functionality"
    cfg_dir.mkdir(parents=True)
    shutil.copy(
        main._REPO_ROOT / "config" / "coastline_functionality" / "default.yaml",
        cfg_dir / "default.yaml",
    )
    # No experiment.yaml in this fake repo root.
    assert not (cfg_dir / "experiment.yaml").exists()

    monkeypatch.setattr(main, "_REPO_ROOT", tmp_path)
    config = _load_strategy_config()

    # default.yaml's strategy is min_gpu (distinguishes it from the built-in default).
    assert config["strategy"]["name"] == "min_gpu"
    # default.yaml declares predictors.performance "intelligent".
    assert config["predictors"]["performance"] == "intelligent"
    # grid section came from default.yaml.
    assert config["grid"]["total_gpus"] == [1, 2, 4, 8, 16]


def test_load_strategy_config_env_var_takes_precedence(tmp_path, monkeypatch, clean_strategy_env):
    """An explicit STRATEGY_CONFIG path overrides the repo config files."""
    custom = tmp_path / "custom.yaml"
    custom.write_text("strategy:\n  name: min_gpu\n  preset: energy\n", encoding="utf-8")
    monkeypatch.setenv("STRATEGY_CONFIG", str(custom))

    config = _load_strategy_config()
    assert config["strategy"]["name"] == "min_gpu"
    assert config["strategy"]["preset"] == "energy"


def test_load_strategy_config_builtin_default_when_no_files(tmp_path, monkeypatch, clean_strategy_env):
    """With no env override and no config files, the built-in default is returned."""
    empty_cfg = tmp_path / "config"
    empty_cfg.mkdir()  # exists but contains no yaml files
    monkeypatch.setattr(main, "_REPO_ROOT", tmp_path)

    config = _load_strategy_config()
    assert config == _DEFAULT_STRATEGY_CONFIG
    assert config["strategy"]["name"] == "multi_objective"


# ---------------------------------------------------------------------------
# Infrastructure (component F) — sysadmin cap + UI feed + enforcement
# ---------------------------------------------------------------------------


def test_infrastructure_endpoint_serves_cluster_cap():
    """GET /api/infrastructure returns the sysadmin-declared cap.

    Assert invariants rather than the sysadmin's exact numbers (which change when
    the cluster is re-sized): a cluster can never advertise more GPUs than its
    nodes × per-node capacity can physically hold, and the GPU the recommend/queue
    tests drive (A100-SXM4-80GB) must be in the advertised catalog or those tests
    would be exercising a GPU the cluster claims not to have.
    """
    with TestClient(main.app) as client:
        resp = client.get("/api/infrastructure")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        # Physical capacity invariant: total budget fits in nodes × GPUs/node.
        assert data["total_gpus"] <= data["max_nodes"] * data["max_gpus_per_node"]
        # The queue/recommend tests all pin this GPU; it must be a real cluster type.
        assert "NVIDIA-A100-SXM4-80GB" in data["gpu_models"]


def test_recommend_rejects_request_beyond_cluster_cap():
    """A request beyond the advertised GPU cap is rejected with 400 (not silently capped)."""
    with TestClient(main.app) as client:
        infra = client.get("/api/infrastructure").json()
        too_many = infra["total_gpus"] + 1
        resp = client.post(
            "/api/recommend",
            json={
                "llm_model": "mistral-7b-v0.1",
                "fine_tuning_method": "lora",
                "gpu_model": "NVIDIA-A100-SXM4-80GB",
                "tokens_per_sample": 1024,
                "batch_size": 8,
                "hardware_mode": "total",
                "total_gpus": too_many,
                "prediction_model": "kavier",
            },
        )
        assert resp.status_code == 400
        assert "exceeds" in (resp.json().get("detail", "") or "").lower()


# ---------------------------------------------------------------------------
# Workload queue + admin (component I — FIFO scheduler harness)
# ---------------------------------------------------------------------------
from coastline.ui import workload_queue as _wq  # noqa: E402


@pytest.fixture(autouse=False)
def _clear_queue():
    _wq.clear_jobs()
    yield
    _wq.clear_jobs()


def test_queue_add_list_remove_round_trip(_clear_queue):
    with TestClient(main.app) as client:
        assert client.get("/api/queue").json()["jobs"] == []
        r = client.post("/api/queue", json={"num_gpus": 4, "predicted_duration_s": 60.0, "llm_model": "test"})
        assert r.status_code == 200
        rid = r.json()["job"]["request_id"]
        jobs = client.get("/api/queue").json()["jobs"]
        assert len(jobs) == 1 and jobs[0]["request_id"] == rid
        assert client.delete(f"/api/queue/{rid}").status_code == 200
        assert client.get("/api/queue").json()["jobs"] == []


def test_queue_rejects_request_beyond_cluster_cap(_clear_queue):
    with TestClient(main.app) as client:
        infra = client.get("/api/infrastructure").json()
        r = client.post("/api/queue", json={"num_gpus": infra["total_gpus"] + 1, "predicted_duration_s": 60})
        assert r.status_code == 400
        assert "exceeds" in (r.json().get("detail", "") or "").lower()


def test_admin_run_returns_per_job_and_total_metrics(_clear_queue):
    with TestClient(main.app) as client:
        # Two 4-GPU jobs. The advertised cluster is 32 GPUs (>= 8), so both co-run
        # from t=0 and the makespan is exactly the longer job's duration.
        client.post("/api/queue", json={"num_gpus": 4, "predicted_duration_s": 10.0})
        client.post("/api/queue", json={"num_gpus": 4, "predicted_duration_s": 5.0})
        r = client.post("/api/admin/run")
        assert r.status_code == 200
        data = r.json()
        assert data["totals"]["n_jobs"] == 2
        assert len(data["jobs"]) == 2
        # Both fit simultaneously -> makespan = max(10, 5) = 10.
        assert data["totals"]["makespan_s"] == pytest.approx(10.0)
        # Energy is scheduling-independent (sum over jobs). Neither job carried a
        # Kavier power, so each uses the 350 W/GPU fallback. By hand:
        #   (350*4*10 + 350*4*5) W·s = 14000 + 7000 = 21000 W·s
        #   21000 / 3_600_000 = 0.0058333.. kWh
        assert data["totals"]["total_energy_kwh"] == pytest.approx(21000 / 3_600_000.0, rel=1e-9)


def test_admin_run_on_empty_queue_is_a_no_op():
    _wq.clear_jobs()
    with TestClient(main.app) as client:
        r = client.post("/api/admin/run")
        assert r.status_code == 200
        assert r.json()["totals"] is None
        assert r.json()["jobs"] == []


def test_admin_import_csv_accepts_operation_simulator_schema(_clear_queue):
    csv_text = "submission_time,user_id,num_gpus,duration_ms\n0,u1,4,5000\n2,u1,8,1000\n"
    with TestClient(main.app) as client:
        r = client.post("/api/admin/import", json={"csv": csv_text})
        assert r.status_code == 200
        assert r.json()["imported"] == 2
        jobs = client.get("/api/queue").json()["jobs"]
        assert len(jobs) == 2
        # duration_ms=5000 -> predicted_duration_s=5.0
        assert any(abs(j["predicted_duration_s"] - 5.0) < 1e-6 for j in jobs)


def test_admin_import_rejects_malformed_csv(_clear_queue):
    bad = "foo,bar\n1,2\n"
    with TestClient(main.app) as client:
        r = client.post("/api/admin/import", json={"csv": bad})
        assert r.status_code == 400
        assert "missing" in r.json().get("detail", "").lower()


# ---------------------------------------------------------------------------
# Kavier-power wiring for queue jobs + defensive oversized-job guard
# ---------------------------------------------------------------------------


def test_queue_add_with_full_workload_config_attaches_kavier_power(_clear_queue):
    """When all of model/method/gpu/tokens/batch are supplied, the backend queries
    Kavier and stores the per-GPU power on the job. The simulator then uses it for
    the cluster-wide energy summary instead of the coarse placeholder."""
    with TestClient(main.app) as client:
        r = client.post(
            "/api/queue",
            json={
                "num_gpus": 4,
                "predicted_duration_s": 100.0,
                "llm_model": "mistral-7b-v0.1",
                "fine_tuning_method": "lora",
                "gpu_model": "NVIDIA-A100-SXM4-80GB",
                "tokens_per_sample": 2048,
                "batch_size": 8,
            },
        )
        assert r.status_code == 200
        job = r.json()["job"]
        p = job["predicted_power_watts_per_gpu"]
        assert p is not None, "Kavier per-GPU power should be populated"
        # Loose physical bound: idle (~75 W) ≤ per-GPU draw ≤ TDP (~500 W).
        assert 50 <= p <= 500, f"unexpected per-GPU power {p}"


def test_queue_add_without_workload_config_uses_simulator_fallback(_clear_queue):
    """A minimal job (only num_gpus + duration) carries no Kavier power; the
    simulator falls back to the cluster-average constant (350 W/GPU) so the
    total-energy summary stays meaningful."""
    with TestClient(main.app) as client:
        r = client.post("/api/queue", json={"num_gpus": 2, "predicted_duration_s": 10.0})
        assert r.status_code == 200
        assert r.json()["job"]["predicted_power_watts_per_gpu"] is None
        run = client.post("/api/admin/run").json()
        # Fallback: 2 GPU × 350 W × 10 s = 7000 J -> 7000 / 3.6M kWh.
        assert run["totals"]["total_energy_kwh"] == pytest.approx(7000 / 3_600_000.0, rel=1e-3)


def test_admin_run_uses_kavier_power_when_present(_clear_queue):
    """Kavier per-GPU power on a job overrides the cluster-average constant in
    the total-energy summary — energy = power × num_gpus × duration."""
    with TestClient(main.app) as client:
        client.post(
            "/api/queue",
            json={
                "num_gpus": 1,
                "predicted_duration_s": 100.0,
                "llm_model": "mistral-7b-v0.1",
                "fine_tuning_method": "lora",
                "gpu_model": "NVIDIA-A100-SXM4-80GB",
                "tokens_per_sample": 2048,
                "batch_size": 8,
            },
        )
        jobs = client.get("/api/queue").json()["jobs"]
        per_gpu = jobs[0]["predicted_power_watts_per_gpu"]
        assert per_gpu is not None and per_gpu > 0
        run = client.post("/api/admin/run").json()
        expected_kwh = (per_gpu * 1 * 100.0) / 3_600_000.0
        assert run["totals"]["total_energy_kwh"] == pytest.approx(expected_kwh, rel=1e-3)


def test_dashboard_route_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_admin_clear_empties_the_queue(_clear_queue):
    with TestClient(main.app) as client:
        client.post("/api/queue", json={"num_gpus": 4, "predicted_duration_s": 30.0})
        assert len(client.get("/api/queue").json()["jobs"]) == 1
        resp = client.post("/api/admin/clear")
        assert resp.status_code == 200 and resp.json()["success"] is True
        assert client.get("/api/queue").json()["jobs"] == []


def test_predict_runs_kavier_in_a_subprocess(client):
    body = {
        "llm_model": "mistral-7b-v0.1",
        "fine_tuning_method": "full",
        "gpu_model": "NVIDIA-A100-SXM4-80GB",
        "tokens_per_sample": 1024,
        "batch_size": 8,
        "models": ["kavier"],
    }
    resp = client.post("/api/predict", json=body)
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["success"] is True
    results = payload["results"]
    assert len(results) == 1
    # kavier is analytical (no pickle) -> available, and the worker only reports
    # available=True after asserting predicted_throughput > 0, so a finite positive
    # throughput must be present for this supported (model, GPU, method).
    assert results[0]["available"] is True
    assert results[0]["predicted_throughput"] > 0
    # Runtime is derived as total_tokens / throughput; cross-check the two reported
    # fields agree. total_tokens = dataset_size(10000) * epochs(3) * tokens(1024).
    total_tokens = 10000 * 3 * 1024
    assert results[0]["predicted_runtime_seconds"] == pytest.approx(
        total_tokens / results[0]["predicted_throughput"], rel=1e-6
    )


def test_queue_add_with_explicit_power_override_skips_kavier(_clear_queue):
    """``predicted_power_watts_per_gpu`` in the request takes precedence over the
    Kavier lookup (so a CSV-imported job's power, or a synthetic test value, is
    honoured verbatim)."""
    with TestClient(main.app) as client:
        r = client.post(
            "/api/queue",
            json={
                "num_gpus": 1,
                "predicted_duration_s": 10.0,
                "predicted_power_watts_per_gpu": 123.0,
                "llm_model": "mistral-7b-v0.1",
                "fine_tuning_method": "lora",
                "gpu_model": "NVIDIA-A100-SXM4-80GB",
                "tokens_per_sample": 2048,
                "batch_size": 8,
            },
        )
        assert r.json()["job"]["predicted_power_watts_per_gpu"] == 123.0


def test_simulate_fifo_skips_oversized_jobs_defensively():
    """A job needing more GPUs than the cluster has would block strict FIFO
    forever; simulate_fifo defensively drops it and keeps the rest of the queue
    schedulable."""
    from coastline.ui import workload_queue as wq

    jobs = [
        wq.QueueJob(request_id="ok", arrival_time=0.0, num_gpus=2, predicted_duration_s=1.0),
        wq.QueueJob(request_id="huge", arrival_time=0.0, num_gpus=999, predicted_duration_s=1.0),
    ]
    result = wq.simulate_fifo(jobs, n_gpus_cluster=4)
    assert result.n_jobs == 1
    assert result.jobs[0].request_id == "ok"


def test_simulate_fifo_head_of_line_blocking_exact_schedule():
    """Two 4-GPU jobs on a 4-GPU cluster cannot co-run: strict FIFO serialises
    them and the second waits for the first. Pin the exact deterministic
    timeline so a regression in the head-of-line dispatch math is caught.

      j1: arrival 0, dur 10  -> [0, 10], wait 0
      j2: arrival 0, dur 5   -> [10, 15], wait 10  (blocked behind j1)
      makespan = 15, avg_wait = (0 + 10) / 2 = 5, avg_jct = (10 + 15) / 2 = 12.5
    """
    from coastline.ui import workload_queue as wq

    jobs = [
        wq.QueueJob(request_id="j1", arrival_time=0.0, num_gpus=4, predicted_duration_s=10.0),
        wq.QueueJob(request_id="j2", arrival_time=0.0, num_gpus=4, predicted_duration_s=5.0),
    ]
    r = wq.simulate_fifo(jobs, n_gpus_cluster=4)
    assert r.n_jobs == 2
    assert r.makespan_s == pytest.approx(15.0)
    by_id = {j.request_id: j for j in r.jobs}
    assert by_id["j1"].start_time == pytest.approx(0.0)
    assert by_id["j1"].end_time == pytest.approx(10.0)
    assert by_id["j1"].wait_time_s == pytest.approx(0.0)
    assert by_id["j2"].start_time == pytest.approx(10.0)
    assert by_id["j2"].end_time == pytest.approx(15.0)
    assert by_id["j2"].wait_time_s == pytest.approx(10.0)
    assert r.avg_waiting_time_s == pytest.approx(5.0)
    assert r.avg_job_completion_time_s == pytest.approx(12.5)


def test_simulate_fifo_two_jobs_co_run_when_capacity_allows():
    """Same two jobs on an 8-GPU cluster fit simultaneously (no head-of-line
    blocking): both start at t=0, nobody waits, makespan is the longer job."""
    from coastline.ui import workload_queue as wq

    jobs = [
        wq.QueueJob(request_id="j1", arrival_time=0.0, num_gpus=4, predicted_duration_s=10.0),
        wq.QueueJob(request_id="j2", arrival_time=0.0, num_gpus=4, predicted_duration_s=5.0),
    ]
    r = wq.simulate_fifo(jobs, n_gpus_cluster=8)
    assert r.makespan_s == pytest.approx(10.0)
    assert r.avg_waiting_time_s == pytest.approx(0.0)
    by_id = {j.request_id: j for j in r.jobs}
    assert by_id["j1"].start_time == pytest.approx(0.0)
    assert by_id["j2"].start_time == pytest.approx(0.0)


def test_admin_import_accepts_power_column_in_csv(_clear_queue):
    """CSV import recognises an optional power_watts_per_gpu column and stores it."""
    csv_text = (
        "submission_time,num_gpus,duration_s,power_watts_per_gpu\n"
        "0,2,10,400\n"
        "5,1,5,\n"  # blank power → None (fallback)
    )
    with TestClient(main.app) as client:
        r = client.post("/api/admin/import", json={"csv": csv_text})
        assert r.status_code == 200
        assert r.json()["imported"] == 2
        jobs = client.get("/api/queue").json()["jobs"]
        powered = [j for j in jobs if j["predicted_power_watts_per_gpu"]]
        bare = [j for j in jobs if not j["predicted_power_watts_per_gpu"]]
        assert len(powered) == 1 and powered[0]["predicted_power_watts_per_gpu"] == 400.0
        assert len(bare) == 1


# ---------------------------------------------------------------------------
# QueueAddRequest schema — Kavier-duration path
# ---------------------------------------------------------------------------


def test_queue_add_request_dataset_size_accepted_and_required_positive():
    from pydantic import ValidationError

    from coastline.ui.app import QueueAddRequest

    req = QueueAddRequest(num_gpus=1, predicted_duration_s=10.0, dataset_size=1000)
    assert req.dataset_size == 1000
    with pytest.raises(ValidationError):
        QueueAddRequest(num_gpus=1, predicted_duration_s=10.0, dataset_size=0)


def test_queue_add_request_predicted_duration_is_optional():
    """Relaxed so a complete workload config can drive Kavier-duration prediction
    without the caller having to supply a placeholder number."""
    from coastline.ui.app import QueueAddRequest

    req = QueueAddRequest(num_gpus=1)
    assert req.predicted_duration_s is None


# ---------------------------------------------------------------------------
# _kavier_predict — unified helper returning (power, duration)
# ---------------------------------------------------------------------------


def test_kavier_predict_power_is_bounded_and_independent_of_duration_inputs():
    """Per-GPU power is a function of the workload + hardware only. Kavier is a
    black-box analytical engine, so we assert the physical envelope and an
    invariance property rather than pinning its exact watts (a snapshot).

    * dataset_size / training_epochs are *duration-only* inputs; adding them must
      leave the per-GPU power unchanged (the power leg never reads them).
    * power must sit in the A100's physical envelope: idle (~50 W) .. SXM4 TDP
      (400 W). This rejects a per-cluster-vs-per-GPU or a W-vs-mW unit regression
      even though the exact value is engine-internal.
    """
    from coastline.ui.app import _kavier_predict

    base = _kavier_predict(  # only the 5 base fields -> power, but no duration
        model="mistral-7b-v0.1",
        method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=8,
        num_gpus=4,
    )
    full = _kavier_predict(  # + duration inputs
        model="mistral-7b-v0.1",
        method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=8,
        dataset_size=10000,
        training_epochs=3,
        num_gpus=4,
    )
    # Invariance: duration inputs do not perturb the power leg.
    assert base.power_watts_per_gpu == pytest.approx(full.power_watts_per_gpu, rel=1e-9)
    # Physical envelope for one A100 GPU (idle .. TDP), not a magic snapshot.
    assert 50.0 <= full.power_watts_per_gpu <= 500.0
    # Duration requires dataset_size + epochs; the base-only call cannot produce one.
    assert base.duration_seconds is None
    assert full.duration_seconds is not None and full.duration_seconds > 0


def test_kavier_predict_duration_scales_linearly_with_epochs_and_dataset():
    """Duration = dataset_size × training_epochs × tokens_per_sample / throughput,
    and throughput depends only on (model, gpu, method, tokens, batch, num_gpus) —
    never on dataset_size or training_epochs. So the two multiplicative token
    factors scale duration *exactly* linearly. This pins the runtime formula's
    units without snapshotting Kavier's raw throughput.
    """
    from coastline.ui.app import _kavier_predict

    def dur(dataset_size, epochs):
        return _kavier_predict(
            model="mistral-7b-v0.1",
            method="lora",
            gpu_model="NVIDIA-A100-SXM4-80GB",
            tokens_per_sample=1024,
            batch_size=8,
            dataset_size=dataset_size,
            training_epochs=epochs,
            num_gpus=4,
        ).duration_seconds

    base = dur(10000, 3)
    assert base is not None and base > 0
    # 2× epochs (throughput unchanged) -> exactly 2× total tokens -> 2× duration.
    assert dur(10000, 6) == pytest.approx(2.0 * base, rel=1e-9)
    # 3× dataset_size -> exactly 3× duration, same reasoning.
    assert dur(30000, 3) == pytest.approx(3.0 * base, rel=1e-9)


def test_kavier_predict_returns_empty_when_base_field_missing():
    """Missing any of the 5 base fields → both outputs None, no exception."""
    from coastline.ui.app import _kavier_predict

    est = _kavier_predict(
        model=None,
        method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=8,
        num_gpus=4,
    )
    assert est.power_watts_per_gpu is None
    assert est.duration_seconds is None


def test_kavier_predict_returns_empty_for_unsupported_model():
    """A model Kavier doesn't have in its library returns the empty estimate,
    not an exception."""
    from coastline.ui.app import _kavier_predict

    est = _kavier_predict(
        model="not-a-real-model",
        method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=8,
        dataset_size=10000,
        training_epochs=3,
        num_gpus=4,
    )
    assert est.power_watts_per_gpu is None
    assert est.duration_seconds is None


# ---------------------------------------------------------------------------
# /api/queue — Kavier-duration overwrite + user/Kavier resolution + 422
# ---------------------------------------------------------------------------


def test_queue_add_with_complete_config_overwrites_duration_with_kavier(_clear_queue):
    """Full workload config → Kavier wins; the wrong user-supplied duration is
    overwritten and the response advertises duration_source == 'kavier'."""
    body = {
        "num_gpus": 4,
        "predicted_duration_s": 1.0,  # deliberately wrong
        "llm_model": "mistral-7b-v0.1",
        "fine_tuning_method": "lora",
        "gpu_model": "NVIDIA-A100-SXM4-80GB",
        "tokens_per_sample": 1024,
        "batch_size": 8,
        "dataset_size": 10000,
        "training_epochs": 3,
    }
    with TestClient(main.app) as client:
        r = client.post("/api/queue", json=body)
        assert r.status_code == 200, r.text
        payload = r.json()
        job = payload["job"]
        assert payload["duration_source"] == "kavier"
        assert job["predicted_duration_s"] != 1.0
        assert job["predicted_duration_s"] > 0


def test_queue_add_without_complete_config_keeps_user_duration(_clear_queue):
    """Without a complete config the user-supplied duration is honoured
    verbatim and duration_source == 'user'."""
    with TestClient(main.app) as client:
        r = client.post("/api/queue", json={"num_gpus": 2, "predicted_duration_s": 30.0})
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["duration_source"] == "user"
        assert payload["job"]["predicted_duration_s"] == 30.0


def test_queue_add_rejects_when_neither_duration_nor_config(_clear_queue):
    """A bare body with no duration and no workload-config is a 422."""
    with TestClient(main.app) as client:
        r = client.post("/api/queue", json={"num_gpus": 2})
        assert r.status_code == 422, r.text


# ---------------------------------------------------------------------------
# QueueJob — optional Kavier-config fields
# ---------------------------------------------------------------------------


def test_queue_job_accepts_optional_kavier_config_fields():
    """gpu_model / tokens_per_sample / dataset_size / gpus_per_node /
    number_of_nodes are all optional and default to None on the model."""
    from coastline.ui.workload_queue import QueueJob

    j = QueueJob(
        request_id="x",
        arrival_time=0.0,
        num_gpus=1,
        predicted_duration_s=1.0,
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        dataset_size=10000,
        gpus_per_node=8,
        number_of_nodes=1,
    )
    assert j.gpu_model == "NVIDIA-A100-SXM4-80GB"
    assert j.tokens_per_sample == 1024
    assert j.dataset_size == 10000
    assert j.gpus_per_node == 8
    assert j.number_of_nodes == 1
    bare = QueueJob(request_id="y", arrival_time=0.0, num_gpus=1, predicted_duration_s=1.0)
    assert bare.gpu_model is None
    assert bare.tokens_per_sample is None
    assert bare.dataset_size is None


def test_queue_add_persists_workload_config_fields_on_queuejob(_clear_queue):
    """The /api/queue handler should store the user-supplied workload-config
    fields on the QueueJob, not just consume them for the Kavier lookup. A
    subsequent GET /api/queue returning gpu_model: null after the caller
    supplied one would be surprising."""
    with TestClient(main.app) as client:
        client.post(
            "/api/queue",
            json={
                "num_gpus": 4,
                "predicted_duration_s": 60.0,  # bare config so user wins
                "llm_model": "mistral-7b-v0.1",
                "fine_tuning_method": "lora",
                "gpu_model": "NVIDIA-A100-SXM4-80GB",
                "tokens_per_sample": 1024,
                "batch_size": 8,
                "dataset_size": 10000,
                "training_epochs": 3,
                "gpus_per_node": 4,
                "number_of_nodes": 1,
            },
        )
        jobs = client.get("/api/queue").json()["jobs"]
        assert len(jobs) == 1
        j = jobs[0]
        assert j["gpu_model"] == "NVIDIA-A100-SXM4-80GB"
        assert j["tokens_per_sample"] == 1024
        assert j["dataset_size"] == 10000
        assert j["gpus_per_node"] == 4
        assert j["number_of_nodes"] == 1


def test_parse_csv_skips_zero_in_gt_zero_fields():
    """A literal '0' in tokens_per_sample/dataset_size/epochs would otherwise
    trip the gt=0 Pydantic constraint and 500 the whole import. _maybe_int
    treats non-positive integers as missing so the row imports cleanly."""
    from coastline.ui.workload_queue import parse_csv

    csv_text = "submission_time,num_gpus,duration_s,tokens_per_sample,dataset_size,num_train_epochs\n0,1,5,0,0,0\n"
    jobs = parse_csv(csv_text)
    assert len(jobs) == 1
    j = jobs[0]
    assert j.tokens_per_sample is None
    assert j.dataset_size is None
    assert j.training_epochs is None


# ---------------------------------------------------------------------------
# parse_csv — standard trace column aliases
# ---------------------------------------------------------------------------


def test_parse_csv_accepts_wt2_aliases():
    """Standard trace header columns import straight through, with all the
    optional workload-config fields populated on the parsed QueueJob."""
    from coastline.ui.workload_queue import parse_csv

    csv_text = (
        "run_hash,submission_time,duration_ms,user_id,num_gpus,model_name,method,"
        "gpu_model,number_nodes,num_gpus_per_node,batch_size,tokens_per_sample,"
        "num_train_epochs\n"
        "abc123,0,5000,u1,4,mistral-7b-v0.1,lora,NVIDIA-A100-SXM4-80GB,1,4,8,1024,3\n"
    )
    jobs = parse_csv(csv_text)
    assert len(jobs) == 1
    j = jobs[0]
    assert j.arrival_time == 0.0
    assert j.num_gpus == 4
    assert j.predicted_duration_s == 5.0
    assert j.llm_model == "mistral-7b-v0.1"
    assert j.fine_tuning_method == "lora"
    assert j.gpu_model == "NVIDIA-A100-SXM4-80GB"
    assert j.number_of_nodes == 1
    assert j.gpus_per_node == 4
    assert j.batch_size == 8
    assert j.tokens_per_sample == 1024
    assert j.training_epochs == 3


def test_parse_csv_skips_rows_with_blank_or_unparseable_duration():
    """Rows with an empty duration_ms (jobs that never completed) are silently
    skipped by the importer rather than 400-ing the whole batch."""
    from coastline.ui.workload_queue import parse_csv

    csv_text = (
        "submission_time,num_gpus,duration_ms\n"
        "0,1,5000\n"  # valid
        "1,1,\n"  # blank
        "2,1,not-a-number\n"  # unparseable
        "3,1,3000\n"  # valid
    )
    jobs = parse_csv(csv_text)
    assert len(jobs) == 2
    assert [j.predicted_duration_s for j in jobs] == [5.0, 3.0]


# ---------------------------------------------------------------------------
# /api/admin/import — Kavier power on imported jobs + predict_durations flag
# ---------------------------------------------------------------------------


def test_admin_import_attaches_kavier_power_to_supported_rows(_clear_queue):
    """Closes the prior gap where bulk-imported jobs never had Kavier power
    populated, even when the row's workload config was complete. After import
    the queued job carries a per-GPU power."""
    csv_text = (
        "submission_time,num_gpus,duration_ms,model_name,method,gpu_model,"
        "tokens_per_sample,batch_size\n"
        "0,1,5000,mistral-7b-v0.1,lora,NVIDIA-A100-SXM4-80GB,1024,8\n"
    )
    with TestClient(main.app) as client:
        r = client.post("/api/admin/import", json={"csv": csv_text})
        assert r.status_code == 200, r.text
        jobs = client.get("/api/queue").json()["jobs"]
        assert len(jobs) == 1
        p = jobs[0]["predicted_power_watts_per_gpu"]
        assert p is not None and 50 <= p <= 500


def test_admin_import_predict_durations_overrides_csv_durations(_clear_queue):
    """With predict_durations=true the importer replaces CSV durations on rows
    whose workload config is complete, and leaves rows that don't qualify
    alone (e.g. an unsupported model). The response also carries a count of
    how many rows were Kavier-substituted."""
    csv_text = (
        "submission_time,num_gpus,duration_ms,model_name,method,gpu_model,"
        "tokens_per_sample,batch_size,dataset_size,num_train_epochs\n"
        # Supported: should get Kavier duration, NOT 1 ms.
        "0,1,1,mistral-7b-v0.1,lora,NVIDIA-A100-SXM4-80GB,1024,8,10000,3\n"
        # Unsupported model: should keep the 7000 ms CSV duration.
        "1,1,7000,not-a-real-model,lora,NVIDIA-A100-SXM4-80GB,1024,8,10000,3\n"
    )
    with TestClient(main.app) as client:
        r = client.post(
            "/api/admin/import",
            json={"csv": csv_text, "predict_durations": True},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["imported"] == 2
        assert body["durations_predicted"] == 1
        jobs = client.get("/api/queue").json()["jobs"]
        assert len(jobs) == 2
        supported = next(j for j in jobs if j["llm_model"] == "mistral-7b-v0.1")
        unsupported = next(j for j in jobs if j["llm_model"] == "not-a-real-model")
        # Supported row: Kavier overwrote the 1 ms CSV value with something >> 1 ms.
        assert supported["predicted_duration_s"] > 0.001 + 1.0
        # Unsupported row: CSV value preserved.
        assert unsupported["predicted_duration_s"] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Cluster timeline (the Exp2/Exp4 figure data) — build_cluster_timeline + the
# `timeline` block carried on /api/admin/run.
# ---------------------------------------------------------------------------


def test_build_cluster_timeline_concurrent_jobs():
    """Two jobs that fit side by side: GPUs ramp to their sum, nothing queues,
    and the series spans [0, makespan] and drains back to 0."""
    from coastline.ui import workload_queue as wq

    jobs = [
        wq.QueueJob(request_id="a", arrival_time=0.0, num_gpus=2, predicted_duration_s=10.0),
        wq.QueueJob(request_id="b", arrival_time=0.0, num_gpus=2, predicted_duration_s=5.0),
    ]
    sim = wq.simulate_fifo(jobs, n_gpus_cluster=8)
    tl = wq.build_cluster_timeline(sim.jobs, 8)
    assert tl.cluster_gpus == 8
    assert tl.peak_gpus == 4  # 2 + 2 run together
    assert tl.peak_queue == 0  # both fit immediately
    assert tl.t[0] == 0.0
    assert tl.makespan_s == pytest.approx(10.0)
    assert tl.t[-1] == pytest.approx(10.0)
    assert tl.gpus_used[-1] == 0  # cluster drains
    # Equal-length, time-sorted series; never over the cap.
    assert len(tl.t) == len(tl.gpus_used) == len(tl.queue_depth)
    assert tl.t == sorted(tl.t)
    assert max(tl.gpus_used) <= tl.cluster_gpus


def test_build_cluster_timeline_queue_buildup():
    """A job that saturates the cluster forces the next arrival to wait: queue
    depth rises to 1, then drains once the first job ends and the second runs."""
    from coastline.ui import workload_queue as wq

    jobs = [
        wq.QueueJob(request_id="big", arrival_time=0.0, num_gpus=4, predicted_duration_s=10.0),
        wq.QueueJob(request_id="wait", arrival_time=0.0, num_gpus=4, predicted_duration_s=5.0),
    ]
    sim = wq.simulate_fifo(jobs, n_gpus_cluster=4)
    tl = wq.build_cluster_timeline(sim.jobs, 4)
    assert tl.peak_gpus == 4  # FIFO never over-allocates
    assert tl.peak_queue == 1  # the second job waits
    assert tl.queue_depth[0] == 1  # waiting at t=0 while "big" runs
    assert tl.queue_depth[-1] == 0  # drained by the end
    assert tl.gpus_used[-1] == 0
    assert tl.makespan_s == pytest.approx(15.0)  # 10 then 5 back-to-back


def test_build_cluster_timeline_empty_is_zeroed():
    from coastline.ui import workload_queue as wq

    tl = wq.build_cluster_timeline([], 8)
    assert tl.t == [] and tl.gpus_used == [] and tl.queue_depth == []
    assert tl.peak_gpus == 0 and tl.peak_queue == 0
    assert tl.cluster_gpus == 8


def test_admin_run_includes_cluster_timeline(_clear_queue):
    """/api/admin/run carries a `timeline` block (the cluster-figure series).
    Arrays are equal-length, start at t=0, agree with the totals, and respect
    the cluster cap."""
    with TestClient(main.app) as client:
        client.post("/api/queue", json={"num_gpus": 4, "predicted_duration_s": 10.0})
        client.post("/api/queue", json={"num_gpus": 4, "predicted_duration_s": 5.0})
        data = client.post("/api/admin/run").json()
        tl = data["timeline"]
        assert set(tl) >= {
            "t",
            "gpus_used",
            "queue_depth",
            "cluster_gpus",
            "makespan_s",
            "peak_gpus",
            "peak_queue",
        }
        assert len(tl["t"]) == len(tl["gpus_used"]) == len(tl["queue_depth"]) >= 2
        assert tl["t"][0] == 0.0
        assert tl["cluster_gpus"] == data["totals"]["cluster_gpus"]
        assert tl["makespan_s"] == pytest.approx(data["totals"]["makespan_s"])
        assert tl["peak_gpus"] <= tl["cluster_gpus"]
        assert tl["gpus_used"][-1] == 0  # cluster drains by the end


def test_admin_run_empty_queue_omits_timeline():
    """No jobs → the early-return empty-run response carries no `timeline`
    (the dashboard guards on `totals` before drawing)."""
    _wq.clear_jobs()
    with TestClient(main.app) as client:
        data = client.post("/api/admin/run").json()
        assert data["totals"] is None
        assert "timeline" not in data
