"""The dashboard applies the policy file's `strategy.max_slowdown`, like the batch CSV path does.

`max_slowdown: 1.0` keeps only the fastest configuration, so under the energy preset the pick moves
from a small, low-power layout to the fastest one. Kavier and the `rules` checker need no AutoConf or
ML install. (min_gpu ignores max_slowdown; it picks by feasibility alone.)
"""

from __future__ import annotations

import pytest
import yaml
from fastapi.testclient import TestClient

import coastline.ui.app as app_module

_BODY = {
    "llm_model": "mistral-7b-v0.1",
    "fine_tuning_method": "lora",
    "gpu_model": "NVIDIA-A100-SXM4-80GB",
    "tokens_per_sample": 1024,
    "batch_size": 16,
    "hardware_mode": "total",
    "total_gpus": 8,
    "prediction_model": "kavier",
    "strategy": "multi_objective",
    "preset": "energy",
}


def _candidates(tmp_path, monkeypatch, strategy: dict) -> list[dict]:
    policy = tmp_path / f"experiment_{len(strategy)}.yaml"
    policy.write_text(
        yaml.safe_dump(
            {
                "strategy": strategy,
                "predictors": {"performance": "kavier", "energy": "kavier_power", "feasibility": "rules"},
                "grid": {"batch_sizes": [8, 16], "total_gpus": [1, 2, 4, 8], "top_k": 5},
            }
        )
    )
    monkeypatch.setenv("EXPERIMENT_CONFIG", str(policy))
    with TestClient(app_module.app) as client:
        resp = client.post("/api/recommend", json=_BODY)
    assert resp.status_code == 200, resp.text
    return resp.json()["candidates"]


def test_max_slowdown_in_the_policy_file_guards_the_dashboard(tmp_path, monkeypatch) -> None:
    unguarded = _candidates(tmp_path, monkeypatch, {"name": "multi_objective"})
    guarded = _candidates(tmp_path, monkeypatch, {"name": "multi_objective", "max_slowdown": 1.0})

    fastest = max(c["predicted_throughput"] for c in unguarded + guarded)
    assert all(c["predicted_throughput"] == pytest.approx(fastest) for c in guarded)
    assert guarded[0]["total_gpus"] > unguarded[0]["total_gpus"]
