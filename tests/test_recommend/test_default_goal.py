"""Every entry point has one default goal: performance (alpha 0.8 on runtime, beta 0.2 on energy).

The tests read each entry point's default where it is declared, and run the cheap ones with
Kavier and the rules feasibility check to see the default reach the strategy.
"""

from __future__ import annotations

import inspect
import json
from importlib import resources
from pathlib import Path

import pandas as pd
import pytest
import yaml

import coastline
from coastline.sdk.constants import DEFAULT_GOAL
from coastline.sdk.policies import PolicyFactory
from coastline.sdk.policies.multi_objective import MultiObjectiveStrategy
from coastline.sdk.recommend import engine
from coastline.sdk.recommend._goals import goal_to_label

GPU = "NVIDIA-A100-SXM4-80GB"
JOB = {
    "llm_model": "mistral-7b-v0.1",
    "fine_tuning_method": "lora",
    "gpu_model": GPU,
    "tokens_per_sample": 1024,
    "batch_size": 8,
}
_REPO = Path(__file__).resolve().parents[2]


def test_the_default_goal_is_performance():
    assert DEFAULT_GOAL == "performance"


def test_batch_api():
    assert inspect.signature(coastline.recommend).parameters["goal"].default == DEFAULT_GOAL
    kw = {"predictor": "kavier", "feasibility": "rules", "max_gpus": 8}
    default = coastline.recommend([JOB], **kw)
    performance = coastline.recommend([JOB], goal="performance", **kw)
    assert default.equals(performance)


def test_facade():
    recs = coastline.Coastline("kavier", feasibility="rules").recommend(JOB, max_gpus=8, top_k=1)
    assert recs[0].metadata["preset"] == "performance"
    assert (recs[0].metadata["alpha"], recs[0].metadata["beta"]) == (0.8, 0.2)


def test_explain(capsys):
    from coastline.cli import main

    main(
        ["explain", "--model", JOB["llm_model"], "--method", JOB["fine_tuning_method"], "--gpu-model", GPU]
        + ["--tokens", "1024", "--batch-size", "8", "--predictor", "kavier", "--feasibility", "rules"]
    )
    assert "preset=performance" in capsys.readouterr().out


def test_recommend_trace():
    from coastline.cli.recommend_trace import _build_parser
    from coastline.sdk.trace.recommend import recommend_trace

    args = _build_parser().parse_args(["--input", "in.csv", "--output", "out.csv"])
    assert args.goal == DEFAULT_GOAL
    assert inspect.signature(recommend_trace).parameters["goal"].default == DEFAULT_GOAL


def test_recommend_job_without_a_preset(tmp_path, capsys):
    from coastline.cli import main

    config = {
        "workload": dict(JOB),
        "strategy": {"name": "multi_objective"},
        "predictors": {"performance": "kavier", "energy": "kavier_power", "feasibility": "rules"},
        "grid": {"batch_sizes": [8], "total_gpus": [1, 8]},
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    main(["recommend-job", "--config", str(path), "--cluster-gpus", "8"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["metadata"]["preset"] == "performance"


def test_recommend_csv_without_a_preset(tmp_path, monkeypatch):
    from coastline.sdk.recommend import batch_csv

    built: list[str] = []
    build = batch_csv.engine.build_strategy

    def recording_build(*args, **kwargs):
        strategy = build(*args, **kwargs)
        built.append(strategy.get_name())
        return strategy

    monkeypatch.setattr(batch_csv.engine, "build_strategy", recording_build)
    config = {
        "strategy": {"name": "multi_objective"},
        "predictors": {"performance": "kavier", "energy": "kavier_power", "feasibility": "rules"},
        "grid": {"batch_sizes": [8], "total_gpus": [1, 8]},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    pd.DataFrame([JOB]).to_csv(tmp_path / "jobs.csv", index=False)

    coastline.recommend_csv(tmp_path / "config.yaml", tmp_path / "jobs.csv", tmp_path / "out.csv", cluster_gpus=8)

    assert built == ["multi_objective_performance"]
    assert bool(pd.read_csv(tmp_path / "out.csv").iloc[0]["feasible"])


class _NoPredictor:
    def predict(self, workload, context):  # pragma: no cover - never called
        return None

    def get_name(self) -> str:  # pragma: no cover - trivial
        return "none"


def test_policy_factory_and_strategy_without_a_preset():
    config = {
        "strategy": {"name": "multi_objective"},
        "predictors": {"performance": "kavier", "energy": "kavier_power", "feasibility": "rules"},
    }
    assert PolicyFactory.create_strategy(config=config).get_name() == "multi_objective_performance"
    built = MultiObjectiveStrategy(throughput_predictor=_NoPredictor(), power_predictor=_NoPredictor(), config=config)
    assert (built.preset, built.alpha, built.beta) == ("performance", 0.8, 0.2)


@pytest.mark.parametrize(
    "path",
    [
        _REPO / "src/coastline/sdk/io/default_experiment.yaml",
        _REPO / "config/coastline_functionality/experiment.yaml",
        _REPO / "config/batch_config.yaml",
        _REPO / "config/user_playground/demo.yaml",
    ],
)
def test_yaml_defaults(path):
    strategy = yaml.safe_load(path.read_text(encoding="utf-8"))["strategy"]
    assert strategy["name"] == "multi_objective"
    assert strategy["preset"] == DEFAULT_GOAL


def test_repl_defaults():
    answers = engine.defaults(engine.resolve_options())
    assert answers["goal_label"] == goal_to_label(DEFAULT_GOAL)


def test_dashboard_defaults():
    from coastline.ui.app import BatchRecommendRequest, RecommendCSVRequest, RecommendRequest

    assert RecommendRequest.model_fields["preset"].default == DEFAULT_GOAL
    assert BatchRecommendRequest.model_fields["goal"].default == DEFAULT_GOAL
    assert RecommendCSVRequest.model_fields["goal"].default == DEFAULT_GOAL
    page = (resources.files("coastline.ui") / "templates" / "index.html").read_text(encoding="utf-8")
    assert '<option value="performance" selected>' in page
    assert '<option value="balanced" selected>' not in page


def test_dashboard_recommend_without_a_preset():
    from fastapi.testclient import TestClient

    from coastline.ui.app import app

    body = {**JOB, "prediction_model": "kavier", "strategy": "multi_objective", "total_gpus": 8}
    with TestClient(app) as client:
        resp = client.post("/api/recommend", json=body)
    assert resp.status_code == 200, resp.text
    assert resp.json()["preset"] == "performance"


def test_bundled_configs_leave_top_k_unset():
    # Unset, multi_objective returns 5 configurations and min_gpu 1.
    for path in (
        _REPO / "src/coastline/sdk/io/default_experiment.yaml",
        _REPO / "config/coastline_functionality/experiment.yaml",
    ):
        grid = yaml.safe_load(path.read_text(encoding="utf-8"))["grid"]
        assert "top_k" not in grid, path
