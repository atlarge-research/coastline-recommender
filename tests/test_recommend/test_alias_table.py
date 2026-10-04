"""Goals and presets read one table of other spellings, and every entry point uses it.

The table is ``GOAL_ALIASES``. The presets use its entries for the multi-objective goals, and
"-", "_" and a space between words read alike. So 'energy_saver', 'energy-saver' and
'Energy Saver' name the energy goal and the energy preset, and 'min-gpu' names min_gpu. The entry
points: the Python API, the facade, recommend_csv, recommend-job, the recommend-trace CLI,
explain and the dashboard.
"""

from __future__ import annotations

import csv
import json

import pandas as pd
import pytest
import yaml

import coastline
from coastline.sdk.constants import GOAL_ALIASES, PRESET_ALIASES, PRESET_WEIGHTS, normalize_preset
from coastline.sdk.policies import PolicyFactory
from coastline.sdk.recommend._goals import GOALS, normalize_goal

GPU = "NVIDIA-A100-SXM4-80GB"
JOB = {
    "llm_model": "mistral-7b-v0.1",
    "fine_tuning_method": "lora",
    "gpu_model": GPU,
    "tokens_per_sample": 1024,
    "batch_size": 8,
}
ENERGY_SPELLINGS = ["energy_saver", "energy-saver", "Energy Saver", " ENERGY_SAVER "]
_PREDICTORS = {"performance": "kavier", "energy": "kavier_power", "feasibility": "rules"}


def test_every_alias_names_a_goal():
    assert set(GOAL_ALIASES.values()) <= set(GOALS)


def test_the_preset_aliases_are_the_multi_objective_goal_aliases():
    expected = {alias.replace("_", "-"): goal for alias, goal in GOAL_ALIASES.items() if goal in PRESET_WEIGHTS}
    assert PRESET_ALIASES == expected
    assert "energy-saver" in PRESET_ALIASES


@pytest.mark.parametrize("alias,goal", sorted(GOAL_ALIASES.items()))
def test_each_alias_works_as_goal_and_as_preset(alias, goal):
    for spelling in (alias, alias.replace("_", "-"), alias.replace("_", " ").title()):
        assert normalize_goal(spelling) == goal
        if goal in PRESET_WEIGHTS:
            assert normalize_preset(spelling) == goal
        else:
            with pytest.raises(ValueError, match="unknown preset"):
                normalize_preset(spelling)


@pytest.mark.parametrize("spelling", ["min-gpu", "Min GPU", "MIN_GPU"])
def test_min_gpu_spellings(spelling):
    assert normalize_goal(spelling) == "min_gpu"


@pytest.mark.parametrize("spelling", ["energy_saver-frontier", "Energy Saver Frontier", "PERFORMANCE_frontier"])
def test_a_frontier_preset_takes_the_aliases_too(spelling):
    base = "performance" if spelling.lower().startswith("performance") else "energy"
    assert normalize_preset(spelling) == f"{base}-frontier"


def test_unknown_names_list_the_choices():
    with pytest.raises(ValueError, match="unknown goal 'fast'; choose from"):
        normalize_goal("fast")
    with pytest.raises(ValueError, match="unknown preset 'fast'; choose from") as excinfo:
        normalize_preset("fast")
    assert "energy-saver" in str(excinfo.value)


@pytest.mark.parametrize("spelling", ENERGY_SPELLINGS)
def test_python_api_goal(spelling):
    kw = {"predictor": "kavier", "feasibility": "rules", "max_gpus": 8}
    saver = coastline.recommend([JOB], goal=spelling, **kw)
    energy = coastline.recommend([JOB], goal="energy", **kw)
    assert saver.equals(energy)


@pytest.mark.parametrize("spelling", ENERGY_SPELLINGS)
def test_facade_preset_and_goal(spelling):
    rec = coastline.Coastline("kavier", feasibility="rules")
    kw = {"total_gpus": [1, 8], "batch_sizes": [8], "max_gpus": 8, "top_k": 1}
    by_preset = rec.recommend(JOB, preset=spelling, **kw)[0]
    by_goal = rec.recommend(JOB, goal=spelling, **kw)[0]
    assert by_preset.metadata["preset"] == by_goal.metadata["preset"] == "energy"


@pytest.mark.parametrize("spelling", ENERGY_SPELLINGS)
def test_config_preset(spelling):
    config = {
        "strategy": {"name": "multi_objective", "preset": spelling},
        "predictors": _PREDICTORS,
        "grid": {"batch_sizes": [8], "total_gpus": [1, 8], "top_k": 1},
    }
    assert PolicyFactory.create_strategy(config=config).get_name() == "multi_objective_energy"


def _config_file(tmp_path, preset: str, **extra) -> str:
    config = {
        "strategy": {"name": "multi_objective", "preset": preset},
        "predictors": _PREDICTORS,
        "grid": {"batch_sizes": [8], "total_gpus": [1, 2, 4, 8]},
        **extra,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(path)


def test_recommend_csv_preset(tmp_path):
    jobs = tmp_path / "jobs.csv"
    pd.DataFrame([JOB]).to_csv(jobs, index=False)
    out = tmp_path / "out.csv"
    coastline.recommend_csv(_config_file(tmp_path, "energy_saver"), jobs, out, cluster_gpus=8)

    row = next(csv.DictReader(out.open(newline="", encoding="utf-8")))
    assert row["feasible"] == "True", row["error"]
    assert "picked for the lowest energy" in row["rationale"]


def test_recommend_job_preset(tmp_path, capsys):
    from coastline.cli import main

    main(["recommend-job", "--config", _config_file(tmp_path, "Energy Saver", workload=dict(JOB))])
    payload = json.loads(capsys.readouterr().out)
    assert payload["metadata"]["preset"] == "energy"


@pytest.mark.parametrize(
    "spelling,goal", [("min-gpu", "min_gpu"), ("energy_saver", "energy"), ("runtime", "performance")]
)
def test_recommend_trace_cli_goal(spelling, goal):
    from coastline.cli.recommend_trace import _build_parser

    args = _build_parser().parse_args(["--input", "in.csv", "--output", "out.csv", "--goal", spelling])
    assert args.goal == goal


def test_recommend_trace_cli_rejects_an_unknown_goal(capsys):
    from coastline.cli.recommend_trace import _build_parser

    with pytest.raises(SystemExit) as excinfo:
        _build_parser().parse_args(["--input", "in.csv", "--output", "out.csv", "--goal", "fast"])
    assert excinfo.value.code == 2
    assert "unknown goal 'fast'" in capsys.readouterr().err


@pytest.mark.parametrize("spelling", ["energy_saver", "Energy Saver"])
def test_explain_preset(spelling, capsys):
    from coastline.cli import main

    main(
        ["explain", "--model", JOB["llm_model"], "--method", "lora", "--gpu-model", GPU, "--tokens", "1024"]
        + ["--batch-size", "8", "--predictor", "kavier", "--feasibility", "rules", "--top-k", "1"]
        + ["--preset", spelling]
    )
    assert "preset=energy" in capsys.readouterr().out


def test_explain_rejects_an_unknown_preset(capsys):
    from coastline.cli.explain import _build_parser

    base = ["--model", "m", "--method", "lora", "--gpu-model", GPU, "--tokens", "1024", "--batch-size", "8"]
    with pytest.raises(SystemExit) as excinfo:
        _build_parser().parse_args([*base, "--preset", "fast"])
    assert excinfo.value.code == 2
    assert "unknown preset 'fast'" in capsys.readouterr().err


@pytest.mark.parametrize("spelling", ["energy_saver", "Energy Saver"])
def test_dashboard_preset_and_goal(spelling):
    from fastapi.testclient import TestClient

    from coastline.ui.app import app

    body = {**JOB, "prediction_model": "kavier", "strategy": "multi_objective", "preset": spelling, "total_gpus": 8}
    batch = {"workloads": [JOB], "goal": spelling, "feasibility": "rules", "max_gpus": 8}
    with TestClient(app) as client:
        single = client.post("/api/recommend", json=body)
        many = client.post("/api/recommend/batch", json=batch)

    assert single.status_code == 200, single.text
    assert single.json()["preset"] == "energy"
    assert many.status_code == 200, many.text
    assert many.json()["results"][0]["feasible"] is True
