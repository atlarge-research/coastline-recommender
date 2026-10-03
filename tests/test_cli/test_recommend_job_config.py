"""`coastline recommend-job --config` (single declared-job mode) runs what the YAML declares.

Covered: the declared workload's GPU, the policy's `strategy.max_slowdown` guard, and a stdout that
holds only the recommendation JSON. Real runs pin Kavier and the `rules` checker, so no AutoConf
model or ML pickle is needed. Assertions compare the single and batch paths or read the request
the engine received, and depend on no Kavier number.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import pytest
import yaml

import coastline
from coastline.cli import main
from coastline.cli.run import _workload_and_context
from coastline.sdk.models.recommendation import Recommendation
from coastline.sdk.recommend import engine

_WORKLOAD = {
    "llm_model": "mistral-7b-v0.1",
    "fine_tuning_method": "lora",
    "gpu_model": "L40S",
    "tokens_per_sample": 1024,
    "batch_size": 8,
}


def _config(**overrides) -> dict:
    config = {
        "workload": dict(_WORKLOAD),
        "strategy": {"name": "multi_objective", "preset": "balanced"},
        "predictors": {"performance": "kavier", "energy": "kavier_power", "feasibility": "rules"},
        "grid": {"batch_sizes": [4, 8, 16], "total_gpus": [1, 2, 4, 8], "top_k": 3},
        "runtime": {"parallel_workers": 1},
    }
    for section, value in overrides.items():
        config[section] = value
    return config


def _write(tmp_path: Path, config: dict, name: str = "experiment.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _first_row(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle))


@pytest.fixture
def logging_on():
    """Undo a `logging.disable` left behind by an earlier test, so log records are emitted."""
    previous = logging.root.manager.disable
    logging.disable(logging.NOTSET)
    yield
    logging.disable(previous)


@pytest.fixture
def captured_request(monkeypatch):
    """Stub the engine call and keep the request it receives."""
    captured: dict = {}

    def fake_run_request(request):
        captured["request"] = request
        rec = Recommendation(
            gpus_per_node=2,
            number_of_nodes=1,
            total_gpus=2,
            strategy="multi_objective_balanced",
            predicted_throughput=100.0,
            metadata={"predicted_power_watts": 200.0, "tokens_per_watt": 0.5},
        )
        return [rec], {}

    monkeypatch.setattr(engine, "run_request", fake_run_request)
    return captured


# the declared workload's GPU


def test_the_declared_workload_gpu_is_the_gpu_recommended_for(tmp_path) -> None:
    raw = _config()
    workload, context = _workload_and_context(_write(tmp_path, raw), raw)

    assert workload.gpu_model == "L40S"
    assert "L40S" in context.available_gpu_models


def test_the_declared_workload_gpu_wins_over_the_grid_gpu_list(tmp_path) -> None:
    raw = _config(grid={"gpu_models": ["NVIDIA-A100-SXM4-80GB"], "batch_sizes": [8], "total_gpus": [1]})
    workload, context = _workload_and_context(_write(tmp_path, raw), raw)

    assert workload.gpu_model == "L40S"
    # The context lists the grid's GPUs and the one the job runs on.
    assert set(context.available_gpu_models) == {"NVIDIA-A100-SXM4-80GB", "L40S"}


def test_without_a_declared_gpu_the_grid_and_system_defaults_still_apply(tmp_path) -> None:
    workload_cfg = {k: v for k, v in _WORKLOAD.items() if k != "gpu_model"}
    from_grid = _config(workload=workload_cfg, grid={"gpu_models": ["L40S"]})
    from_system = _config(workload=workload_cfg, system={"default_gpu": "NVIDIA-H100-PCIe"})
    no_gpu_anywhere = _config(workload=workload_cfg)

    assert _workload_and_context(_write(tmp_path, from_grid), from_grid)[0].gpu_model == "L40S"
    assert _workload_and_context(_write(tmp_path, from_system), from_system)[0].gpu_model == "NVIDIA-H100-PCIe"
    assert _workload_and_context(_write(tmp_path, no_gpu_anywhere), no_gpu_anywhere)[0].gpu_model == (
        "NVIDIA-A100-SXM4-80GB"
    )


def test_single_mode_matches_the_batch_csv_path_for_the_declared_gpu(tmp_path, capsys, logging_on) -> None:
    """The batch CSV path reads the GPU from the row and single mode from the YAML; both give the
    same recommendation."""
    config_path = _write(tmp_path, _config())
    rows = tmp_path / "one.csv"
    rows.write_text(",".join(_WORKLOAD) + "\n" + ",".join(str(v) for v in _WORKLOAD.values()) + "\n")
    out_csv = tmp_path / "recs.csv"

    coastline.recommend_csv(str(config_path), str(rows), str(out_csv))
    batch = _first_row(out_csv)
    capsys.readouterr()

    main(["recommend-job", "--config", str(config_path)])
    single = json.loads(capsys.readouterr().out)

    assert single["configuration"]["total_gpus"] == int(batch["recommended_total_gpus"])
    assert single["performance"]["throughput_tokens_per_sec"] == pytest.approx(
        float(batch["predicted_throughput"]), rel=1e-3
    )


# stdout holds only the JSON


def test_single_mode_stdout_is_only_the_recommendation_json(tmp_path, capsys, captured_request, logging_on) -> None:
    main(["recommend-job", "--config", str(_write(tmp_path, _config()))])
    captured = capsys.readouterr()

    payload = json.loads(captured.out)  # fails if a log line precedes the JSON
    assert payload["configuration"]["total_gpus"] == 2
    # The run still logs; the records go to stderr.
    assert "Recommender starting" in captured.err


# strategy.max_slowdown


def test_max_slowdown_in_the_yaml_reaches_the_engine_as_the_runtime_guard(tmp_path, captured_request) -> None:
    config = _config(strategy={"name": "min_gpu", "max_slowdown": 2.0})
    main(["recommend-job", "--config", str(_write(tmp_path, config))])

    assert captured_request["request"].config["strategy"]["runtime_guard_k"] == 2.0


def test_max_slowdown_gives_the_same_pick_as_the_batch_csv_path(tmp_path, capsys, logging_on) -> None:
    """min_gpu alone picks the fewest GPUs; max_slowdown 1.0 allows only the fastest config.
    The batch CSV path applies the same guard, so both paths pick the same config."""
    config = _config(strategy={"name": "min_gpu", "max_slowdown": 1.0})
    config_path = _write(tmp_path, config)
    rows = tmp_path / "one.csv"
    rows.write_text(",".join(_WORKLOAD) + "\n" + ",".join(str(v) for v in _WORKLOAD.values()) + "\n")
    out_csv = tmp_path / "recs.csv"

    coastline.recommend_csv(str(config_path), str(rows), str(out_csv))
    batch = _first_row(out_csv)
    capsys.readouterr()

    main(["recommend-job", "--config", str(config_path)])
    guarded = json.loads(capsys.readouterr().out)

    del config["strategy"]["max_slowdown"]
    main(["recommend-job", "--config", str(_write(tmp_path, config, "unguarded.yaml"))])
    unguarded = json.loads(capsys.readouterr().out)

    assert guarded["configuration"]["total_gpus"] == int(batch["recommended_total_gpus"])
    # The guard changed the pick: without it min_gpu keeps the smallest layout.
    assert guarded["configuration"]["total_gpus"] > unguarded["configuration"]["total_gpus"]
