"""Input handling of the batch CSV recommender ``recommend_csv`` (Kavier path, rules feasibility).

Covers the predictor name in the config, empty config sections, the header check, rows with
extra cells, the per-row error column, an input with no data rows, and the example in the
shipped config/batch_config.yaml.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest
import yaml

from coastline.sdk.policies import PolicyFactory
from coastline.sdk.recommend import engine
from coastline.sdk.recommend.batch_csv import recommend_csv

_REPO_ROOT = Path(__file__).resolve().parents[2]

_HEADER = "llm_model,fine_tuning_method,gpu_model,tokens_per_sample,batch_size"
_ROW = "mistral-7b-v0.1,lora,NVIDIA-A100-SXM4-80GB,1024,16"


def _config(**predictors) -> dict:
    return {
        "strategy": {"name": "min_gpu"},
        "predictors": {"performance": "kavier", "energy": "kavier_power", "feasibility": "rules", **predictors},
        "grid": {"batch_sizes": [8, 16], "total_gpus": [1, 2, 4, 8]},
    }


def _run(tmp_path, csv_text: str, config=None) -> Path:
    """Write the config (a dict, or YAML text as written) and the input, run, return the output path."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(config if isinstance(config, str) else yaml.safe_dump(config or _config()))
    inp = tmp_path / "in.csv"
    inp.write_text(csv_text)
    out = tmp_path / "out.csv"
    recommend_csv(cfg, inp, out)
    return out


def _rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# predictors.performance
def test_an_unknown_predictor_in_the_config_fails_before_any_row(tmp_path):
    # A misspelt predictor stops the run before any output is written.
    with pytest.raises(ValueError, match="unknown predictor 'kavir'"):
        _run(tmp_path, f"{_HEADER}\n{_ROW}\n", _config(performance="kavir"))
    assert not (tmp_path / "out.csv").exists()


def test_the_predictor_name_in_the_config_ignores_letter_case(tmp_path, monkeypatch):
    # 'Kavier' reaches PolicyFactory as 'kavier' and builds KavierPredictor.
    built = []
    original = PolicyFactory.throughput_predictor

    def spy(predictor_config):
        predictor = original(predictor_config)
        built.append((predictor_config.get("performance"), type(predictor).__name__))
        return predictor

    monkeypatch.setattr(PolicyFactory, "throughput_predictor", staticmethod(spy))
    rows = _rows(_run(tmp_path, f"{_HEADER}\n{_ROW}\n", _config(performance="Kavier")))
    assert set(built) == {("kavier", "KavierPredictor")}
    assert rows[0]["feasible"] == "True"


# Empty config sections
def test_empty_config_sections_get_the_defaults(tmp_path):
    # A section whose keys are all commented out loads as None.
    config = (
        "strategy:\n"
        "  # name: min_gpu\n"
        "predictors:\n"
        "  performance: kavier\n"
        "  feasibility: rules\n"
        "grid:\n"
        "  # total_gpus: [1, 2]\n"
        "input:\n"
        "  columns:\n"
        "    # my_model_column: llm_model\n"
    )
    rows = _rows(_run(tmp_path, f"{_HEADER}\n{_ROW}\n", config))
    assert rows[0]["feasible"] == "True"
    assert int(rows[0]["recommended_total_gpus"]) >= 1


def test_blank_strategy_values_mean_unset(tmp_path):
    # 'max_slowdown:' and 'name:' with no value load as None and mean unset.
    with_blank = _config()
    with_blank["strategy"] = {"name": None, "max_slowdown": None}
    without = _config()
    without["strategy"] = {}
    blank = _rows(_run(tmp_path, f"{_HEADER}\n{_ROW}\n", with_blank))
    unset = _rows(_run(tmp_path, f"{_HEADER}\n{_ROW}\n", without))
    assert blank == unset
    assert blank[0]["feasible"] == "True"


# Header check
def test_a_header_without_the_workload_columns_is_rejected(tmp_path):
    # Without this check every row would come back blank and feasible=False, like an
    # infeasible workload.
    with pytest.raises(ValueError) as excinfo:
        _run(tmp_path, "model_name,method,gpu,seq_len,batch\nmistral-7b-v0.1,lora,NVIDIA-A100-SXM4-80GB,1024,16\n")
    message = str(excinfo.value)
    assert "no column for llm_model, fine_tuning_method, gpu_model, tokens_per_sample, batch_size" in message
    assert "input.columns" in message
    assert not (tmp_path / "out.csv").exists()


def test_the_missing_columns_are_named(tmp_path):
    with pytest.raises(ValueError, match=r"no column for gpu_model, batch_size \(header: "):
        _run(tmp_path, "llm_model,fine_tuning_method,tokens_per_sample\nmistral-7b-v0.1,lora,1024\n")


def test_input_columns_still_maps_other_headers(tmp_path):
    config = _config()
    config["input"] = {"columns": {"model_name": "llm_model", "seq_len": "tokens_per_sample"}}
    rows = _rows(
        _run(
            tmp_path,
            "model_name,fine_tuning_method,gpu_model,seq_len,batch_size\n"
            "mistral-7b-v0.1,lora,NVIDIA-A100-SXM4-80GB,1024,16\n",
            config,
        )
    )
    assert rows[0]["feasible"] == "True"
    assert rows[0]["error"] == ""


# Rows with more cells than the header
def test_a_trailing_empty_cell_is_ignored(tmp_path):
    # csv keeps a cell past the header under the key None, which the CSV writer rejects.
    rows = _rows(_run(tmp_path, f"{_HEADER}\n{_ROW}\n{_ROW},\n"))
    assert [r["feasible"] for r in rows] == ["True", "True"]
    assert rows[1]["recommended_total_gpus"] == rows[0]["recommended_total_gpus"]
    assert rows[1]["error"] == ""


def test_a_row_with_extra_data_fails_alone(tmp_path):
    # An unquoted '1,024' shifts the row to tokens_per_sample=1, batch_size=24 with '16' left
    # over; running it would recommend for a workload nobody asked for.
    good, shifted = _rows(_run(tmp_path, f"{_HEADER}\n{_ROW}\nmistral-7b-v0.1,lora,NVIDIA-A100-SXM4-80GB,1,024,16\n"))
    assert good["feasible"] == "True"
    assert shifted["feasible"] == "False"
    assert shifted["error"] == "row has more cells than the header: '16'"
    assert shifted["recommended_total_gpus"] == ""


def test_a_failure_while_writing_leaves_the_previous_output_whole(tmp_path, monkeypatch):
    out = _run(tmp_path, f"{_HEADER}\n{_ROW}\n")
    before = out.read_text()

    calls = []

    def rationale_failing_on_the_second_row(recs, meta):
        calls.append(meta)
        if len(calls) == 2:
            raise RuntimeError("rationale failed")
        return "rationale."

    monkeypatch.setattr(engine, "recommendation_rationale", rationale_failing_on_the_second_row)
    with pytest.raises(RuntimeError, match="rationale failed"):
        _run(tmp_path, f"{_HEADER}\n{_ROW}\n{_ROW}\n")
    assert out.read_text() == before


# The error column
def test_each_failed_row_says_why(tmp_path):
    good, gpu, tokens, blank, model = _rows(
        _run(
            tmp_path,
            f"{_HEADER}\n"
            f"{_ROW}\n"
            "mistral-7b-v0.1,lora,NOT-A-REAL-GPU,1024,16\n"
            "mistral-7b-v0.1,lora,NVIDIA-A100-SXM4-80GB,abc,16\n"
            "mistral-7b-v0.1,lora,,1024,16\n"
            "not-a-model,lora,NVIDIA-A100-SXM4-80GB,1024,16\n",
        )
    )
    assert (good["feasible"], good["error"]) == ("True", "")
    for row in (gpu, tokens, blank, model):
        assert row["feasible"] == "False"
        assert row["recommended_total_gpus"] == ""
        assert row["error"]
        # One line, so the output keeps one row per line.
        assert "\n" not in row["error"]
    assert "NOT-A-REAL-GPU" in gpu["error"]
    assert tokens["error"].startswith("tokens_per_sample: Input should be a valid integer")
    assert blank["error"] == "missing required field: gpu_model"


def test_the_error_column_comes_after_the_existing_columns(tmp_path):
    header = _run(tmp_path, f"{_HEADER}\n{_ROW}\n").read_text().splitlines()[0]
    assert header == (
        f"{_HEADER},recommended_total_gpus,recommended_gpus_per_node,recommended_number_of_nodes,"
        "recommended_batch_size,predicted_throughput,predicted_runtime_seconds,predicted_power_watts,"
        "tokens_per_watt,feasible,rationale,error"
    )


# No data rows
def test_an_input_with_only_a_header_replaces_the_previous_output(tmp_path):
    # A header-only input still rewrites the output, so no old recommendations remain.
    out = _run(tmp_path, f"{_HEADER}\n{_ROW}\n")
    assert len(_rows(out)) == 1
    _run(tmp_path, f"{_HEADER}\n")
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith(f"{_HEADER},recommended_total_gpus,")


def test_an_empty_input_file_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="no header row"):
        _run(tmp_path, "")
    assert not (tmp_path / "out.csv").exists()


# The shipped example
def test_the_example_command_in_batch_config_runs(tmp_path):
    # Run the command given in the header comment of config/batch_config.yaml.
    text = (_REPO_ROOT / "config" / "batch_config.yaml").read_text(encoding="utf-8")
    config = re.search(r"--config (\S+)", text).group(1)
    inp = re.search(r"--input (\S+)", text).group(1)
    out = tmp_path / "recommendations.csv"
    recommend_csv(_REPO_ROOT / config, _REPO_ROOT / inp, out)
    rows = _rows(out)
    assert rows
    assert all((r["feasible"], r["error"]) == ("True", "") for r in rows)
