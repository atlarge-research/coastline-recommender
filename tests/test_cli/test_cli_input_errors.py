"""Bad input to any subcommand prints one `error:` line and no Python traceback.

A wrong flag value, a missing or malformed file, or an unknown GPU is a usage error (exit 2, as
argparse does for an unknown subcommand). A run that finds nothing to recommend exits 1, like
`explain` and `simulate` with no result. Every case runs in process, so an uncaught exception
fails the test.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import yaml

from coastline.cli import main

_TRACE = Path(__file__).resolve().parents[2] / "config" / "coastline_functionality" / "sample_trace.csv"

_WORKLOAD = {
    "llm_model": "mistral-7b-v0.1",
    "fine_tuning_method": "lora",
    "gpu_model": "NVIDIA-A100-SXM4-80GB",
    "tokens_per_sample": 1024,
    "batch_size": 16,
}

_SIMULATE = [
    "simulate",
    "--model",
    "mistral-7b-v0.1",
    "--method",
    "lora",
    "--gpu-model",
    "NVIDIA-A100-SXM4-80GB",
    "--tokens",
    "1024",
    "--batch-size",
    "16",
    "--predictor",
    "kavier",
    "--feasibility",
    "rules",
]

_EXPLAIN = ["explain", *_SIMULATE[1:]]


def _swap(argv: list[str], flag: str, value: str) -> list[str]:
    """``argv`` with the value after ``flag`` replaced."""
    out = list(argv)
    out[out.index(flag) + 1] = value
    return out


@pytest.fixture(autouse=True)
def _drop_stale_log_handlers():
    """An earlier test's setup_logging() can leave a root handler on a closed capture stream, which
    prints a logging traceback of its own; a real CLI process never has one."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(getattr(handler, "stream", None), "closed", False):
            root.removeHandler(handler)


def _fails(capsys, argv: list[str], code: int = 2) -> str:
    """Run the CLI, expect ``code``, and return stderr (checked for a clean `error:` line)."""
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    err = capsys.readouterr().err
    assert excinfo.value.code == code, err
    assert "error:" in err
    assert "Traceback" not in err
    return err


@pytest.fixture
def config(tmp_path) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "workload": _WORKLOAD,
                "strategy": {"name": "min_gpu"},
                "predictors": {"performance": "kavier", "energy": "kavier_power", "feasibility": "rules"},
                "grid": {"batch_sizes": [16], "total_gpus": [1, 2], "top_k": 1},
                "runtime": {"parallel_workers": 1},
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def workloads_csv(tmp_path) -> Path:
    path = tmp_path / "workloads.csv"
    path.write_text(",".join(_WORKLOAD) + "\n" + ",".join(str(v) for v in _WORKLOAD.values()) + "\n")
    return path


# recommend-job


def test_recommend_job_names_a_missing_config(capsys, tmp_path, workloads_csv) -> None:
    missing = tmp_path / "missing.yaml"
    argv = ["recommend-job", "--config", str(missing), "--input", str(workloads_csv), "--output", "o.csv"]
    err = _fails(capsys, argv)
    assert str(missing) in err


def test_recommend_job_names_a_missing_input(capsys, tmp_path, config) -> None:
    missing = tmp_path / "missing.csv"
    err = _fails(capsys, ["recommend-job", "--config", str(config), "--input", str(missing), "--output", "o.csv"])
    assert str(missing) in err


def test_recommend_job_explains_a_csv_given_to_single_mode(capsys, config, workloads_csv) -> None:
    err = _fails(capsys, ["recommend-job", "--config", str(config), "--input", str(workloads_csv)])
    assert '"workload"' in err and '"context"' in err  # the JSON shape single mode reads
    assert "--output" in err  # and how to run a CSV instead


def test_recommend_job_names_the_json_shape_for_a_flat_job(capsys, tmp_path, config) -> None:
    job = tmp_path / "job.json"
    job.write_text(json.dumps(_WORKLOAD))
    err = _fails(capsys, ["recommend-job", "--config", str(config), "--input", str(job)])
    assert '"workload"' in err and '"context"' in err


def test_recommend_job_names_an_unknown_predictor_in_the_config(capsys, tmp_path, config, workloads_csv) -> None:
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    raw["predictors"]["performance"] = "kavierr"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    output = tmp_path / "o.csv"
    err = _fails(
        capsys, ["recommend-job", "--config", str(config), "--input", str(workloads_csv), "--output", str(output)]
    )
    assert "unknown predictor 'kavierr'" in err
    assert not output.exists()


def test_recommend_job_names_a_missing_workload_column(capsys, tmp_path, config) -> None:
    workloads = tmp_path / "workloads.csv"
    workloads.write_text("llm_model,gpu_model\nmistral-7b-v0.1,NVIDIA-A100-SXM4-80GB\n")
    argv = ["recommend-job", "--config", str(config), "--input", str(workloads), "--output", str(tmp_path / "o.csv")]
    err = _fails(capsys, argv)
    assert "no column for fine_tuning_method" in err


def test_recommend_job_reports_an_input_without_a_header(capsys, tmp_path, config) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    argv = ["recommend-job", "--config", str(config), "--input", str(empty), "--output", str(tmp_path / "o.csv")]
    err = _fails(capsys, argv)
    assert "no header row" in err


@pytest.mark.parametrize("value", ["0", "-8"])
def test_recommend_job_rejects_a_cluster_without_gpus(capsys, config, value) -> None:
    err = _fails(capsys, ["recommend-job", "--config", str(config), "--cluster-gpus", value])
    assert "--cluster-gpus" in err


# recommend-trace


def test_recommend_trace_reports_an_empty_input(capsys, tmp_path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    err = _fails(capsys, ["recommend-trace", "--input", str(empty), "--output", str(tmp_path / "o.csv")])
    assert "empty" in err


def test_recommend_trace_names_a_missing_input(capsys, tmp_path) -> None:
    missing = tmp_path / "missing.csv"
    err = _fails(capsys, ["recommend-trace", "--input", str(missing), "--output", str(tmp_path / "o.csv")])
    assert str(missing) in err


@pytest.mark.parametrize(("flag", "value"), [("--cluster-gpus", "0"), ("--cluster-gpus", "-8"), ("--node-gpus", "0")])
def test_recommend_trace_rejects_a_non_positive_cluster(capsys, tmp_path, flag, value) -> None:
    argv = ["recommend-trace", "--input", str(_TRACE), "--output", str(tmp_path / "o.csv"), flag, value]
    err = _fails(capsys, argv)
    assert flag in err


# simulate


def test_simulate_reports_an_unknown_gpu(capsys) -> None:
    err = _fails(capsys, _swap(_SIMULATE, "--gpu-model", "NOT-A-GPU"))
    assert "NOT-A-GPU" in err


def test_simulate_rejects_an_unknown_feasibility_mode(capsys) -> None:
    err = _fails(capsys, _swap(_SIMULATE, "--feasibility", "bogus"))
    assert "--feasibility" in err


def test_simulate_rejects_an_unknown_predictor(capsys) -> None:
    err = _fails(capsys, _swap(_SIMULATE, "--predictor", "bogus"))
    assert "unknown predictor" in err


@pytest.mark.parametrize("flag", ["--tokens", "--batch-size"])
def test_simulate_rejects_a_zero_workload_size(capsys, flag) -> None:
    err = _fails(capsys, _swap(_SIMULATE, flag, "0"))
    assert flag in err


# explain


def test_explain_reports_an_unknown_gpu(capsys) -> None:
    err = _fails(capsys, _swap(_EXPLAIN, "--gpu-model", "NOT-A-GPU"))
    assert "NOT-A-GPU" in err


def test_explain_rejects_zero_max_gpus(capsys) -> None:
    err = _fails(capsys, [*_EXPLAIN, "--max-gpus", "0"])
    assert "--max-gpus" in err


def test_explain_rejects_an_unknown_feasibility_mode(capsys) -> None:
    err = _fails(capsys, _swap(_EXPLAIN, "--feasibility", "bogus"))
    assert "--feasibility" in err


def test_explain_reports_a_model_no_predictor_knows_as_a_failed_run(capsys) -> None:
    """The input parses; the run finds nothing to explain. Exit 1, with the predictor's reason."""
    err = _fails(capsys, _swap(_EXPLAIN, "--model", "not-a-model"), code=1)
    assert "not-a-model" in err


# utils


def test_trace_to_runs_reports_an_unrecognised_schema(capsys, tmp_path) -> None:
    junk = tmp_path / "junk.csv"
    junk.write_text("foo,bar\n1,2\n")
    err = _fails(capsys, ["utils", "trace-to-runs", "--input", str(junk), "--output", str(tmp_path / "o.csv")])
    assert "missing trace columns" in err


def test_trace_to_runs_names_a_missing_input(capsys, tmp_path) -> None:
    missing = tmp_path / "missing.csv"
    err = _fails(capsys, ["utils", "trace-to-runs", "--input", str(missing), "--output", str(tmp_path / "o.csv")])
    assert str(missing) in err


def test_plot_trace_reports_an_unsupported_figure_format(capsys, tmp_path) -> None:
    trace = tmp_path / "trace.csv"
    trace.write_text(
        "resources.num_gpus_per_node,resources.num_nodes,metadata.output.extrapolated_duration\n8,1,3600\n"
    )
    argv = [
        "utils",
        "plot-trace",
        "--input",
        str(trace),
        "--output",
        str(tmp_path / "plot.xyz"),
        "--duration-col",
        "metadata.output.extrapolated_duration",
        "--cluster-gpus",
        "8",
        "--node-gpus",
        "8",
    ]
    err = _fails(capsys, argv)
    assert "xyz" in err


def test_plot_trace_names_a_missing_input(capsys, tmp_path) -> None:
    missing = tmp_path / "missing.csv"
    err = _fails(capsys, ["utils", "plot-trace", "--input", str(missing), "--output", str(tmp_path / "p.pdf")])
    assert str(missing) in err


@pytest.mark.parametrize("column", ["no_such_column", "submitted_at"], ids=["absent", "neither numbers nor times"])
def test_plot_trace_reports_an_unusable_submit_column(capsys, tmp_path, column) -> None:
    trace = tmp_path / "trace.csv"
    trace.write_text(
        "resources.num_gpus_per_node,resources.num_nodes,metadata.output.extrapolated_duration,submitted_at\n"
        "8,1,3600,not a time\n"
    )
    argv = [
        "utils",
        "plot-trace",
        "--input",
        str(trace),
        "--output",
        str(tmp_path / "plot.pdf"),
        "--duration-col",
        "metadata.output.extrapolated_duration",
        "--submit-col",
        column,
        "--cluster-gpus",
        "8",
        "--node-gpus",
        "8",
    ]
    err = _fails(capsys, argv)
    assert column in err
