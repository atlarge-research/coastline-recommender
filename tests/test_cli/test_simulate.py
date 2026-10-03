"""Tests for `coastline simulate`, which predicts one configuration.

Most tests pin the analytical Kavier predictor and `--feasibility rules`, so the command needs
neither trained ML models nor AutoConf. Assertions check identities, signs and presence, since
Kavier's numbers may change.
"""

from __future__ import annotations

import json

import pytest

from coastline.cli import main

_BASE = [
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


def _run_json(capsys, *extra: str) -> dict:
    main([*_BASE, "--json", *extra])
    return json.loads(capsys.readouterr().out)


def test_simulate_predicts_throughput_and_power_for_one_config(capsys) -> None:
    result = _run_json(capsys, "--gpus-per-node", "4")

    assert result["error"] is None
    assert result["feasible"] is True
    assert result["predicted_throughput"] > 0
    assert result["predicted_power_watts"] > 0
    # The config echoed back is the one asked for, and the GPU identity holds.
    assert result["gpus_per_node"] == 4
    assert result["number_of_nodes"] == 1
    assert result["total_gpus"] == result["gpus_per_node"] * result["number_of_nodes"]
    assert result["cluster_power_watts"] == pytest.approx(result["predicted_power_watts"] * result["total_gpus"])
    assert result["tokens_per_watt"] == pytest.approx(result["predicted_throughput"] / result["predicted_power_watts"])


def test_simulate_omits_runtime_and_energy_without_total_tokens(capsys) -> None:
    """Kavier reports the time per step, so runtime and energy need a dataset size."""
    result = _run_json(capsys, "--gpus-per-node", "2")

    assert result["predicted_runtime_seconds"] is None
    assert result["energy_kwh"] is None


def test_simulate_derives_runtime_and_energy_from_total_tokens(capsys) -> None:
    total_tokens = 100_000_000
    result = _run_json(capsys, "--gpus-per-node", "2", "--total-tokens", str(total_tokens))

    runtime = result["predicted_runtime_seconds"]
    # runtime * throughput == the dataset, by construction.
    assert runtime == pytest.approx(total_tokens / result["predicted_throughput"])
    # energy is the cluster draw over that runtime, in kWh.
    assert result["energy_kwh"] == pytest.approx(result["cluster_power_watts"] * runtime / 3_600_000.0)


def test_simulate_does_not_report_a_score(capsys) -> None:
    """Policy scores are min-max normalised across a grid; for one config they are meaningless."""
    result = _run_json(capsys, "--gpus-per-node", "1")

    assert "combined_score" not in result
    assert "power_score" not in result
    assert "throughput_score" not in result


def test_simulate_reports_an_unsupported_model_as_an_error_and_exits_nonzero(capsys) -> None:
    argv = [a if a != "mistral-7b-v0.1" else "totally-unknown-model-xyz" for a in _BASE]

    with pytest.raises(SystemExit) as excinfo:
        main(argv)

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "error" in out


def test_simulate_text_report_names_the_config_and_the_predictions(capsys) -> None:
    main([*_BASE, "--gpus-per-node", "4", "--total-tokens", "1000000"])

    out = capsys.readouterr().out
    assert "4x1 = 4 GPU(s)" in out
    assert "mistral-7b-v0.1" in out
    for label in ("feasible", "thr", "power", "tok/W", "runtime", "energy"):
        assert label in out


def test_simulate_reports_power_for_a_non_kavier_predictor(capsys) -> None:
    """Only Kavier returns power with the throughput; other predictors need the separate power
    predictor, as in the grid pipeline."""
    argv = [a if a != "kavier" else "intelligent" for a in _BASE]
    main([*argv, "--gpus-per-node", "4", "--json"])
    result = json.loads(capsys.readouterr().out)

    assert result["error"] is None
    assert result["predicted_power_watts"] > 0
    assert result["cluster_power_watts"] > 0
    assert result["tokens_per_watt"] > 0


def test_simulate_rejects_an_unknown_predictor_rather_than_silently_defaulting(capsys) -> None:
    """A misspelled predictor exits 2 instead of running the `intelligent` default under the typed
    name. The facade and batch API use the same validator."""
    argv = [a if a != "kavier" else "catbost" for a in _BASE]

    with pytest.raises(SystemExit) as excinfo:
        main(argv)

    assert excinfo.value.code == 2
    assert "unknown predictor" in capsys.readouterr().err


def test_simulate_rejects_a_negative_dataset_size(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([*_BASE, "--total-tokens", "-1"])

    assert excinfo.value.code == 2


def test_simulate_labels_a_historical_runtime_as_not_this_dataset(capsys) -> None:
    """Without --total-tokens a cache or ML predictor reports the wall-clock time of the run it
    matched, for a dataset the caller did not declare. The report labels it as such."""
    argv = [a if a != "kavier" else "intelligent" for a in _BASE]
    main([*argv, "--gpus-per-node", "4", "--json"])
    result = json.loads(capsys.readouterr().out)

    if result["predicted_runtime_seconds"] is not None:
        assert result["runtime_source"] == "predictor_history"
        main([*argv, "--gpus-per-node", "4"])
        assert "for the dataset of the matched historical run" in capsys.readouterr().out


def test_simulate_marks_a_declared_dataset_runtime_as_such(capsys) -> None:
    result = _run_json(capsys, "--gpus-per-node", "4", "--total-tokens", "1000000")

    assert result["runtime_source"] == "total_tokens"


# flag spellings
# The thesis listings use underscore flags and a total GPU count; both spellings run the same
# command. `--number_gpus` is the cluster total, and the GPUs per node follow from the node count.

_THESIS_LISTING = [
    "simulate",
    "--model_name",
    "mistral-7b-v0.1",
    "--method",
    "lora",
    "--gpu_model",
    "NVIDIA-A100-SXM4-80GB",
    "--tokens_per_sample",
    "2048",
    "--batch_size",
    "8",
    "--number_gpus",
    "4",
    "--number_nodes",
    "1",
]

_BASE_UNDERSCORE = [
    "simulate",
    "--model_name",
    "mistral-7b-v0.1",
    "--method",
    "lora",
    "--gpu_model",
    "NVIDIA-A100-SXM4-80GB",
    "--tokens_per_sample",
    "1024",
    "--batch_size",
    "16",
    "--predictor",
    "kavier",
    "--feasibility",
    "rules",
]


def test_simulate_runs_the_thesis_listing_as_printed(capsys) -> None:
    """The command printed in the thesis runs as printed.

    It pins no predictor or feasibility checker, so it runs the defaults. Without AutoConf the
    checker reports itself unavailable and the command exits 1; the flags still parse and the
    report is printed.
    """
    try:
        main(list(_THESIS_LISTING))
    except SystemExit as excinfo:  # AutoConf unavailable; an argument error would exit 2
        assert excinfo.code == 1

    out = capsys.readouterr().out
    assert "4x1 = 4 GPU(s), batch 8 per device" in out
    assert "mistral-7b-v0.1 / lora / NVIDIA-A100-SXM4-80GB / 2048 tok" in out


def test_simulate_underscore_and_hyphen_spellings_are_the_same_command(capsys) -> None:
    """The underscore flags are aliases: 4 total GPUs on the default single node is 4 per node."""
    main([*_BASE, "--gpus-per-node", "4", "--json"])
    hyphenated = json.loads(capsys.readouterr().out)

    main([*_BASE_UNDERSCORE, "--number_gpus", "4", "--json"])
    underscored = json.loads(capsys.readouterr().out)

    assert underscored == hyphenated


def test_simulate_divides_total_gpus_over_the_declared_nodes(capsys) -> None:
    main([*_BASE_UNDERSCORE, "--number_gpus", "8", "--number_nodes", "2", "--json"])
    result = json.loads(capsys.readouterr().out)

    assert result["gpus_per_node"] == 4
    assert result["number_of_nodes"] == 2
    assert result["total_gpus"] == 8


def test_simulate_rejects_a_total_gpu_count_that_does_not_divide_over_the_nodes(capsys) -> None:
    """A total that does not divide evenly over the nodes is rejected; guessing a layout would change the job."""
    with pytest.raises(SystemExit) as excinfo:
        main([*_BASE_UNDERSCORE, "--number_gpus", "8", "--number_nodes", "3"])

    assert excinfo.value.code == 2
    assert "divide" in capsys.readouterr().err


def test_simulate_rejects_a_total_gpu_count_too_large_for_one_node(capsys) -> None:
    """Without --nodes the total must fit a single node; more than that needs a declared layout."""
    with pytest.raises(SystemExit) as excinfo:
        main([*_BASE_UNDERSCORE, "--number_gpus", "16"])

    assert excinfo.value.code == 2
    assert "--nodes" in capsys.readouterr().err


def test_simulate_refuses_a_total_and_a_per_node_gpu_count_together(capsys) -> None:
    """--number_gpus is the total, --gpus-per-node the width: together they can contradict."""
    with pytest.raises(SystemExit) as excinfo:
        main([*_BASE_UNDERSCORE, "--number_gpus", "4", "--gpus-per-node", "2"])

    assert excinfo.value.code == 2
    assert "not allowed with" in capsys.readouterr().err
