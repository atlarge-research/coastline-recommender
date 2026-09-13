"""`coastline simulate` — the single-config predict verb.

Every test pins the analytical Kavier predictor and `--feasibility rules` so the command needs
neither trained ML artifacts nor AutoConf. Assertions are invariants (identities, signs,
presence), never a magic engine number: the numbers are Kavier's to change.
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
    """Kavier reports per-step time, not total runtime, so both need a dataset size."""
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
    """Regression: only Kavier returns power from the throughput call. Every other predictor
    needs the dedicated power predictor, the same fallback the grid pipeline uses. Pinning only
    Kavier here is what hid this."""
    argv = [a if a != "kavier" else "intelligent" for a in _BASE]
    main([*argv, "--gpus-per-node", "4", "--json"])
    result = json.loads(capsys.readouterr().out)

    assert result["error"] is None
    assert result["predicted_power_watts"] > 0
    assert result["cluster_power_watts"] > 0
    assert result["tokens_per_watt"] > 0


def test_simulate_rejects_an_unknown_predictor_rather_than_silently_defaulting(capsys) -> None:
    """A typo must not resolve to the `intelligent` default and report its numbers under the
    typed name. Same validator the facade and batch API use."""
    argv = [a if a != "kavier" else "catbost" for a in _BASE]

    with pytest.raises(ValueError, match="unknown predictor"):
        main(argv)


def test_simulate_rejects_a_negative_dataset_size(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([*_BASE, "--total-tokens", "-1"])

    assert excinfo.value.code == 2


def test_simulate_labels_a_historical_runtime_as_not_this_dataset(capsys) -> None:
    """Without --total-tokens a cache/ML predictor reports the wall clock of the run it matched,
    for a dataset the caller never declared. That must not read as this job's runtime."""
    argv = [a if a != "kavier" else "intelligent" for a in _BASE]
    main([*argv, "--gpus-per-node", "4", "--json"])
    result = json.loads(capsys.readouterr().out)

    if result["predicted_runtime_seconds"] is not None:
        assert result["runtime_source"] == "predictor_history"
        main([*argv, "--gpus-per-node", "4"])
        assert "NOT this dataset" in capsys.readouterr().out


def test_simulate_marks_a_declared_dataset_runtime_as_such(capsys) -> None:
    result = _run_json(capsys, "--gpus-per-node", "4", "--total-tokens", "1000000")

    assert result["runtime_source"] == "total_tokens"


# --- flag spellings -------------------------------------------------------------------------
# The thesis prints its listings with underscore flags and a TOTAL GPU count; both spellings
# must reach the same command. `--number_gpus` is NOT `--gpus-per-node`: it is the cluster
# total, and the per-node layout is derived from the node count.

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


def test_simulate_runs_the_thesis_listing_verbatim(capsys) -> None:
    """The command as printed in the thesis must run as printed — it used to exit 2.

    It pins no predictor and no feasibility checker, so it runs the defaults: where AutoConf is
    absent the checker reports itself unavailable and the command exits 1. Either way the flags
    parsed and the report was produced, which is what this verb was failing to do.
    """
    try:
        main(list(_THESIS_LISTING))
    except SystemExit as excinfo:  # AutoConf unavailable — never an argument error
        assert excinfo.code == 1

    out = capsys.readouterr().out
    assert "4x1 = 4 GPU(s), batch 8 per device" in out
    assert "mistral-7b-v0.1 / lora / NVIDIA-A100-SXM4-80GB / 2048 tok" in out


def test_simulate_underscore_and_hyphen_spellings_are_the_same_command(capsys) -> None:
    """Aliases, not a second code path: 4 total GPUs over the default single node is 4 per node."""
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
    """A layout with uneven nodes is not a layout; guessing one would silently change the job."""
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
