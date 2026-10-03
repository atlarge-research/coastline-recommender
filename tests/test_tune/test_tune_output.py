"""`coastline utils tune` checks that it can save the model before it trains one.

An output folder it cannot write stops the run before training with a plain error, and a save
that still fails ends the same way: exit 1 from the CLI. No ML backend is imported; the fit is a
stand-in.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

from coastline.cli import main
from coastline.sdk.predictors.performance.data_driven import tune as tune_module
from coastline.sdk.predictors.performance.data_driven.tune import tune

_RUN = {
    "model_name": "mistral-7b-v0.1",
    "method": "lora",
    "gpu_model": "NVIDIA-A100-SXM4-80GB",
    "number_nodes": 1,
    "number_gpus": 8,
    "tokens_per_sample": 1024,
    "batch_size": 8,
    "dataset_tokens_per_second": 5000.0,
    "train_runtime": 600.0,
    "is_valid": 1.0,
}

_needs_an_unwritable_folder = pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs a folder the current user cannot write (POSIX, as a non-root user)",
)


@pytest.fixture
def runs_csv(tmp_path):
    path = tmp_path / "runs.csv"
    pd.DataFrame([{**_RUN, "batch_size": batch} for batch in (4, 8, 16)]).to_csv(path, index=False)
    return path


@pytest.fixture
def read_only_folder(tmp_path):
    folder = tmp_path / "read_only"
    folder.mkdir()
    folder.chmod(0o555)
    yield folder
    folder.chmod(0o755)


def _no_fit(*args, **kwargs):
    raise AssertionError("trained a model before checking the output folder")


def _quick_fit(*args, **kwargs):
    """What _fit_xgboost returns: (artifacts, metrics, fit_seconds, device)."""
    return {"model": None}, {}, 0.0, "cpu"


@_needs_an_unwritable_folder
def test_an_unwritable_output_folder_fails_before_training(runs_csv, read_only_folder, monkeypatch):
    monkeypatch.setattr(tune_module, "_fit_xgboost", _no_fit)

    with pytest.raises(OSError, match="cannot write the tuned model"):
        tune(str(runs_csv), model="xgboost", output=str(read_only_folder / "xgboost.pkl"))


@_needs_an_unwritable_folder
def test_the_cli_reports_an_unwritable_output_folder(runs_csv, read_only_folder, monkeypatch, capsys):
    monkeypatch.setattr(tune_module, "_fit_xgboost", _no_fit)
    output = read_only_folder / "xgboost.pkl"

    with pytest.raises(SystemExit) as excinfo:
        main(["utils", "tune", "--data", str(runs_csv), "--model", "xgboost", "--output", str(output)])

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert f"coastline utils tune: cannot write the tuned model to {output}" in err
    assert "Traceback" not in err


def test_a_failed_save_exits_1_with_a_plain_error(runs_csv, tmp_path, monkeypatch, capsys):
    # The folder is writable, but the output path is a folder, so the save itself fails.
    monkeypatch.setattr(tune_module, "_fit_xgboost", _quick_fit)
    output = tmp_path / "taken"
    output.mkdir()

    with pytest.raises(SystemExit) as excinfo:
        main(["utils", "tune", "--data", str(runs_csv), "--model", "xgboost", "--output", str(output)])

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert f"coastline utils tune: cannot write the tuned model to {output}" in err
    assert "Traceback" not in err


def test_a_missing_dataset_exits_1_with_a_plain_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tune_module, "_fit_xgboost", _no_fit)
    missing = tmp_path / "missing.csv"

    with pytest.raises(SystemExit) as excinfo:
        main(["utils", "tune", "--data", str(missing), "--model", "xgboost", "--output", str(tmp_path / "x.pkl")])

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert str(missing) in err
    assert "Traceback" not in err


def test_a_writable_folder_is_created_and_the_model_saved(runs_csv, tmp_path, monkeypatch):
    monkeypatch.setattr(tune_module, "_fit_xgboost", _quick_fit)
    output = tmp_path / "new" / "folder" / "xgboost.pkl"

    result = tune(str(runs_csv), model="xgboost", output=str(output))

    assert result["path"] == str(output)
    assert output.is_file()
    assert sorted(path.name for path in output.parent.iterdir()) == ["xgboost.pkl"]  # no probe file left
