"""Dataset validation for `coastline tune` — the loud-failure contract.

Oracles are hand-built DataFrames: every expected error/warning is derived from
the row counts and column values constructed in the test, never from the
validator's own output. No ML backend is imported (validation is light).
"""

import pandas as pd
import pytest

from coastline.sdk.predictors.performance.data_driven.tune import (
    MIN_ROWS,
    DatasetFormatError,
    dataset_format_help,
    tune,
    validate_dataset,
)

# A structurally-valid row: known model+GPU (in Kavier's library), positive targets.
_GOOD = {
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


def test_missing_columns_fail_loudly_with_the_schema():
    """Dropping two required columns must name exactly those two and print the contract."""
    df = pd.DataFrame([_GOOD]).drop(columns=["gpu_model", "train_runtime"])
    with pytest.raises(DatasetFormatError) as err:
        validate_dataset(df)
    msg = str(err.value)
    assert "gpu_model" in msg and "train_runtime" in msg
    assert "model_name" not in msg.split("\n")[0]  # present columns are not listed as missing
    assert "A valid tuning dataset" in msg  # the full contract rides along


def test_all_rows_filtered_fails_loudly():
    """is_valid=0 on every row -> no usable rows -> DatasetFormatError, not a silent empty fit."""
    df = pd.DataFrame([{**_GOOD, "is_valid": 0.0}] * 3)
    with pytest.raises(DatasetFormatError, match="no usable rows"):
        validate_dataset(df)


def test_filters_and_dropped_row_warning():
    """3 good + 1 invalid + 1 zero-throughput -> 3 kept, and the drop is called out."""
    rows = [
        {**_GOOD, "batch_size": 4},
        {**_GOOD, "batch_size": 8},
        {**_GOOD, "batch_size": 16, "number_gpus": 4},
        {**_GOOD, "is_valid": 0.0},
        {**_GOOD, "dataset_tokens_per_second": 0.0},
    ]
    clean, warnings = validate_dataset(pd.DataFrame(rows))
    assert len(clean) == 3
    assert any("dropped" in w and "2 row(s)" in w for w in warnings)


def test_quality_warnings_name_the_violated_properties():
    """1 row, 1 config, model+GPU unknown to Kavier -> each property is spelled out."""
    row = {**_GOOD, "model_name": "totally-made-up-llm-9b", "gpu_model": "FAKE-GPU-1"}
    clean, warnings = validate_dataset(pd.DataFrame([row]))
    assert len(clean) == 1
    joined = "\n".join(warnings)
    assert f"at least {MIN_ROWS} valid rows" in joined
    assert "at least 2 distinct configurations" in joined
    assert "totally-made-up-llm-9b" in joined  # unknown model is named
    assert "FAKE-GPU-1" in joined  # unknown GPU is named


def test_clean_large_dataset_yields_no_warnings():
    """MIN_ROWS known-model rows across several configs -> zero quality warnings."""
    rows = [{**_GOOD, "batch_size": b, "number_gpus": g} for b in (4, 8, 16, 32) for g in (1, 2, 4, 8)][:MIN_ROWS]
    rows += [dict(_GOOD)] * (MIN_ROWS - len(rows))
    clean, warnings = validate_dataset(pd.DataFrame(rows))
    assert len(clean) == MIN_ROWS
    assert warnings == []


def test_tune_rejects_bad_train_percentage_and_unknown_model(tmp_path):
    """Argument validation fires before any dataset/ML work."""
    with pytest.raises(ValueError, match="train-percentage"):
        tune("does-not-matter.csv", train_percentage=0.0)
    # tabpfn and xgboost are tunable; a portfolio model like catboost is not (use dev/trainer).
    with pytest.raises(ValueError, match="only"):
        tune("does-not-matter.csv", model="catboost")


def test_format_help_lists_every_required_column():
    text = dataset_format_help()
    for col in _GOOD:
        assert col in text
