"""Dataset validation for `coastline utils tune`: a bad dataset fails with a clear error.

The expected errors and warnings follow from the rows each test builds. No ML backend is
imported.
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

# A valid row: model and GPU in Kavier's library, positive targets.
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


def test_missing_columns_fail_with_the_schema():
    """Dropping two required columns names those two and prints the dataset format."""
    df = pd.DataFrame([_GOOD]).drop(columns=["gpu_model", "train_runtime"])
    with pytest.raises(DatasetFormatError) as err:
        validate_dataset(df)
    msg = str(err.value)
    assert "gpu_model" in msg and "train_runtime" in msg
    assert "model_name" not in msg.split("\n")[0]  # present columns are not listed as missing
    assert "A valid tuning dataset" in msg  # the format help is included


def test_all_rows_filtered_raises():
    """With is_valid=0 on every row there are no usable rows, and validation raises DatasetFormatError."""
    df = pd.DataFrame([{**_GOOD, "is_valid": 0.0}] * 3)
    with pytest.raises(DatasetFormatError, match="no usable rows"):
        validate_dataset(df)


def test_filters_and_dropped_row_warning():
    """Of 3 good, 1 invalid and 1 zero-throughput row, 3 are kept and a warning reports the 2 dropped."""
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
    """One row, one config, and a model and GPU unknown to Kavier: each problem gets a warning."""
    row = {**_GOOD, "model_name": "totally-made-up-llm-9b", "gpu_model": "FAKE-GPU-1"}
    clean, warnings = validate_dataset(pd.DataFrame([row]))
    assert len(clean) == 1
    joined = "\n".join(warnings)
    assert f"at least {MIN_ROWS} valid rows" in joined
    assert "at least 2 distinct configurations" in joined
    assert "totally-made-up-llm-9b" in joined  # unknown model is named
    assert "FAKE-GPU-1" in joined  # unknown GPU is named


def test_clean_large_dataset_yields_no_warnings():
    """MIN_ROWS rows of a known model across several configs give no warnings."""
    rows = [{**_GOOD, "batch_size": b, "number_gpus": g} for b in (4, 8, 16, 32) for g in (1, 2, 4, 8)][:MIN_ROWS]
    rows += [dict(_GOOD)] * (MIN_ROWS - len(rows))
    clean, warnings = validate_dataset(pd.DataFrame(rows))
    assert len(clean) == MIN_ROWS
    assert warnings == []


def test_tune_rejects_bad_train_percentage_and_unknown_model(tmp_path):
    """Bad arguments are rejected before any dataset or ML work."""
    with pytest.raises(ValueError, match="train-percentage"):
        tune("does-not-matter.csv", train_percentage=0.0)
    # tabpfn and xgboost are tunable; a portfolio model like catboost is not (use dev/trainer).
    with pytest.raises(ValueError, match="only"):
        tune("does-not-matter.csv", model="catboost")


def test_format_help_lists_every_required_column():
    text = dataset_format_help()
    for col in _GOOD:
        assert col in text
