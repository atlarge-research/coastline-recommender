"""Tests for the available-options loader (coastline.sdk.io.options_loader).

The loader reads ``<trace-archive>/profiling-dataset/curated_trace.csv``, whose method column is
named ``method``. With a wrong path or column name it falls back to a hardcoded list, and the
web UI then offers models the backend cannot predict. Expected values are sort orders and
distinct-value sets worked out from the inputs, plus models found only in the curated catalog or
only in the fallback, so a silent fallback is detected.
"""

import pandas as pd
import pytest

from coastline.sdk.io.options_loader import (
    DEFAULT_OPTIONS_PATH,
    get_fallback_options,
    load_available_options,
)


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _row(**overrides):
    base = {
        "model_name": "m1",
        "method": "full",
        "gpu_model": "G1",
        "tokens_per_sample": 512,
        "batch_size": 4,
    }
    base.update(overrides)
    return base


def test_reads_method_column_not_fine_tuning_method(tmp_path):
    # The curated CSV names the column 'method'. Reading 'fine_tuning_method' raises KeyError and
    # returns the fallback, whose methods are not the ones in this file. The two come back sorted.
    csv = tmp_path / "options.csv"
    _write_csv(
        csv,
        [
            _row(method="z-method", tokens_per_sample=512),
            _row(method="a-method", tokens_per_sample=1024),
        ],
    )
    load_available_options.cache_clear()
    opts = load_available_options(csv)
    assert opts["methods"] == ["a-method", "z-method"]


def test_string_options_returned_in_ascending_sort_order(tmp_path):
    # The rows are out of order; the loader sorts them: 'granite...' < 'mistral...' and
    # 'L40S' < 'NVIDIA...'.
    csv = tmp_path / "options.csv"
    _write_csv(
        csv,
        [
            _row(model_name="mistral-7b-v0.1", gpu_model="NVIDIA-A100-SXM4-80GB"),
            _row(model_name="granite-3.3-8b", gpu_model="L40S"),
        ],
    )
    load_available_options.cache_clear()
    opts = load_available_options(csv)
    assert opts["models"] == ["granite-3.3-8b", "mistral-7b-v0.1"]
    assert opts["gpus"] == ["L40S", "NVIDIA-A100-SXM4-80GB"]


def test_numeric_fields_drop_nan_cast_to_python_int_and_sort(tmp_path):
    # The NaN in the third row makes pandas read the numeric columns as float64. The loader drops
    # the NaN, casts to Python int and sorts; uncast numpy floats would show as "512.0" in the UI.
    csv = tmp_path / "options.csv"
    _write_csv(
        csv,
        [
            _row(tokens_per_sample=2048, batch_size=16),
            _row(tokens_per_sample=512, batch_size=4),
            _row(tokens_per_sample=float("nan"), batch_size=float("nan")),
        ],
    )
    load_available_options.cache_clear()
    opts = load_available_options(csv)
    assert opts["tokens_per_sample"] == [512, 2048]
    assert opts["batch_sizes"] == [4, 16]
    assert all(type(t) is int for t in opts["tokens_per_sample"])
    assert all(type(b) is int for b in opts["batch_sizes"])


def test_duplicate_rows_collapse_to_distinct_values(tmp_path):
    # Four rows with two distinct methods give two methods, so the dropdown lists 'lora' once.
    csv = tmp_path / "options.csv"
    _write_csv(
        csv,
        [
            _row(method="lora"),
            _row(method="lora"),
            _row(method="full"),
            _row(method="full"),
        ],
    )
    load_available_options.cache_clear()
    opts = load_available_options(csv)
    assert opts["methods"] == ["full", "lora"]


def test_nan_option_value_is_excluded(tmp_path):
    # One row has no gpu_model; it is dropped, so the dropdown gets no NaN or empty entry.
    csv = tmp_path / "options.csv"
    _write_csv(
        csv,
        [
            _row(gpu_model="L40S"),
            _row(gpu_model=float("nan")),
        ],
    )
    load_available_options.cache_clear()
    opts = load_available_options(csv)
    assert opts["gpus"] == ["L40S"]


def test_missing_file_falls_back_to_hardcoded_options(tmp_path):
    # A missing file does not raise; it returns the hardcoded fallback.
    # 'mixtral-8x7b-instruct-v0.1' is in the fallback and not in the curated catalog.
    load_available_options.cache_clear()
    opts = load_available_options(tmp_path / "does_not_exist.csv")
    assert opts == get_fallback_options()
    assert "mixtral-8x7b-instruct-v0.1" in opts["models"]


def test_missing_required_column_falls_back_instead_of_crashing(tmp_path):
    # Without a 'method' column the loader catches the KeyError and returns the fallback.
    csv = tmp_path / "broken.csv"
    _write_csv(csv, [{"model_name": "m1", "gpu_model": "G1"}])  # no 'method' etc.
    load_available_options.cache_clear()
    opts = load_available_options(csv)
    assert opts == get_fallback_options()
    assert "gptq-lora" in opts["methods"]


def test_data_dir_env_overrides_lookup_location(tmp_path, monkeypatch):
    # DATA_DIR moves the default lookup to <DATA_DIR>/profiling-dataset/curated_trace.csv.
    # 'sentinel-model' is in neither the curated catalog nor the fallback, so it can come only
    # from the file planted there.
    dataset = tmp_path / "profiling-dataset"
    dataset.mkdir()
    _write_csv(dataset / "curated_trace.csv", [_row(model_name="sentinel-model")])
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    load_available_options.cache_clear()
    opts = load_available_options()
    assert opts["models"] == ["sentinel-model"]


def test_default_path_points_to_curated_trace_under_profiling_dataset():
    # The default is <trace-archive>/profiling-dataset/curated_trace.csv; 'curated' is part of the
    # file name.
    assert "trace-archive" in DEFAULT_OPTIONS_PATH.parts
    assert "profiling-dataset" in DEFAULT_OPTIONS_PATH.parts
    assert DEFAULT_OPTIONS_PATH.name == "curated_trace.csv"


@pytest.mark.skipif(
    not DEFAULT_OPTIONS_PATH.exists(),
    reason="curated_trace.csv not present in this environment",
)
def test_real_curated_file_loads_and_is_not_the_fallback():
    # 'llama3.2-3b' is in the curated catalog alone and 'mixtral-8x7b-instruct-v0.1' in the
    # fallback alone, so a real load has the first and not the second.
    load_available_options.cache_clear()
    opts = load_available_options()
    assert "llama3.2-3b" in opts["models"]
    assert "mixtral-8x7b-instruct-v0.1" not in opts["models"]
