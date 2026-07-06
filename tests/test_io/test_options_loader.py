"""Tests for the centralized available-options loader.

These guard the two bugs that made the web UI offer models the backend
cannot predict: a wrong default path (pointed at a non-existent
``coastline/common/data`` dir) and reading a ``fine_tuning_method`` column that
the curated CSV does not have (the column is named ``method``).

Every assertion is checked against an oracle independent of the loader:
either a hand-computed sort order, a set/dedup invariant, or the
distinguishing markers between the real curated catalog and the hardcoded
fallback (so a silent fallback is actually detectable).
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
    # Regression: the curated CSV names the column 'method'. Reading a
    # 'fine_tuning_method' column (the old bug) would KeyError -> fallback,
    # so the returned methods would NOT be the ones in this file.
    # Oracle: two distinct methods 'z-method','a-method' sort to a<z.
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
    # Oracle: input is deliberately out of order; expected output is the
    # hand-computed lexicographic sort. 'granite...' < 'mistral...' ('g'<'m');
    # 'L40S' < 'NVIDIA...' ('L'<'N'). A no-op (unsorted) impl would return
    # insertion order and fail.
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
    # The third row's NaN forces pandas to read the numeric columns as
    # float64. The loader must (a) drop the NaN, (b) cast to native python
    # int, (c) sort. Oracle: {512,2048}->[512,2048]; {16,4}->[4,16]; and
    # every element is a python int (a float64 column left un-cast would
    # yield numpy floats -> the JSON UI would show "512.0"; isinstance fails).
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
    # Oracle: 4 rows, only 2 distinct methods -> exactly 2 methods out
    # (a UI dropdown must not list 'lora' twice). set-cardinality invariant.
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
    # Oracle: one row has a missing gpu_model. dropna must remove it, so the
    # result is exactly the non-null gpu, with no NaN/empty entry leaking
    # into the dropdown.
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
    # Contract: a non-existent path must not raise; it returns the hardcoded
    # fallback verbatim. Oracle: equality with get_fallback_options() plus a
    # fallback-only marker ('mixtral-8x7b-instruct-v0.1' is in the fallback
    # but not the real curated catalog), so we know the fallback fired.
    load_available_options.cache_clear()
    opts = load_available_options(tmp_path / "does_not_exist.csv")
    assert opts == get_fallback_options()
    assert "mixtral-8x7b-instruct-v0.1" in opts["models"]


def test_missing_required_column_falls_back_instead_of_crashing(tmp_path):
    # A CSV that lacks the 'method' column raises KeyError inside the loader;
    # the contract is to swallow it and return the fallback, never propagate.
    # Oracle: result equals the fallback set (fallback-only marker present).
    csv = tmp_path / "broken.csv"
    _write_csv(csv, [{"model_name": "m1", "gpu_model": "G1"}])  # no 'method' etc.
    load_available_options.cache_clear()
    opts = load_available_options(csv)
    assert opts == get_fallback_options()
    assert "gptq-lora" in opts["methods"]


def test_data_dir_env_overrides_lookup_location(tmp_path, monkeypatch):
    # Contract: DATA_DIR redirects the default lookup to
    # <DATA_DIR>/profiling-dataset/curated_trace.csv. Oracle: we plant a file
    # there containing a model ('sentinel-model') that appears in neither the
    # real catalog nor the fallback, and assert it comes back -> proves the
    # env path (not the baked-in default and not the fallback) was read.
    dataset = tmp_path / "profiling-dataset"
    dataset.mkdir()
    _write_csv(dataset / "curated_trace.csv", [_row(model_name="sentinel-model")])
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    load_available_options.cache_clear()
    opts = load_available_options()
    assert opts["models"] == ["sentinel-model"]


def test_default_path_points_to_curated_trace_under_profiling_dataset():
    # Regression for the old wrong default (coastline/common/data). Contract:
    # the baked-in default resolves to <trace-archive>/profiling-dataset/
    # curated_trace.csv -- 'curated' belongs to the FILENAME, not a dir.
    assert "trace-archive" in DEFAULT_OPTIONS_PATH.parts
    assert "profiling-dataset" in DEFAULT_OPTIONS_PATH.parts
    assert DEFAULT_OPTIONS_PATH.name == "curated_trace.csv"


@pytest.mark.skipif(
    not DEFAULT_OPTIONS_PATH.exists(),
    reason="curated_trace.csv not present in this environment",
)
def test_real_curated_file_loads_and_is_not_the_fallback():
    # Distinguish a genuine load from a silent fallback. 'llama3.2-3b' exists
    # ONLY in the real curated catalog, and 'mixtral-8x7b-instruct-v0.1'
    # exists ONLY in the fallback. A real load has the former and not the
    # latter; a fallback has it reversed. (The previous version asserted
    # 'granite-3.3-8b'/'mistral-7b-v0.1', which are in BOTH sets and so could
    # never detect a fallback.)
    load_available_options.cache_clear()
    opts = load_available_options()
    assert "llama3.2-3b" in opts["models"]
    assert "mixtral-8x7b-instruct-v0.1" not in opts["models"]
