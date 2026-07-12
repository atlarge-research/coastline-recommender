"""Tests for the trace -> flat measured-runs converter (coastline/sdk/trace/to_runs.py).

`coastline trace-to-runs` turns a fine-tuning TRACE CSV (dotted ``metadata.*``/``resources.*``
columns) into the FLAT measured-runs schema that ``coastline tune``, the cache/intelligent
retrieval lookup, and ``kavier calibrate`` all consume. Oracles here are the exact column
rename map and the ``is_valid`` derivation rule (positive throughput AND runtime).
"""

import pandas as pd
import pytest

from coastline.sdk.trace.to_runs import _FLAT_REQUIRED, trace_to_runs

# A trace row Kavier knows (granite-8b / A100 / lora), with observed throughput + runtime.
_TRACE_ROW = {
    "metadata.model_name": "granite-3.1-8b-instruct",
    "metadata.method": "lora",
    "resources.gpu_model": "NVIDIA-A100-SXM4-80GB",
    "metadata.tokens_per_sample": 2048,
    "metadata.batch_size": 8,
    "resources.num_gpus_per_node": 8,
    "resources.num_nodes": 1,
    "metadata.output.train_tokens_per_second": 15000.0,
    "metadata.train_runtime": 3600.0,
    "metadata.uid": "job-1",  # an extra trace column that must be dropped
}


def _write_csv(tmp_path, rows, name="trace.csv"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_trace_is_renamed_to_the_flat_schema(tmp_path):
    """Every trace column maps to its flat name; the output carries exactly the flat schema."""
    out = tmp_path / "runs.csv"
    df = trace_to_runs(str(_write_csv(tmp_path, [_TRACE_ROW])), str(out))

    assert list(df.columns) == [*_FLAT_REQUIRED, "is_valid"]
    row = df.iloc[0]
    assert row["model_name"] == "granite-3.1-8b-instruct"
    assert row["method"] == "lora"
    assert row["gpu_model"] == "NVIDIA-A100-SXM4-80GB"
    assert int(row["number_nodes"]) == 1
    assert int(row["number_gpus"]) == 8
    assert int(row["tokens_per_sample"]) == 2048
    assert int(row["batch_size"]) == 8
    assert row["dataset_tokens_per_second"] == 15000.0
    assert row["train_runtime"] == 3600.0
    # the extra trace column is not carried through
    assert "metadata.uid" not in df.columns
    # written file round-trips to the same schema
    assert list(pd.read_csv(out).columns) == [*_FLAT_REQUIRED, "is_valid"]


def test_is_valid_is_derived_from_positive_targets(tmp_path):
    """is_valid = (dataset_tokens_per_second > 0) AND (train_runtime > 0)."""
    good = {**_TRACE_ROW, "metadata.uid": "good"}
    zero_tps = {**_TRACE_ROW, "metadata.uid": "zero-tps", "metadata.output.train_tokens_per_second": 0.0}
    no_runtime = {**_TRACE_ROW, "metadata.uid": "no-runtime", "metadata.train_runtime": 0.0}
    df = trace_to_runs(str(_write_csv(tmp_path, [good, zero_tps, no_runtime])))

    assert df["is_valid"].tolist() == [1.0, 0.0, 0.0]


def test_already_flat_input_passes_through(tmp_path):
    """A CSV already in the flat schema is returned unchanged (idempotent); missing is_valid is filled."""
    flat_rows = [
        {
            "model_name": "granite-3.1-2b",
            "method": "full",
            "gpu_model": "NVIDIA-A100-SXM4-80GB",
            "number_nodes": 1,
            "number_gpus": 2,
            "tokens_per_sample": 4096,
            "batch_size": 4,
            "dataset_tokens_per_second": 5000.0,
            "train_runtime": 1200.0,
        }
    ]
    df = trace_to_runs(str(_write_csv(tmp_path, flat_rows, name="flat.csv")))

    assert list(df.columns) == [*_FLAT_REQUIRED, "is_valid"]
    assert df.iloc[0]["model_name"] == "granite-3.1-2b"
    assert df.iloc[0]["is_valid"] == 1.0  # derived because the flat input omitted it


def test_flat_input_keeps_its_own_is_valid(tmp_path):
    """When the flat input already carries is_valid, it is preserved (not recomputed)."""
    flat_rows = [
        {
            "model_name": "granite-3.1-2b",
            "method": "lora",
            "gpu_model": "NVIDIA-A100-SXM4-80GB",
            "number_nodes": 1,
            "number_gpus": 1,
            "tokens_per_sample": 512,
            "batch_size": 1,
            "dataset_tokens_per_second": 900.0,
            "train_runtime": 300.0,
            "is_valid": 0.0,  # explicitly marked invalid despite positive targets
        }
    ]
    df = trace_to_runs(str(_write_csv(tmp_path, flat_rows, name="flat.csv")))

    assert df.iloc[0]["is_valid"] == 0.0


def test_unrecognized_schema_raises(tmp_path):
    """A CSV that is neither a trace nor the flat schema fails loudly."""
    junk = [{"foo": 1, "bar": 2}]
    with pytest.raises(ValueError, match="neither a fine-tuning trace nor a flat measured-runs CSV"):
        trace_to_runs(str(_write_csv(tmp_path, junk, name="junk.csv")))
