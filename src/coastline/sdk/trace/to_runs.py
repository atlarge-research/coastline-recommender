"""Convert a fine-tuning trace CSV to the flat measured-runs schema.

A *trace* CSV has dotted ``metadata.*`` / ``resources.*`` columns (one recorded job per
row — the input to ``coastline recommend-trace``). ``coastline tune``, the cache/intelligent
retrieval lookup (``$DATA_DIR/profiling-dataset/raw_trace.csv``), and ``kavier calibrate`` all
consume the *flat* measured-runs schema instead:

    model_name, method, gpu_model, number_nodes, number_gpus, tokens_per_sample,
    batch_size, dataset_tokens_per_second, train_runtime, is_valid

``trace_to_runs`` bridges the two. It is idempotent: a CSV that is already in the flat schema is
passed through unchanged (so callers can feed either shape). ``is_valid`` is derived from the
targets when absent (a row is valid iff its observed throughput and runtime are both positive).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from coastline.sdk.trace.recommend import (
    _ACT_RUNTIME,
    _ACT_TPS,
    _BATCH,
    _GPN,
    _GPU,
    _METHOD,
    _MODEL,
    _NODES,
    _REC_PER_DEVICE,
    _TOKENS,
)

# trace column -> flat measured-runs column
_TRACE_TO_FLAT: dict[str, str] = {
    _MODEL: "model_name",
    _METHOD: "method",
    _GPU: "gpu_model",
    _NODES: "number_nodes",
    _GPN: "number_gpus",
    _TOKENS: "tokens_per_sample",
    _BATCH: "batch_size",
    _ACT_TPS: "dataset_tokens_per_second",
    _ACT_RUNTIME: "train_runtime",
}

# the flat schema, in canonical order (matches cache_predictor / tune REQUIRED_COLUMNS)
_FLAT_REQUIRED: list[str] = list(_TRACE_TO_FLAT.values())


def _derive_is_valid(flat: pd.DataFrame) -> pd.Series:
    """A row is valid iff its observed throughput and runtime are both present and positive."""
    thr = pd.to_numeric(flat["dataset_tokens_per_second"], errors="coerce")
    rt = pd.to_numeric(flat["train_runtime"], errors="coerce")
    return ((thr > 0) & (rt > 0)).astype(float)


def trace_to_runs(input_csv: str, output_csv: Optional[str] = None) -> pd.DataFrame:
    """Return the flat measured-runs DataFrame for ``input_csv``; also write it when ``output_csv`` is given.

    Accepts either a fine-tuning trace CSV (dotted columns) or an already-flat measured-runs CSV.
    Raises ``ValueError`` when the input is neither.
    """
    df = pd.read_csv(input_csv, low_memory=False)

    # The flat schema's batch_size is PER-DEVICE (Kavier/calibrate convention). Prefer the trace's
    # per_device_train_batch_size when present; else fall back to metadata.batch_size (the total
    # effective batch — a known imprecision on legacy traces without the per-device column).
    mapping = dict(_TRACE_TO_FLAT)
    if _REC_PER_DEVICE in df.columns:
        mapping.pop(_BATCH, None)
        mapping[_REC_PER_DEVICE] = "batch_size"

    if set(_FLAT_REQUIRED).issubset(df.columns):
        flat = df.copy()  # already flat — pass through
    elif set(mapping).issubset(df.columns):
        flat = df.rename(columns=mapping)
    else:
        missing = [c for c in mapping if c not in df.columns]
        raise ValueError(
            "input is neither a fine-tuning trace nor a flat measured-runs CSV; "
            f"missing trace columns: {missing}. Provide either the flat schema "
            f"({', '.join(_FLAT_REQUIRED)}) or the trace columns ({', '.join(mapping)})."
        )

    if "is_valid" not in flat.columns:
        flat["is_valid"] = _derive_is_valid(flat)
    flat = flat[[*_FLAT_REQUIRED, "is_valid"]]

    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        flat.to_csv(output_csv, index=False)
    return flat
