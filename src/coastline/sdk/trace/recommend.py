"""Recommend a config for every job in a fine-tuning trace.

Replaces the GPU layout columns and appends:
- ``metadata.estimated_throughput_<method>`` — always written; the predicted
  tokens/sec under the recommended config.
- ``metadata.estimated_duration_<method>`` — written only when ``tot_tokens_col``
  resolves to a non-null value for the row.

  When ``setup_time_col`` is also provided:
    ``estimated_duration = setup_time + tot_tokens_col / estimated_throughput``
  Otherwise (legacy):
    ``estimated_duration = tot_tokens_col / estimated_throughput``

``tot_tokens_col`` holds the config-independent total token count for the job
(e.g. ``metadata.output.extrapolated_num_tokens``).
``setup_time_col`` holds the per-job setup overhead
(e.g. ``metadata.output.setup_time``), which is added back so the estimated
duration matches the identity from ``add_auxiliary_information.py``:
  extrapolated_duration = setup_time + extrapolated_num_tokens / tps

With ``--visual`` it also renders the operational cluster timeline (the recommended
configs FIFO-scheduled — GPUs in use + jobs queued over time) to a PDF.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

import coastline
from coastline.sdk.constants import FeasibilityMode
from coastline.sdk.io.infrastructure import resolve_cluster_caps

logger = logging.getLogger(__name__)

# --- trace schema column names (the one home; also imported by trace.to_runs) ---
_MODEL = "metadata.model_name"
_METHOD = "metadata.method"
_GPU = "resources.gpu_model"
_TOKENS = "metadata.tokens_per_sample"  # the int the user wants; nominal seq length is fine here
_BATCH = "metadata.batch_size"
_GPN = "resources.num_gpus_per_node"  # GPUs per node, never the cluster total
_NODES = "resources.num_nodes"
# ground-truth work (config-independent): tokens the job actually processed
# These are used ONLY for the fallback path (_unchanged) and legacy _job_total_tokens.
_ACT_TPS = "metadata.output.train_tokens_per_second"
_ACT_RUNTIME = "metadata.train_runtime"
# observed job duration — the fallback when no recommendation can be made
_ACT_DURATION = "metadata.output.extrapolated_duration"

# When BOTH of these columns are present in the trace the recommended effective
# batch size is NOT written back to _BATCH / _ORIG_PER_DEVICE (they record the
# original submitted config and must not be touched).  Instead the per-device
# equivalent for the new layout is derived and written to _REC_PER_DEVICE.
_ORIG_PER_DEVICE = "metadata.orig_per_device_train_batch_size"
_REC_PER_DEVICE = "per_device_train_batch_size"

_NO_TOT_TOKENS = None  # sentinel: tot_tokens_col not provided

# coastline predictor keys for the trace's method names
_METHOD_TO_PREDICTOR = {"kavier": "kavier", "tabpfn": "tabpfn", "xgb": "xgboost", "xgboost": "xgboost"}

# reader-first output layout: the columns a human scans for, in this order
_FRONT_COLS = [
    "metadata.submission_time_issue_85_rescaled",
    _MODEL,
    _GPU,
    _GPN,
    _NODES,
    _BATCH,
    _REC_PER_DEVICE,
]


def _tidy_columns(df: pd.DataFrame, throughput_col: str, duration_col: Optional[str]) -> pd.DataFrame:
    """Put the decision columns first and drop raw fine-tuning args (adam_beta1, ...).

    Only dotted ``metadata.*`` / ``resources.*`` columns are kept after the front
    block — the flat launcher-arg columns carry no signal for trace analysis.
    """
    priority = [throughput_col]
    if duration_col:
        priority.append(duration_col)
    priority.append("metadata.recommendation_note")
    front = [c for c in [*_FRONT_COLS, *priority] if c in df.columns]
    rest = [c for c in df.columns if c not in front and "." in c]
    return df[front + rest]


def _as_int(value: Any) -> Optional[int]:
    n = pd.to_numeric(value, errors="coerce")
    return int(n) if pd.notna(n) and n >= 1 else None


def _job_total_tokens(row: pd.Series, tot_tokens_col: Optional[str]) -> Optional[float]:
    """Config-independent total token count for the job.

    When ``tot_tokens_col`` is provided, reads that column directly — the caller is
    responsible for pre-computing the value (e.g. max_seq_length × batch × steps).

    Legacy fallback (``tot_tokens_col`` is None): derives the value from the measured
    run outputs ``metadata.output.train_tokens_per_second × metadata.train_runtime``.
    This requires output data and should not be used for forward predictions on new jobs.
    """
    if tot_tokens_col is not None:
        v = pd.to_numeric(row.get(tot_tokens_col), errors="coerce")
        return float(v) if pd.notna(v) and v > 0 else None
    # legacy path — output columns required
    tps = pd.to_numeric(row.get(_ACT_TPS), errors="coerce")
    rt = pd.to_numeric(row.get(_ACT_RUNTIME), errors="coerce")
    if pd.notna(tps) and pd.notna(rt) and tps > 0 and rt > 0:
        return float(tps) * float(rt)
    return None


def _kavier_can_predict(wl: dict[str, Any], goal: str, feasibility: str, max_gpus: int) -> bool:
    """True when the kavier physics path yields a feasible config with a throughput."""
    try:
        out = coastline.recommend(
            [wl], predictor="kavier", goal=goal, max_gpus=max_gpus, top_k=1, feasibility=feasibility
        )
        if out.empty or not bool(out.iloc[0]["feasible"]):
            return False
        thr = out.iloc[0]["throughput_tok_s"]
        return bool(pd.notna(thr) and thr > 0)
    except Exception:
        return False


def _infeasible_note(max_gpus: int, feasibility: str) -> str:
    cause = (
        "every config would run out of GPU memory (autoconf OOM check)"
        if feasibility == FeasibilityMode.AUTOCONF
        else "no config passes the divisibility rules"
    )
    return f"infeasible within {max_gpus} GPUs: {cause}"


def _observed_duration(row: pd.Series) -> Optional[float]:
    """The job's measured duration (extrapolated_duration, else train_runtime), if positive."""
    for col in (_ACT_DURATION, _ACT_RUNTIME):
        v = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(v) and v > 0:
            return float(v)
    return None


def _unchanged(keep: dict[str, Any], row: pd.Series, reason: str) -> dict[str, Any]:
    """The job appears in the trace UNCHANGED: original config + observed duration.

    This keeps unrecommendable jobs in the cluster replay (the scheduler still
    received them) instead of silently dropping them from the timeline.
    """
    keep["thr"] = None
    keep["dur"] = _observed_duration(row)
    tail = (
        "job kept unchanged (original config + observed duration)"
        if keep["dur"] is not None
        else "job kept with the original config but NO observed duration — it will be missing from the timeline"
    )
    keep["note"] = f"{reason} — {tail}"
    return keep


def _recommend_row(
    row: pd.Series,
    predictor: str,
    goal: str,
    feasibility: str,
    max_gpus: int,
    lookup: Optional[str] = None,
    tokens_col: str = _TOKENS,
    tot_tokens_col: Optional[str] = _NO_TOT_TOKENS,
    setup_time_col: Optional[str] = None,
) -> dict[str, Any]:
    """Recommend a layout for one trace row; fall back to the original layout on any failure.

    ``max_gpus`` is the cluster GPU budget (from infrastructure.yaml or ``--cluster-gpus``): every
    job is optimised within the SAME cluster ceiling, never past it, and never keyed off the job's
    own submitted footprint (a trace never carries cluster size).

    ``tokens_col`` is the sequence-length column fed to the physics/ML predictor.
    ``tot_tokens_col`` is the pre-computed total-token-count column used to compute
    the duration estimate.  When None, duration is not computed (throughput only).
    ``setup_time_col`` is an optional column holding per-job setup overhead.  When
    provided, the duration formula is:
        estimated_duration = setup_time + tot_tokens / throughput
    matching the ``add_auxiliary_information.py`` identity exactly.

    The returned dict carries:
    - ``thr``:  predicted throughput (tok/s) under the recommended config, or None on failure.
    - ``dur``:  estimated duration (s), or None when tot_tokens_col is absent/null.
               Falls back to the legacy tps×runtime path only when tot_tokens_col is None.
    - ``note``: None on full success; otherwise the reason the row was kept unchanged.
    """
    keep = {"nodes": row.get(_NODES), "gpn": row.get(_GPN), "batch": row.get(_BATCH), "per_device": None, "thr": None, "dur": None, "note": None}
    tokens, batch = _as_int(row.get(tokens_col)), _as_int(row.get(_BATCH))
    gpn, nodes = _as_int(row.get(_GPN)), _as_int(row.get(_NODES))
    if not (tokens and batch and gpn and nodes):
        return _unchanged(keep, row, "no recommendation: missing/invalid workload fields")

    # When the trace carries per_device_train_batch_size (the flat SFTTrainer launcher
    # arg), use it as the batch_size pivot for the recommender.  Kavier's physics engine
    # treats batch_size as PER-DEVICE and multiplies internally by num_gpus — so feeding
    # the total effective batch (metadata.batch_size) would overcount by num_gpus.
    # Fall back to metadata.batch_size only when the per-device column is absent (legacy traces).
    per_device_orig = _as_int(row.get(_REC_PER_DEVICE))
    if per_device_orig is not None:
        wl_batch = per_device_orig
        logger.debug(
            "row (%s): using %s=%d as batch_size pivot (metadata.batch_size=%d is total effective)",
            row.get(_MODEL, "?"), _REC_PER_DEVICE, per_device_orig, batch,
        )
    else:
        wl_batch = batch

    wl = {
        "llm_model": str(row[_MODEL]),
        "fine_tuning_method": str(row[_METHOD]),
        "gpu_model": str(row[_GPU]),
        "tokens_per_sample": tokens,
        "batch_size": wl_batch,
    }

    def kavier_hint() -> str:
        if predictor == "kavier":
            return ""
        if _kavier_can_predict(wl, goal, feasibility, max_gpus):
            return " — kavier CAN handle this workload: rerun with --method kavier"
        return ""

    try:
        out = coastline.recommend(
            [wl], predictor=predictor, goal=goal, max_gpus=max_gpus, top_k=1, feasibility=feasibility, lookup=lookup
        )
        if out.empty or not bool(out.iloc[0]["feasible"]):
            hint = kavier_hint()
            reason = (
                f"'{predictor}' could not predict this workload{hint}"
                if hint
                else _infeasible_note(max_gpus, feasibility)
            )
            return _unchanged(keep, row, reason)
        top = out.iloc[0]
        thr = top["throughput_tok_s"]
        if not (pd.notna(thr) and thr > 0):
            return _unchanged(keep, row, f"'{predictor}' returned no throughput for this workload{kavier_hint()}")
        # Duration: use tot_tokens_col when provided; legacy fallback when not.
        total_tokens = _job_total_tokens(row, tot_tokens_col)
        setup_time: Optional[float] = None
        if setup_time_col is not None:
            v = pd.to_numeric(row.get(setup_time_col), errors="coerce")
            if pd.notna(v) and v >= 0:
                setup_time = float(v)
        if total_tokens:
            dur = (setup_time or 0.0) + total_tokens / thr
        else:
            dur = None
        if dur is None and tot_tokens_col is not None:
            # col was provided but this row has no value — warn but still write throughput
            logger.warning(
                "row (%s): tot_tokens_col '%s' is null — throughput written, duration skipped",
                row.get(_MODEL, "?"), tot_tokens_col,
            )
        elif dur is None and tot_tokens_col is None:
            # legacy path: no output data available
            logger.warning(
                "row (%s): no tot_tokens_col and no measured tps/runtime — throughput written, duration skipped",
                row.get(_MODEL, "?"),
            )
        rec_batch = _as_int(top["batch_size"]) or batch
        rec_nodes = int(top["number_of_nodes"])
        rec_gpn = int(top["gpus_per_node"])
        num_devices = rec_gpn * rec_nodes
        per_device = max(1, rec_batch // num_devices) if num_devices > 0 else rec_batch
        return {
            "nodes": rec_nodes,
            "gpn": rec_gpn,
            "batch": rec_batch,
            "per_device": per_device,
            "thr": float(thr),
            "dur": dur,
            "note": None,
        }
    except Exception as exc:  # one bad row must not sink the whole trace
        return _unchanged(keep, row, f"'{predictor}' failed ({type(exc).__name__}){kavier_hint()}")


def recommend_trace(
    input_csv: str,
    output_csv: str,
    *,
    method: str = "kavier",
    goal: str = "min_gpu",
    feasibility: str = "autoconf",
    lookup: Optional[str] = None,
    cluster_gpus: Optional[int] = None,
    node_gpus: Optional[int] = None,
    tokens_col: str = _TOKENS,
    tot_tokens_col: Optional[str] = _NO_TOT_TOKENS,
    setup_time_col: Optional[str] = None,
) -> pd.DataFrame:
    """Recommend a layout per trace row, write the recommended-trace CSV, and return the DataFrame.

    ``feasibility="autoconf"`` (default) runs the real AutoConf OOM check — it
    fail-closes if AutoConf (the ``coastline[autoconf]`` extra) is
    not installed (use ``COASTLINE_ALLOW_RULES_FALLBACK=1`` to suppress).  Pass
    ``feasibility="rules"`` to use the divisibility-only path (no OOM check,
    works on any install).

    ``lookup`` points the ``cache``/``intelligent`` methods at a measured-runs CSV
    (flat sfttrainer schema), or ``"default"`` for the small bundled lookup DB.

    ``cluster_gpus`` / ``node_gpus`` bound every job to the cluster: they resolve (with
    ``infrastructure.yaml`` as the default) to the GPU budget each job is optimised within, so no
    recommendation ever exceeds the cluster. The cluster size is never taken from the trace.

    ``tokens_col`` overrides the default ``metadata.tokens_per_sample`` column used as the
    ``tokens_per_sample`` input to the physics/ML predictors. Override with e.g.
    ``metadata.estimated_max_seq_length`` to feed the actual (shrunk) sequence length
    instead of the nominal dataset max.

    ``tot_tokens_col`` is the column holding the pre-computed config-independent total
    token count for each job (e.g. ``metadata.output.extrapolated_num_tokens``).
    ``setup_time_col`` is the column holding per-job setup overhead
    (e.g. ``metadata.output.setup_time``).  When both are provided, the duration
    formula is ``setup_time + tot_tokens / throughput``, matching the identity in
    ``add_auxiliary_information.py``.  When only ``tot_tokens_col`` is given, the
    legacy ``tot_tokens / throughput`` formula is used.  When neither is provided,
    ``train_tokens_per_second × train_runtime`` is the fallback (requires output data).
    In all cases ``metadata.estimated_throughput_<method>`` is always written.
    """
    total_gpus, _, _ = resolve_cluster_caps(cluster_gpus, node_gpus)
    predictor = _METHOD_TO_PREDICTOR.get(method.lower(), method.lower())
    df = pd.read_csv(input_csv, low_memory=False)
    recs = [
        _recommend_row(
            row, predictor, goal, feasibility, total_gpus, lookup,
            tokens_col=tokens_col, tot_tokens_col=tot_tokens_col,
            setup_time_col=setup_time_col,
        )
        for _, row in df.iterrows()
    ]
    for i, r in enumerate(recs):
        if r["note"]:
            logger.warning("row %d (%s): %s", i, df.iloc[i].get(_MODEL, "?"), r["note"])

    thr_col = f"metadata.estimated_throughput_{method}"
    dur_col = f"metadata.estimated_duration_{method}"

    df[_NODES] = [r["nodes"] for r in recs]
    df[_GPN]   = [r["gpn"]   for r in recs]

    # Batch-size write-back: when the trace carries both the total effective
    # batch size (metadata.batch_size) AND the original per-device value
    # (metadata.orig_per_device_train_batch_size), those two columns record the
    # original submitted config and must NOT be overwritten.  Instead the
    # recommended effective batch size is converted to a per-device equivalent
    # for the new layout and written to a separate column.
    has_orig_per_device = (
        _ORIG_PER_DEVICE in df.columns and _BATCH in df.columns
    )
    if has_orig_per_device:
        logger.info(
            "Trace contains both '%s' and '%s': leaving original batch-size "
            "metadata untouched and writing recommended per-device batch size "
            "to '%s' (= recommended_effective_batch / (gpus_per_node × nodes)).",
            _BATCH,
            _ORIG_PER_DEVICE,
            _REC_PER_DEVICE,
        )
        df[_REC_PER_DEVICE] = [r["per_device"] for r in recs]
    else:
        # Legacy path: no per-device origin tracking; overwrite metadata.batch_size
        # with the recommended effective batch size as before.
        df[_BATCH] = [r["batch"] for r in recs]

    df[thr_col] = [r["thr"] for r in recs]
    df[dur_col] = [r["dur"] for r in recs]
    df["metadata.recommendation_note"] = [r["note"] for r in recs]

    has_duration = df[dur_col].notna().any()
    df = _tidy_columns(df, thr_col, dur_col if has_duration else None)
    df.to_csv(output_csv, index=False)
    return df
