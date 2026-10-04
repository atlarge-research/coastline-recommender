"""Recommend a config for every job in a fine-tuning trace.

Replaces the GPU layout columns and adds:
- ``metadata.estimated_throughput_<method>``: predicted tokens/s under the recommended config,
  always written.
- ``metadata.estimated_duration_<method>``: written when the row's total token count is known.
  With ``setup_time_col`` it is ``setup_time + tot_tokens / estimated_throughput``, otherwise
  ``tot_tokens / estimated_throughput``.

``tot_tokens_col`` holds the job's config-independent total token count (for example
``metadata.output.extrapolated_num_tokens``). ``setup_time_col`` holds the per-job setup time
(for example ``metadata.output.setup_time``); adding it back matches the identity in
``add_auxiliary_information.py``:
  extrapolated_duration = setup_time + extrapolated_num_tokens / tps

With ``--visual`` the CLI also draws the cluster timeline of the recommended configs (FIFO
schedule: GPUs in use and jobs queued over time) to a PDF.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

import coastline
from coastline.sdk.constants import DEFAULT_BATCH_SIZES, DEFAULT_GOAL, FeasibilityMode, Strategy
from coastline.sdk.io.infrastructure import resolve_cluster_caps
from coastline.sdk.policies import normalize_predictor
from coastline.sdk.recommend._goals import normalize_goal
from coastline.sdk.recommend.engine import StrategyCache

logger = logging.getLogger(__name__)

# Trace column names (also used by trace.to_runs).
_MODEL = "metadata.model_name"
_METHOD = "metadata.method"
_GPU = "resources.gpu_model"
_TOKENS = "metadata.tokens_per_sample"  # nominal sequence length
_BATCH = "metadata.batch_size"
_GPN = "resources.num_gpus_per_node"  # GPUs in each node of the job
_NODES = "resources.num_nodes"
# Measured outputs, used by _unchanged and by the legacy path of _job_total_tokens.
_ACT_TPS = "metadata.output.train_tokens_per_second"
_ACT_RUNTIME = "metadata.train_runtime"
# observed job duration, used when no recommendation can be made
_ACT_DURATION = "metadata.output.extrapolated_duration"

# Kavier's batch_size is per device (it multiplies by the GPU count itself); metadata.batch_size
# is the total effective batch (per_device x gpn x nodes) for every goal. VV reads the
# recommendation from per_device_train_batch_size, and metadata.batch_size is recomputed from it.
# A trace is in per-device mode when either of these columns is present; without them (legacy
# mode) the per-device batch is metadata.batch_size / (gpn x nodes).
_REC_PER_DEVICE = "per_device_train_batch_size"  # write target for the recommended per-device batch
_ORIG_PER_DEVICE = "metadata.orig_per_device_train_batch_size"  # seed fallback (original per-device)
_PER_DEVICE_COLS = (_REC_PER_DEVICE, _ORIG_PER_DEVICE)

_NO_TOT_TOKENS = None  # sentinel: tot_tokens_col not provided

# Read with row[...] for every row, so a trace without one of them cannot be recommended at all.
_IDENTITY_COLS = (_MODEL, _METHOD, _GPU)

# coastline predictor keys for the trace's method names
_METHOD_TO_PREDICTOR = {"kavier": "kavier", "tabpfn": "tabpfn", "xgb": "xgboost", "xgboost": "xgboost"}

# Output columns shown first, in this order.
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
    """Put the decision columns first and keep every other input column unchanged.

    Only the layout and estimate columns change. The others, including launcher arguments such as
    ``model_name_or_path``, ``learning_rate`` and ``optim``, pass through as they are, so a
    recommended row is still a complete job spec for the workload-generator replayer.
    """
    priority = [throughput_col]
    if duration_col:
        priority.append(duration_col)
    priority.append("metadata.recommendation_note")
    front = [c for c in [*_FRONT_COLS, *priority] if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


def _as_int(value: Any) -> Optional[int]:
    n = pd.to_numeric(value, errors="coerce")
    return int(n) if pd.notna(n) and n >= 1 else None


def _job_total_tokens(row: pd.Series, tot_tokens_col: Optional[str]) -> Optional[float]:
    """Config-independent total token count for the job.

    With ``tot_tokens_col``, the value of that column, precomputed by the caller (for example
    max_seq_length x batch x steps). Without it (legacy), the measured
    ``metadata.output.train_tokens_per_second x metadata.train_runtime``, which needs output data
    and does not suit predictions for new jobs.
    """
    if tot_tokens_col is not None:
        v = pd.to_numeric(row.get(tot_tokens_col), errors="coerce")
        return float(v) if pd.notna(v) and v > 0 else None
    # legacy path: needs the output columns
    tps = pd.to_numeric(row.get(_ACT_TPS), errors="coerce")
    rt = pd.to_numeric(row.get(_ACT_RUNTIME), errors="coerce")
    if pd.notna(tps) and pd.notna(rt) and tps > 0 and rt > 0:
        return float(tps) * float(rt)
    return None


def _kavier_can_predict(
    wl: dict[str, Any],
    goal: str,
    feasibility: str,
    max_gpus: int,
    strategy_cache: Optional[StrategyCache] = None,
    gpus_per_node: Optional[int] = None,
) -> bool:
    """True when the kavier physics path yields a feasible config with a throughput."""
    try:
        out = coastline.recommend(
            [wl],
            predictor="kavier",
            goal=goal,
            max_gpus=max_gpus,
            top_k=1,
            feasibility=feasibility,
            strategy_cache=strategy_cache,
            max_gpus_per_node=gpus_per_node,
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
        else "no config passes the structural feasibility guards"
    )
    return f"infeasible within {max_gpus} GPUs: {cause}"


def _engine_error(out: pd.DataFrame) -> Optional[str]:
    """The error text of the engine's top row, or None when it gives none."""
    error = None if out.empty else out.iloc[0].get("error")
    return error if isinstance(error, str) and error.strip() else None


def _failure_note(out: pd.DataFrame, max_gpus: int, feasibility: str) -> str:
    """Why the engine gave no recommendation for a row.

    The infeasible note when no config in the grid passed the feasibility check (the engine's
    "no feasible ..." errors), else the engine's own error, for example an unknown model.
    """
    error = _engine_error(out)
    if error is None or "no feasible" in error:
        return _infeasible_note(max_gpus, feasibility)
    return f"no recommendation: {error}"


def _observed_duration(row: pd.Series) -> Optional[float]:
    """The job's measured duration (extrapolated_duration, else train_runtime), if positive."""
    for col in (_ACT_DURATION, _ACT_RUNTIME):
        v = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(v) and v > 0:
            return float(v)
    return None


def _unchanged(keep: dict[str, Any], row: pd.Series, reason: str) -> dict[str, Any]:
    """Keep the job with its original config and observed duration.

    Jobs without a recommendation stay in the cluster replay, since the scheduler received them.
    """
    keep["thr"] = None
    keep["dur"] = _observed_duration(row)
    tail = (
        "job kept unchanged (original config + observed duration)"
        if keep["dur"] is not None
        else "job kept with the original config but no observed duration, so it is missing from the timeline"
    )
    keep["note"] = f"{reason} - {tail}"
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
    per_device_mode: bool = False,
    strategy_cache: Optional[StrategyCache] = None,
    workers: int = 1,
    gpus_per_node: Optional[int] = None,
) -> dict[str, Any]:
    """Recommend a layout for one trace row; keep the original layout on any failure.

    ``max_gpus`` is the cluster GPU budget and ``gpus_per_node`` the cluster's node width (from
    infrastructure.yaml or the CLI flags); every job is optimized within them, whatever its own
    footprint. ``tokens_col`` is the sequence-length column given to the predictor. The duration
    uses ``tot_tokens_col`` (the legacy tps x runtime when None) plus ``setup_time_col`` if given.

    Returns a dict with the layout and batch keys, ``thr`` (predicted throughput [tokens/s], None
    on failure), ``dur`` (estimated duration [s], None without a total token count) and ``note``
    (None on success, else why the row was kept unchanged).
    """
    # In per-device mode the recommendation starts from the per-device batch and is written to
    # per_device_train_batch_size. A row kept unchanged keeps its per-device and total batch.
    seed_pd = _as_int(row.get(_REC_PER_DEVICE)) or _as_int(row.get(_ORIG_PER_DEVICE))
    keep = {
        "nodes": row.get(_NODES),
        "gpn": row.get(_GPN),
        "batch": row.get(_BATCH),
        "per_device": seed_pd,
        "thr": None,
        "dur": None,
        "note": None,
    }
    tokens, batch = _as_int(row.get(tokens_col)), _as_int(row.get(_BATCH))
    gpn, nodes = _as_int(row.get(_GPN)), _as_int(row.get(_NODES))
    if not (tokens and batch and gpn and nodes):
        return _unchanged(keep, row, "no recommendation: missing/invalid workload fields")
    if goal == Strategy.MIN_GPU:
        # min_gpu keeps the job's total batch: per-device x gpn x nodes in per-device mode, the
        # trace's total in legacy mode. The workload below gives no layout, so the engine takes
        # this seed as the total batch of a 1-GPU job.
        seed_batch = seed_pd * gpn * nodes if (per_device_mode and seed_pd) else batch
    elif per_device_mode:
        # Kavier's batch_size is per device, so per-device mode seeds with the per-device value;
        # the DEFAULT_BATCH_SIZES sweep below replaces the seed.
        seed_batch = seed_pd or batch
    elif batch % (gpn * nodes):
        return _unchanged(
            keep, row, f"no recommendation: the total batch {batch} does not split evenly over {gpn * nodes} GPUs"
        )
    else:
        # Legacy mode: the per-device batch of the job's total.
        seed_batch = batch // (gpn * nodes)
    wl = {
        "llm_model": str(row[_MODEL]),
        "fine_tuning_method": str(row[_METHOD]),
        "gpu_model": str(row[_GPU]),
        "tokens_per_sample": tokens,
        "batch_size": seed_batch,
    }

    def kavier_hint() -> str:
        if predictor == "kavier":
            return ""
        if _kavier_can_predict(wl, goal, feasibility, max_gpus, strategy_cache, gpus_per_node):
            return " - kavier can handle this workload: rerun with --method kavier"
        return ""

    try:
        # Per-device mode sweeps all of DEFAULT_BATCH_SIZES; legacy mode uses the batch API's
        # neighbours of the seed. min_gpu searches no grid.
        sweep = {"batch_sizes": list(DEFAULT_BATCH_SIZES)} if per_device_mode else {}
        out = coastline.recommend(
            [wl],
            predictor=predictor,
            goal=goal,
            max_gpus=max_gpus,
            top_k=1,
            feasibility=feasibility,
            lookup=lookup,
            strategy_cache=strategy_cache,
            workers=workers,
            max_gpus_per_node=gpus_per_node,
            **sweep,
        )
        if out.empty or not bool(out.iloc[0]["feasible"]):
            hint = kavier_hint()
            if hint:
                error = _engine_error(out)
                detail = f": {error}" if error else ""
                reason = f"'{predictor}' could not predict this workload{detail}{hint}"
            else:
                reason = _failure_note(out, max_gpus, feasibility)
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
            # column given but empty for this row: warn and still write the throughput
            logger.warning(
                "row (%s): tot_tokens_col '%s' is null - throughput written, duration skipped",
                row.get(_MODEL, "?"),
                tot_tokens_col,
            )
        elif dur is None and tot_tokens_col is None:
            # legacy path: no output data available
            logger.warning(
                "row (%s): no tot_tokens_col and no measured tps/runtime - throughput written, duration skipped",
                row.get(_MODEL, "?"),
            )
        rec_gpn = int(top["gpus_per_node"])
        rec_nodes = int(top["number_of_nodes"])
        rec_pd = _as_int(top["batch_size"])  # the engine treats batch_size as per-device
        if per_device_mode and rec_pd is None:
            # the total batch is not a per-device batch, so keep the row unchanged
            return _unchanged(keep, row, f"'{predictor}' returned no batch size - kept unchanged")
        # metadata.batch_size is the total effective batch: per_device x gpn x nodes. A legacy
        # row without a recommended batch keeps its total.
        total_batch = rec_pd * rec_gpn * rec_nodes if rec_pd is not None else batch
        return {
            "nodes": rec_nodes,
            "gpn": rec_gpn,
            "batch": total_batch,
            "per_device": rec_pd if per_device_mode else None,
            "thr": float(thr),
            "dur": dur,
            "note": None,
        }
    except Exception as exc:  # one bad row does not stop the trace
        return _unchanged(keep, row, f"'{predictor}' failed ({type(exc).__name__}){kavier_hint()}")


def recommend_trace(
    input_csv: str,
    output_csv: str,
    *,
    method: str = "kavier",
    goal: str = DEFAULT_GOAL,
    feasibility: str = "autoconf",
    lookup: Optional[str] = None,
    cluster_gpus: Optional[int] = None,
    node_gpus: Optional[int] = None,
    tokens_col: str = _TOKENS,
    tot_tokens_col: Optional[str] = _NO_TOT_TOKENS,
    setup_time_col: Optional[str] = None,
    workers: int = 1,
) -> pd.DataFrame:
    """Recommend a layout per trace row, write the recommended-trace CSV and return the DataFrame.

    goal: ``"performance"`` (default), ``"balanced"`` and ``"energy"`` rank a grid. ``"min_gpu"``
        keeps each job's total batch and gives it the first feasible GPU count in 1, 2, 4, ...
        For every goal, ``metadata.batch_size`` is the job's total batch, written back as the
        recommended per-device batch x GPUs. In a trace without a per-device batch column, the
        weighted goals start from ``metadata.batch_size`` / (GPUs per node x nodes), and a row
        whose total does not split evenly over its GPUs is kept unchanged with a note.
    feasibility: ``"autoconf"`` (default) runs the AutoConf OOM check and raises if AutoConf (the
        ``coastline[autoconf]`` extra) is missing, unless ``COASTLINE_ALLOW_RULES_FALLBACK=1``.
        ``"rules"`` only checks for a positive GPU count and a per-device batch of at least 1.
    lookup: measured-runs CSV (flat sfttrainer schema) for the ``cache`` and ``intelligent``
        methods, or ``"default"`` for the small bundled lookup DB.
    cluster_gpus, node_gpus: the GPU budget every job is optimized within (default:
        ``infrastructure.yaml``). The cluster size is not read from the trace.
    workers: processes per pipeline stage (1 runs in this process). The CLI takes it from
        ``--workers`` or ``runtime.parallel_workers``.
    tokens_col: the predictors' ``tokens_per_sample`` column (default
        ``metadata.tokens_per_sample``), for example ``metadata.estimated_max_seq_length`` for the
        real, shorter sequence length.
    tot_tokens_col: each job's precomputed, config-independent total token count (for example
        ``metadata.output.extrapolated_num_tokens``).
    setup_time_col: each job's setup time (for example ``metadata.output.setup_time``).

    The duration is ``setup_time + tot_tokens / throughput`` with both columns (the identity in
    ``add_auxiliary_information.py``), ``tot_tokens / throughput`` with only ``tot_tokens_col``,
    and the measured ``train_tokens_per_second x train_runtime`` without ``tot_tokens_col``
    (needs output data). ``metadata.estimated_throughput_<method>`` is always written.
    """
    # A wrong goal, mode or method affects every row, so it raises here.
    goal = normalize_goal(goal)
    if feasibility not in {mode.value for mode in FeasibilityMode}:
        raise ValueError(
            f"unknown feasibility mode {feasibility!r}: expected one of {[mode.value for mode in FeasibilityMode]}"
        )
    predictor = normalize_predictor(_METHOD_TO_PREDICTOR.get(method.lower(), method.lower()))
    total_gpus, gpus_per_node, _ = resolve_cluster_caps(cluster_gpus, node_gpus)
    df = pd.read_csv(input_csv, low_memory=False)
    missing = [col for col in _IDENTITY_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"{input_csv} is missing the trace column(s) {', '.join(missing)}")
    # Per-device mode when the trace has a per-device batch column: recommend on the per-device
    # batch and write per_device_train_batch_size, which VV reads.
    per_device_mode = any(c in df.columns for c in _PER_DEVICE_COLS)
    # One strategy per distinct config. In legacy mode build_config derives grid.batch_sizes from
    # each row's batch size, so a single shared strategy would be wrong.
    strategy_cache = StrategyCache()
    recs = [
        _recommend_row(
            row,
            predictor,
            goal,
            feasibility,
            total_gpus,
            lookup,
            tokens_col=tokens_col,
            tot_tokens_col=tot_tokens_col,
            setup_time_col=setup_time_col,
            per_device_mode=per_device_mode,
            strategy_cache=strategy_cache,
            workers=workers,
            gpus_per_node=gpus_per_node,
        )
        for _, row in df.iterrows()
    ]
    logger.info(
        "strategy cache: %d built, %d reused across %d rows", strategy_cache.builds, strategy_cache.hits, len(df)
    )
    for i, r in enumerate(recs):
        if r["note"]:
            logger.warning("row %d (%s): %s", i, df.iloc[i].get(_MODEL, "?"), r["note"])

    thr_col = f"metadata.estimated_throughput_{method}"
    dur_col = f"metadata.estimated_duration_{method}"

    df[_NODES] = [r["nodes"] for r in recs]
    df[_GPN] = [r["gpn"] for r in recs]
    df[_BATCH] = [r["batch"] for r in recs]
    if per_device_mode:
        # The recommended per-device batch, which VV reads. metadata.batch_size above is already
        # per_device x gpn x nodes for recommended rows.
        df[_REC_PER_DEVICE] = [r["per_device"] for r in recs]
    df[thr_col] = [r["thr"] for r in recs]
    df[dur_col] = [r["dur"] for r in recs]
    df["metadata.recommendation_note"] = [r["note"] for r in recs]

    # Recommended rows get the uid "{METHOD}:{orig_uid}" (the VV convention), which links them to
    # the source row; rows kept unchanged (r["note"] set) keep their uid.
    _uid = "metadata.uid"
    if _uid in df.columns:
        prefix = f"{method.upper()}:"
        df[_uid] = [prefix + str(u) if not r["note"] else str(u) for u, r in zip(df[_uid], recs)]

    has_duration = df[dur_col].notna().any()
    df = _tidy_columns(df, thr_col, dur_col if has_duration else None)
    df.to_csv(output_csv, index=False)
    return df
