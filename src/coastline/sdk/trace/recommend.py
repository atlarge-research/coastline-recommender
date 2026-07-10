"""Recommend a config for every job in a fine-tuning trace.

Replaces the GPU layout columns and appends ``metadata.estimated_duration_<method>``
(= job_total_tokens / recommended_throughput). With ``--visual`` it also renders the
operational cluster timeline (the recommended configs FIFO-scheduled — GPUs in use +
jobs queued over time) to a PDF beside the output CSV.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

import coastline

logger = logging.getLogger(__name__)

# --- trace schema -> coastline workload fields ---
_MODEL = "metadata.model_name"
_METHOD = "metadata.method"
_GPU = "resources.gpu_model"
_TOKENS = "metadata.tokens_per_sample"  # the int the user wants; nominal seq length is fine here
_BATCH = "metadata.batch_size"
_GPN = "resources.num_gpus_per_node"
_NODES = "resources.num_nodes"
# ground-truth work (config-independent): tokens the job actually processed
_ACT_TPS = "metadata.output.train_tokens_per_second"
_ACT_RUNTIME = "metadata.train_runtime"
# observed job duration — the fallback when no recommendation can be made
_ACT_DURATION = "metadata.output.extrapolated_duration"

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
]


def _tidy_columns(df: pd.DataFrame, duration_col: str) -> pd.DataFrame:
    """Put the decision columns first and drop raw fine-tuning args (adam_beta1, ...).

    Only dotted ``metadata.*`` / ``resources.*`` columns are kept after the front
    block — the flat launcher-arg columns carry no signal for trace analysis.
    """
    front = [c for c in [*_FRONT_COLS, duration_col, "metadata.recommendation_note"] if c in df.columns]
    rest = [c for c in df.columns if c not in front and "." in c]
    return df[front + rest]


def _as_int(value: Any) -> Optional[int]:
    n = pd.to_numeric(value, errors="coerce")
    return int(n) if pd.notna(n) and n >= 1 else None


def _job_total_tokens(row: pd.Series) -> Optional[float]:
    """Tokens the job processed = throughput x runtime (config-independent 'work')."""
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


def _infeasible_note(gpn: int, nodes: int, feasibility: str) -> str:
    cause = (
        "every config would run out of GPU memory (autoconf OOM check)"
        if feasibility == "autoconf"
        else "no config passes the divisibility rules"
    )
    return f"infeasible within {gpn * nodes} GPUs: {cause}"


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
    keep["dur"] = _observed_duration(row)
    tail = (
        "job kept unchanged (original config + observed duration)"
        if keep["dur"] is not None
        else "job kept with the original config but NO observed duration — it will be missing from the timeline"
    )
    keep["note"] = f"{reason} — {tail}"
    return keep


def _recommend_row(
    row: pd.Series, predictor: str, goal: str, feasibility: str, lookup: Optional[str] = None
) -> dict[str, Any]:
    """Recommend a layout for one trace row; fall back to the original layout on any failure.

    The returned dict carries a ``note`` (None on full success) saying why a row kept
    its original layout or got no estimated duration — surfaced as a per-row warning
    and in the ``metadata.recommendation_note`` output column. When the chosen
    predictor is the problem, the note says whether kavier could handle the workload.
    """
    keep = {"nodes": row.get(_NODES), "gpn": row.get(_GPN), "batch": row.get(_BATCH), "dur": None, "note": None}
    tokens, batch = _as_int(row.get(_TOKENS)), _as_int(row.get(_BATCH))
    gpn, nodes = _as_int(row.get(_GPN)), _as_int(row.get(_NODES))
    if not (tokens and batch and gpn and nodes):
        return _unchanged(keep, row, "no recommendation: missing/invalid workload fields")
    wl = {
        "llm_model": str(row[_MODEL]),
        "fine_tuning_method": str(row[_METHOD]),
        "gpu_model": str(row[_GPU]),
        "tokens_per_sample": tokens,
        "batch_size": batch,
    }

    def kavier_hint() -> str:
        if predictor == "kavier":
            return ""
        if _kavier_can_predict(wl, goal, feasibility, gpn * nodes):
            return " — kavier CAN handle this workload: rerun with --method kavier"
        return ""

    try:
        out = coastline.recommend(
            [wl], predictor=predictor, goal=goal, max_gpus=gpn * nodes, top_k=1, feasibility=feasibility, lookup=lookup
        )
        if out.empty or not bool(out.iloc[0]["feasible"]):
            # feasible=False covers two very different causes: the feasibility check
            # rejected every config, or the predictor had no answer. The kavier probe
            # (same feasibility checker) tells them apart.
            hint = kavier_hint()
            reason = (
                f"'{predictor}' could not predict this workload{hint}"
                if hint
                else _infeasible_note(gpn, nodes, feasibility)
            )
            return _unchanged(keep, row, reason)
        top = out.iloc[0]
        total_tokens, thr = _job_total_tokens(row), top["throughput_tok_s"]
        if not (pd.notna(thr) and thr > 0):
            return _unchanged(keep, row, f"'{predictor}' returned no throughput for this workload{kavier_hint()}")
        if not total_tokens:
            return _unchanged(
                keep, row, "recommended config discarded: no observed throughput/runtime to derive the job's work"
            )
        return {
            "nodes": int(top["number_of_nodes"]),
            "gpn": int(top["gpus_per_node"]),
            "batch": _as_int(top["batch_size"]) or batch,
            "dur": total_tokens / thr,
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
) -> pd.DataFrame:
    """Recommend a layout per trace row, write the recommended-trace CSV, and return the DataFrame.

    ``feasibility="autoconf"`` (default) runs the real AutoConf OOM check — it
    fail-closes if AutoConf (the ``coastline[autoconf]`` extra) is
    not installed (use ``COASTLINE_ALLOW_RULES_FALLBACK=1`` to suppress).  Pass
    ``feasibility="rules"`` to use the divisibility-only path (no OOM check,
    works on any install).

    ``lookup`` points the ``cache``/``intelligent`` methods at a measured-runs CSV
    (flat sfttrainer schema), or ``"default"`` for the small bundled lookup DB.
    """
    predictor = _METHOD_TO_PREDICTOR.get(method.lower(), method.lower())
    df = pd.read_csv(input_csv, low_memory=False)
    recs = [_recommend_row(row, predictor, goal, feasibility, lookup) for _, row in df.iterrows()]
    for i, r in enumerate(recs):
        if r["note"]:
            logger.warning("row %d (%s): %s", i, df.iloc[i].get(_MODEL, "?"), r["note"])
    df[_NODES] = [r["nodes"] for r in recs]
    df[_GPN] = [r["gpn"] for r in recs]
    df[_BATCH] = [r["batch"] for r in recs]
    df[f"metadata.estimated_duration_{method}"] = [r["dur"] for r in recs]
    df["metadata.recommendation_note"] = [r["note"] for r in recs]
    df = _tidy_columns(df, f"metadata.estimated_duration_{method}")
    df.to_csv(output_csv, index=False)
    return df
