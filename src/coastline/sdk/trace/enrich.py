"""Enrich a fine-tuning trace with coastline recommendations.

Replaces the GPU layout columns and appends ``metadata.estimated_duration_<method>``
(= job_total_tokens / recommended_throughput). With ``--visual`` it also renders the
operational cluster timeline (the recommended configs FIFO-scheduled — GPUs in use +
jobs queued over time) to a PDF beside the enriched CSV.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

import coastline

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

# coastline predictor keys for the trace's method names
_METHOD_TO_PREDICTOR = {"kavier": "kavier", "tabpfn": "tabpfn", "xgb": "xgboost", "xgboost": "xgboost"}


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


def _recommend_row(row: pd.Series, predictor: str, goal: str, feasibility: str) -> dict[str, Any]:
    """Recommend a layout for one trace row; fall back to the original layout on any failure."""
    keep = {"nodes": row.get(_NODES), "gpn": row.get(_GPN), "batch": row.get(_BATCH), "dur": None}
    tokens, batch = _as_int(row.get(_TOKENS)), _as_int(row.get(_BATCH))
    gpn, nodes = _as_int(row.get(_GPN)), _as_int(row.get(_NODES))
    if not (tokens and batch and gpn and nodes):
        return keep
    try:
        wl = {
            "llm_model": str(row[_MODEL]),
            "fine_tuning_method": str(row[_METHOD]),
            "gpu_model": str(row[_GPU]),
            "tokens_per_sample": tokens,
            "batch_size": batch,
        }
        out = coastline.recommend(
            [wl], predictor=predictor, goal=goal, max_gpus=gpn * nodes, top_k=1, feasibility=feasibility
        )
        if out.empty or not bool(out.iloc[0]["feasible"]):
            return keep
        top = out.iloc[0]
        total_tokens, thr = _job_total_tokens(row), top["throughput_tok_s"]
        est = (total_tokens / thr) if (total_tokens and thr and thr > 0) else None
        return {
            "nodes": int(top["number_of_nodes"]),
            "gpn": int(top["gpus_per_node"]),
            "batch": _as_int(top["batch_size"]) or batch,
            "dur": est,
        }
    except Exception:  # one bad row must not sink the whole trace
        return keep


def enrich_trace(
    input_csv: str, output_csv: str, *, method: str = "kavier", goal: str = "min_gpu", feasibility: str = "autoconf"
) -> pd.DataFrame:
    """Recommend a layout per trace row, write the enriched CSV, and return the DataFrame.

    ``feasibility="autoconf"`` (default) runs the real AutoConf OOM check — it
    fail-closes if AutoConf (the ``coastline[autoconf]`` extra) is
    not installed (use ``COASTLINE_ALLOW_RULES_FALLBACK=1`` to suppress).  Pass
    ``feasibility="rules"`` to use the divisibility-only path (no OOM check,
    works on any install).
    """
    predictor = _METHOD_TO_PREDICTOR.get(method.lower(), method.lower())
    df = pd.read_csv(input_csv, low_memory=False)
    recs = [_recommend_row(row, predictor, goal, feasibility) for _, row in df.iterrows()]
    df[_NODES] = [r["nodes"] for r in recs]
    df[_GPN] = [r["gpn"] for r in recs]
    df[_BATCH] = [r["batch"] for r in recs]
    df[f"metadata.estimated_duration_{method}"] = [r["dur"] for r in recs]
    df.to_csv(output_csv, index=False)
    return df
