"""Public batch API: ``coastline.recommend(batch, ...) -> pd.DataFrame``.

Knobs work both as kwargs (batch default) and as per-row columns (which override the
kwarg for that row). Runs through the same engine as the CLI and UI.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import pandas as pd

from coastline.sdk.models.aliases import WORKLOAD_FIELD_ALIASES
from coastline.sdk.policies import normalize_predictor
from coastline.sdk.recommend import engine
from coastline.sdk.recommend._goals import goal_to_label

Batch = Union[pd.DataFrame, list, dict]

# Public batch column -> the accepted spellings (a row may use any). The workload columns
# source their spellings from the ONE shared vocabulary (models/aliases); the engine-knob
# columns are batch-API-specific (they configure the search, not the job). The ``answers``
# key each column fills is in ``_COLUMN_TO_ANSWER``.
_ALIASES: dict[str, tuple[str, ...]] = {
    "model": WORKLOAD_FIELD_ALIASES["llm_model"],
    "method": WORKLOAD_FIELD_ALIASES["fine_tuning_method"],
    "gpu_model": WORKLOAD_FIELD_ALIASES["gpu_model"],
    "tokens_per_sample": WORKLOAD_FIELD_ALIASES["tokens_per_sample"],
    "batch_size": WORKLOAD_FIELD_ALIASES["batch_size"],
    "dataset_size": ("dataset_size", "num_samples"),
    "epochs": ("epochs",),
    "max_gpus": ("max_gpus", "gpu_budget"),
    "goal": ("goal", "goal_label"),
    "predictor": ("predictor",),
    "lookup": ("lookup", "lookup_csv"),
    "max_slowdown": ("max_slowdown", "runtime_guard_k"),
}
# Public column -> the ``engine`` answers key it fills (the engine's own schema).
_COLUMN_TO_ANSWER = {
    "model": "llm_model",
    "method": "fine_tuning_method",
    "gpu_model": "gpu_model",
    "tokens_per_sample": "tokens_per_sample",
    "batch_size": "batch_size",
    "dataset_size": "dataset_size",
    "epochs": "epochs",
    "max_gpus": "max_gpus",
    "goal": "goal_label",
    "predictor": "predictor",
    "lookup": "lookup",
}
_INT_COLUMNS = ("tokens_per_sample", "batch_size", "dataset_size", "epochs", "max_gpus")

# Core workload fields a batch/CSV/API caller MUST supply. Unlike the interactive /
# no-TTY UI (where engine.defaults() legitimately fills these), a batch row that omits
# one must NOT silently inherit the default (mistral-7b / A100 / 1024 / 32) — that would
# return a confident feasible=True for a workload the caller never gave. We require them
# present (in the row or as a batch kwarg) and emit a failed row otherwise.
_REQUIRED_COLUMNS = ("model", "gpu_model", "tokens_per_sample", "batch_size")

# The batch output columns (kavier-style names: throughput_tok_s / runtime_s / energy_wh).
_OUTPUT_COLUMNS = (
    "rank",
    "total_gpus",
    "gpus_per_node",
    "number_of_nodes",
    "batch_size",
    "throughput_tok_s",
    "runtime_s",
    "energy_wh",
    "energy_kwh",
    "tokens_per_watt",
    "power_w",
    "feasible",
    "error",
    "rationale",
)

def _drop_missing(row: dict[str, Any]) -> dict[str, Any]:
    """Drop NaN/None/blank cells so ``.get(key)`` means 'absent'."""
    out: dict[str, Any] = {}
    for k, v in row.items():
        if v is None:
            continue
        if isinstance(v, float) and pd.isna(v):
            continue
        if isinstance(v, str) and not v.strip():
            continue
        out[k] = v
    return out


def _normalise(batch: Batch) -> list[dict[str, Any]]:
    """Coerce batch (DataFrame | list[dict] | dict) into plain row dicts with NaN dropped."""
    if isinstance(batch, pd.DataFrame):
        records = [{str(k): v for k, v in rec.items()} for rec in batch.to_dict(orient="records")]
    elif isinstance(batch, dict):
        records = [dict(batch)]
    elif isinstance(batch, (list, tuple)):
        if not all(isinstance(row, dict) for row in batch):
            raise TypeError("each row of a list batch must be a dict (one workload per row)")
        records = [dict(row) for row in batch]
    else:
        raise TypeError(
            f"batch must be a pandas DataFrame, a list of dicts, or a single dict; got {type(batch).__name__}"
        )
    return [_drop_missing(r) for r in records]


def _pick(row: dict[str, Any], column: str) -> Any:
    """First present alias value for a public column, else None."""
    for alias in _ALIASES[column]:
        if alias in row:
            return row[alias]
    return None


def _resolve_goal(value: Any) -> str:
    """Map a goal column/kwarg to an engine GOALS label, via the shared goal vocabulary."""
    if value in engine.GOALS:  # already a full engine label
        return value
    return goal_to_label(value)


def _missing_required(row: dict[str, Any], kwargs: dict[str, Any]) -> Optional[str]:
    """First core field absent from both the row (any alias) and the batch kwargs, else None.

    Runs on the missing-dropped row, so a NaN/None/blank cell counts as absent. This is
    what stops a batch/CSV/API row from silently inheriting an engine default for a field
    the caller never gave (the present-but-invalid case is left to the engine to reject).
    """
    for column in _REQUIRED_COLUMNS:
        if _pick(row, column) is None and kwargs.get(column) is None:
            return column
    return None


def _answers_for(
    row: dict[str, Any], kwargs: dict[str, Any], base: dict[str, Any]
) -> tuple[dict[str, Any], Optional[float]]:
    """Build one engine ``answers`` dict: per-row column > kwarg > engine default."""
    answers = dict(base)
    for column, answer_key in _COLUMN_TO_ANSWER.items():
        value = _pick(row, column)
        if value is None:
            value = kwargs.get(column)
        if value is None:
            continue
        answers[answer_key] = int(value) if column in _INT_COLUMNS else value
    answers["goal_label"] = _resolve_goal(answers["goal_label"])
    if answers.get("predictor") is not None:
        # Fail a typo'd predictor visibly per row rather than silently defaulting in the engine.
        normalize_predictor(answers["predictor"])

    slowdown = _pick(row, "max_slowdown")
    if slowdown is None:
        slowdown = kwargs.get("max_slowdown")
    return answers, (None if slowdown is None else float(slowdown))


def _predict(rec, total_tokens: int) -> dict[str, Any]:
    """The batch-API column names over the shared flattener (kavier-style throughput_tok_s)."""
    f = engine.flatten_recommendation(rec, total_tokens)
    return {
        "total_gpus": f["total_gpus"],
        "gpus_per_node": f["gpus_per_node"],
        "number_of_nodes": f["number_of_nodes"],
        "batch_size": f["batch_size"],
        "throughput_tok_s": f["throughput"],
        "runtime_s": f["runtime_s"],
        "energy_wh": f["energy_wh"],
        "energy_kwh": f["energy_kwh"],
        "tokens_per_watt": f["tokens_per_watt"],
        "power_w": f["power_w"],
    }


def _failed_row(row: dict[str, Any], error: Optional[str]) -> dict[str, Any]:
    """Output row for a workload with no feasible config; reason in ``error``."""
    out = {**row, "rank": 1, "feasible": False, "error": error}
    for col in _OUTPUT_COLUMNS:
        out.setdefault(col, None)
    return out


def recommend(
    batch: Batch,
    *,
    top_k: int = 1,
    goal: str = "balanced",
    predictor: str = "kavier",
    max_gpus: Optional[int] = None,
    max_slowdown: Optional[float] = None,
    dataset_size: Optional[int] = None,
    epochs: Optional[int] = None,
    feasibility: str = "autoconf",
    lookup: Optional[str] = None,
) -> pd.DataFrame:
    """Recommend GPU/node configurations for a batch — returns a ``pandas.DataFrame`` of the input
    rows plus the chosen config + predictions (one row per ranked pick).

    ``goal`` (``"balanced"`` | ``"performance"`` | ``"energy"`` | ``"min_gpu"``) and ``predictor``
    use the same vocabulary as ``Coastline.recommend``. Per-row columns override kwargs. One bad row
    yields ``feasible=False`` without failing the rest. ``max_slowdown`` keeps only configs within k×
    of the fastest. ``feasibility`` picks the OOM checker (``autoconf`` | ``rules`` | ``none``); use
    ``rules`` for the divisibility-only path that needs no AutoConf install.
    ``lookup`` points the ``cache``/``intelligent`` predictors at a measured-runs CSV
    (or ``"default"`` for the small bundled lookup DB); other predictors ignore it.
    """
    rows = _normalise(batch)
    base = engine.defaults(engine.resolve_options())
    kwargs = {
        "goal": goal,
        "predictor": predictor,
        "max_gpus": max_gpus,
        "max_slowdown": max_slowdown,
        "dataset_size": dataset_size,
        "epochs": epochs,
        "lookup": lookup,
    }

    out_rows: list[dict[str, Any]] = []
    for row in rows:
        # A row that OMITS a core field must NOT inherit the engine default (which would
        # return a confident recommendation for a workload the caller never gave). Reject
        # it as a failed row before defaults paper over the gap. (The interactive/no-TTY UI
        # default-fill is a separate path and intentionally keeps the defaults.)
        absent = _missing_required(row, kwargs)
        if absent is not None:
            out_rows.append(_failed_row(row, f"missing required field: {absent}"))
            continue
        # Per-row isolation: a bad workload (unknown GPU/model, invalid value) yields a
        # failed row with the reason, never crashing the rest of the batch.
        try:
            answers, slowdown = _answers_for(row, kwargs, base)
            recs, meta = engine.run_pipeline(answers, top_k=top_k, max_slowdown=slowdown, feasibility=feasibility)
        except Exception as exc:  # noqa: BLE001 — isolate any per-row error
            out_rows.append(_failed_row(row, str(exc)[:200] or type(exc).__name__))
            continue
        if not recs:
            out_rows.append(_failed_row(row, "no feasible configuration in the search space"))
            continue
        total_tokens = meta["total_tokens"]
        rationale = engine.recommendation_rationale(recs, meta)
        for rank, rec in enumerate(recs, start=1):
            out_rows.append(
                {
                    **row,
                    "rank": rank,
                    "feasible": True,
                    "error": None,
                    "rationale": rationale if rank == 1 else None,
                    **_predict(rec, total_tokens),
                }
            )

    if not out_rows:  # empty batch -> empty frame, but with a stable column schema
        return pd.DataFrame(columns=list(_OUTPUT_COLUMNS))
    return pd.DataFrame(out_rows)
