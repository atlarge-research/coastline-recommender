"""Public batch API: ``coastline.recommend(batch, ...) -> pd.DataFrame``.

Settings can be keyword arguments (for the whole batch) or per-row columns (which override the
argument for that row). Uses the same engine as the CLI and UI.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Union

import pandas as pd

from coastline.sdk.pipeline.grid import check_top_k
from coastline.sdk.policies import normalize_predictor
from coastline.sdk.recommend import engine
from coastline.sdk.recommend._goals import goal_to_label

Batch = Union[pd.DataFrame, list, dict]

# Batch column and the ``engine`` answers key it fills. Workload columns are the WorkloadSpec
# field names, with no synonyms; the other columns configure the search. ``max_slowdown`` is
# handled separately.
_COLUMN_TO_ANSWER = {
    "llm_model": "llm_model",
    "fine_tuning_method": "fine_tuning_method",
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
_INT_COLUMNS = ("tokens_per_sample", "batch_size", "max_gpus")
# Kept as positive floats: epochs may be fractional (HF num_train_epochs is a float), and only
# the total token count is rounded (engine.run_pipeline).
_POSITIVE_NUMBER_COLUMNS = ("dataset_size", "epochs")

# Longest error text kept in a failed row; long enough for a predictor's own reason.
_MAX_ERROR_CHARS = 500

# Workload fields every batch row needs, in the row or as a keyword argument. The interactive
# and no-TTY UI fill them from engine.defaults(); a batch row without one fails instead of
# getting a recommendation for a default workload the caller did not ask for.
_REQUIRED_COLUMNS = ("llm_model", "gpu_model", "tokens_per_sample", "batch_size")

# Output columns (Kavier-style names: throughput_tok_s, runtime_s, energy_wh). ``batch_size`` is
# the recommended per-device batch on feasible rows and the input value on failed rows;
# ``recommended_batch_size`` holds the recommendation only and is empty when there is none.
_OUTPUT_COLUMNS = (
    "rank",
    "total_gpus",
    "gpus_per_node",
    "number_of_nodes",
    "batch_size",
    "recommended_batch_size",
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
    """The row's value for a column, or None."""
    return row.get(column)


def _resolve_goal(value: Any) -> str:
    """Map a goal value to its engine GOALS label."""
    if value in engine.GOALS:  # already a full engine label
        return value
    return goal_to_label(value)


def _missing_required(row: dict[str, Any], kwargs: dict[str, Any]) -> Optional[str]:
    """The first required field missing from both the row and the keyword arguments, or None.

    NaN, None and blank cells were dropped from the row, so they count as missing. A field that
    is present but invalid is left for the engine to reject.
    """
    for column in _REQUIRED_COLUMNS:
        if _pick(row, column) is None and kwargs.get(column) is None:
            return column
    return None


def _positive_number(column: str, value: Any) -> float:
    """``value`` as a finite float above zero; ValueError naming ``column`` otherwise."""
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{column} must be a positive number, got {value!r}")
    return number


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
        if column in _INT_COLUMNS:
            value = int(value)
        elif column in _POSITIVE_NUMBER_COLUMNS:
            # A zero or negative token count would make the runtime fall back to the
            # predictor's own (historical) runtime.
            value = _positive_number(column, value)
        answers[answer_key] = value
    answers["goal_label"] = _resolve_goal(answers["goal_label"])
    if answers.get("predictor") is not None:
        # An unknown predictor fails the row; the normalized key makes 'XGBoost' run xgboost.
        answers["predictor"] = normalize_predictor(answers["predictor"])
    if kwargs.get("batch_sizes"):
        # A batch grid (a list) skips the int conversion above.
        answers["batch_sizes"] = list(kwargs["batch_sizes"])

    slowdown = _pick(row, "max_slowdown")
    if slowdown is None:
        slowdown = kwargs.get("max_slowdown")
    return answers, (None if slowdown is None else float(slowdown))


def _predict(rec, total_tokens: int) -> dict[str, Any]:
    """One recommendation as batch-API columns (Kavier-style names such as throughput_tok_s)."""
    f = engine.flatten_recommendation(rec, total_tokens)
    return {
        "total_gpus": f["total_gpus"],
        "gpus_per_node": f["gpus_per_node"],
        "number_of_nodes": f["number_of_nodes"],
        "batch_size": f["batch_size"],
        "recommended_batch_size": f["batch_size"],
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
    epochs: Optional[float] = None,
    feasibility: str = "autoconf",
    lookup: Optional[str] = None,
    batch_sizes: Optional[list[int]] = None,
    strategy_cache: Optional[engine.StrategyCache] = None,
    workers: Optional[int] = None,
    max_gpus_per_node: Optional[int] = None,
) -> pd.DataFrame:
    """Recommend GPU and node configurations for a batch of workloads.

    Returns a ``pandas.DataFrame``: the input rows with the chosen configuration and its
    predictions, one row per ranked pick. Per-row columns override the keyword arguments, and a
    bad row gets ``feasible=False`` without stopping the others. ``recommended_batch_size`` is
    the recommended per-device batch, empty on a failed row.

    goal, predictor: same values as in ``Coastline.recommend``; ``goal`` is ``"balanced"``,
        ``"performance"``, ``"energy"`` or ``"min_gpu"``.
    max_slowdown: keep configs within k times the fastest (finite k >= 1).
    dataset_size, epochs: positive; ``epochs`` may be fractional.
    feasibility: OOM checker, ``autoconf``, ``rules`` or ``none``. ``rules`` only checks for a
        positive GPU count and a per-device batch of at least 1 and needs no AutoConf install.
    lookup: measured-runs CSV for the ``cache`` and ``intelligent`` predictors, or
        ``"default"`` for the small bundled lookup DB; other predictors ignore it.
    strategy_cache: reuse one strategy across calls that share a config (for example a trace
        recommended one row per call); with ``None`` each call builds its own.
    workers: processes per pipeline stage; ``None`` keeps all work in the calling process.
    max_gpus_per_node: cap on the GPUs per node of every layout (the cluster's node width,
        default 8). It applies to the whole call and is not read from rows.
    """
    # top_k applies to the whole call, so a bad value raises here. int() as in the grid, so a
    # numeric string works.
    top_k = int(top_k)
    check_top_k(top_k)
    rows = _normalise(batch)
    base = engine.defaults(engine.resolve_options())
    if max_gpus_per_node is not None:
        if int(max_gpus_per_node) < 1:
            raise ValueError(f"max_gpus_per_node must be at least 1, got {max_gpus_per_node!r}")
        base["max_gpus_per_node"] = int(max_gpus_per_node)
    kwargs = {
        "goal": goal,
        "predictor": predictor,
        "max_gpus": max_gpus,
        "max_slowdown": max_slowdown,
        "dataset_size": dataset_size,
        "epochs": epochs,
        "lookup": lookup,
        "batch_sizes": batch_sizes,
    }

    out_rows: list[dict[str, Any]] = []
    for row in rows:
        # A row missing a required field fails here, before the engine defaults would fill it.
        absent = _missing_required(row, kwargs)
        if absent is not None:
            out_rows.append(_failed_row(row, f"missing required field: {absent}"))
            continue
        # A bad workload (unknown GPU or model, invalid value) gives a failed row with the
        # reason; the other rows still run.
        try:
            answers, slowdown = _answers_for(row, kwargs, base)
            recs, meta = engine.run_pipeline(
                answers,
                top_k=top_k,
                max_slowdown=slowdown,
                feasibility=feasibility,
                strategy_cache=strategy_cache,
                workers=workers,
            )
        except Exception as exc:  # noqa: BLE001 (any error fails only this row)
            out_rows.append(_failed_row(row, str(exc)[:_MAX_ERROR_CHARS] or type(exc).__name__))
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

    if not out_rows:  # empty batch: an empty frame with the output columns
        return pd.DataFrame(columns=list(_OUTPUT_COLUMNS))
    return pd.DataFrame(out_rows)
