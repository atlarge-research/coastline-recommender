"""CSV to CSV recommender: a config and a CSV of workloads in, one recommended configuration per
row out. The strategy is built once for all rows, so the predictors and the AutoConf model load once.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pydantic
import yaml

from coastline.sdk.exceptions import RecommenderSystemError
from coastline.sdk.io.infrastructure import resolve_cluster_caps
from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.policies import normalize_predictor
from coastline.sdk.recommend import engine

_INT_FIELDS = ("tokens_per_sample", "batch_size", "gpus_per_node", "number_of_nodes")

# Every row needs these, so a header with no column for one of them is rejected up front.
_REQUIRED_FIELDS = tuple(name for name, field in WorkloadSpec.model_fields.items() if field.is_required())

_OUTPUT_FIELDS = (
    "recommended_total_gpus",
    "recommended_gpus_per_node",
    "recommended_number_of_nodes",
    "recommended_batch_size",
    "predicted_throughput",
    "predicted_runtime_seconds",
    "predicted_power_watts",
    "tokens_per_watt",
    "feasible",
    "rationale",
    "error",  # why a row has no recommendation; blank on feasible rows
)

_NO_FEASIBLE_CONFIG = "no feasible configuration in the search space"


def recommend_csv(config_path, input_csv, output_csv, *, cluster_gpus=None) -> None:
    """Recommend the best configuration for every workload row in ``input_csv``.

    The GPU search is bounded by ``cluster_gpus`` (the ``--cluster-gpus`` flag) or, when unset,
    the total in ``infrastructure.yaml``. The config's ``grid.total_gpus`` still applies, capped
    to the cluster. ``min_gpu`` uses no grid: it keeps each row's total batch, ``batch_size`` x
    ``gpus_per_node`` x ``number_of_nodes`` (1 GPU when the row gives no layout), and tries 1, 2,
    4, ... GPUs with that total split over them.

    A row that cannot be used, or has no feasible configuration, gets ``feasible=False`` and the
    reason in ``error``; the other rows still run. An unknown ``predictors.performance`` name, or a
    header with no column for a required workload field, raises ValueError before any row runs.
    The output file is written after all rows are done, so an error leaves the old file in place,
    and an input with no data rows gives a file with only the header.
    """
    config = _load_config(config_path)
    name = config["strategy"].get("name") or "multi_objective"
    preset = config["strategy"].get("preset")
    header, workloads = _read_workloads(input_csv, _column_map(config))
    # One strategy for all rows, so the predictors and AutoConf load once.
    strategy = engine.build_strategy(config, name, preset)
    max_gpus, gpus_per_node, max_nodes = resolve_cluster_caps(cluster_gpus)
    predictor = config["predictors"].get("performance")

    rows = []
    for original, fields, problem in workloads:
        recs, meta, error = [], {}, problem
        if problem is None:
            try:
                # Built inside the try: a blank or invalid required field raises pydantic's
                # ValidationError (a ValueError), which fails only this row, as in batch_api.
                workload = WorkloadSpec(**fields)
                context = SystemContext.for_gpus(
                    [workload.gpu_model], max_gpus=max_gpus, gpus_per_node=gpus_per_node, max_nodes=max_nodes
                )
                recs, meta = engine.execute_strategy(
                    strategy,
                    workload,
                    context,
                    strategy_name=name,
                    preset=preset,
                    grid=config["grid"],
                    predictor=predictor,
                )
            except (RuntimeError, ValueError, RecommenderSystemError) as exc:
                error = _row_error(exc)  # invalid/incomplete row, or no feasible config in the grid
        rows.append(_output_row(original, recs, meta, error))

    _write_output(output_csv, header, rows)


def _load_config(path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    for section in ("strategy", "predictors", "grid"):
        # A section whose keys are all commented out loads as None; it gets the defaults.
        config[section] = config.get(section) or {}
    # max_slowdown (keep configs at most X times slower than the fastest feasible one) is the
    # engine's runtime_guard_k. A blank value sets no cap. min_gpu ignores it.
    if config["strategy"].get("max_slowdown") is not None:
        config["strategy"]["runtime_guard_k"] = float(config["strategy"]["max_slowdown"])
    # An unknown predictor name fails here, before any row runs; any letter case resolves.
    # A missing or blank name keeps the engine's default, as in PolicyFactory.
    if config["predictors"].get("performance"):
        config["predictors"]["performance"] = normalize_predictor(config["predictors"]["performance"])
    return config


def _column_map(config: dict[str, Any]) -> dict[str, str]:
    """Map CSV columns to WorkloadSpec fields. Columns are the field names; ``input.columns`` in
    the config maps other column names."""
    mapping = {field: field for field in WorkloadSpec.model_fields}
    mapping.update((config.get("input") or {}).get("columns") or {})
    return mapping


def _check_header(header: list[str], column_map: dict[str, str], input_csv) -> None:
    """Raise ValueError when the header has no column for a required workload field."""
    if not header:
        raise ValueError(f"input CSV {input_csv} has no header row")
    covered = {column_map[column] for column in header if column in column_map}
    missing = [field for field in _REQUIRED_FIELDS if field not in covered]
    if missing:
        raise ValueError(
            f"input CSV {input_csv} has no column for {', '.join(missing)} (header: {', '.join(header)}). "
            "Name the columns after these fields, or map your column names to them under input.columns "
            "in the config."
        )


def _read_workloads(input_csv, column_map) -> tuple[list[str], list[tuple[dict, dict[str, Any], str | None]]]:
    """The header and ``(raw_row, workload_fields, problem)`` per CSV row, where ``problem`` says
    why the row cannot run, or is None. The caller builds the WorkloadSpec per row, so an invalid
    field fails only that row. A UTF-8 byte order mark (written by spreadsheet exports) is skipped."""
    with open(input_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # A repeated column name is one key in each row dict, so it is one output column.
        header = list(dict.fromkeys(reader.fieldnames or []))
        _check_header(header, column_map, input_csv)
        workloads = []
        for row in reader:
            # csv keeps cells past the header under the key None. Empty ones (trailing commas)
            # are dropped; a row with data there is misaligned, so it is not run.
            extra = [cell for cell in row.pop(None, None) or [] if cell.strip()]
            problem = None
            if extra:
                problem = "row has more cells than the header: " + ", ".join(repr(cell) for cell in extra)
            fields: dict[str, Any] = {}
            for col, value in row.items():
                field = column_map.get(col)
                if field and value not in (None, ""):
                    if field in _INT_FIELDS:
                        try:
                            fields[field] = int(float(value))
                        except (TypeError, ValueError):
                            # Not a number: keep the raw value so WorkloadSpec fails this row only.
                            fields[field] = value
                    else:
                        fields[field] = value
            workloads.append((row, fields, problem))
    return header, workloads


def _row_error(exc: Exception) -> str:
    """One line saying why a row has no recommendation, for the ``error`` column."""
    if isinstance(exc, pydantic.ValidationError):
        problems = []
        for err in exc.errors():
            field = ".".join(str(part) for part in err["loc"])
            if err["type"] == "missing":
                problems.append(f"missing required field: {field}")
            else:
                problems.append(f"{field}: {err['msg']}")
        return "; ".join(problems)
    return " ".join(str(exc).split()) or type(exc).__name__


def _blank(value):
    """CSV blank for a missing prediction: None -> "" (an empty cell), else the value."""
    return "" if value is None else value


def _output_row(original: dict, recs, meta, error: str | None = None) -> dict:
    row = dict(original)
    if not recs:
        row.update({field: "" for field in _OUTPUT_FIELDS})
        row["feasible"] = False
        row["error"] = error or _NO_FEASIBLE_CONFIG
        return row
    # recommended_* and predicted_* columns from the shared flattener; None becomes a blank cell.
    # total_tokens=0, so runtime_s is the predictor's predicted_runtime_seconds (no dataset size here).
    f = engine.flatten_recommendation(recs[0])
    row.update(
        recommended_total_gpus=f["total_gpus"],
        recommended_gpus_per_node=f["gpus_per_node"],
        recommended_number_of_nodes=f["number_of_nodes"],
        recommended_batch_size=_blank(f["batch_size"]),
        predicted_throughput=f["throughput"],
        predicted_runtime_seconds=_blank(f["runtime_s"]),
        predicted_power_watts=_blank(f["power_w"]),
        tokens_per_watt=_blank(f["tokens_per_watt"]),
        feasible=True,
        rationale=engine.recommendation_rationale(recs, meta),
        error="",
    )
    return row


def _write_output(output_csv, header: list[str], rows: list[dict]) -> None:
    """Write the input columns plus the output columns, a header even when there are no rows."""
    fieldnames = header + [f for f in _OUTPUT_FIELDS if f not in header]
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
