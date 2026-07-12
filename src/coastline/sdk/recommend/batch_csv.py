"""Batch CSV -> CSV recommender: config + input workloads CSV -> one recommended
configuration per row. The strategy is built once and reused across rows so the
predictors and the AutoConf feasibility model load a single time.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterator

import yaml

from coastline.sdk.exceptions import RecommenderSystemError
from coastline.sdk.io.infrastructure import resolve_cluster_caps
from coastline.sdk.models.aliases import col_to_field_map
from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.policies import PolicyFactory
from coastline.sdk.recommend.engine import recommendation_rationale

_INT_FIELDS = ("tokens_per_sample", "batch_size", "gpus_per_node", "number_of_nodes")

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
)


def recommend_csv(config_path, input_csv, output_csv, *, cluster_gpus=None) -> None:
    """Recommend the best configuration for every workload row in ``input_csv``.

    The GPU search is bounded by the cluster: ``cluster_gpus`` (a ``--cluster-gpus`` flag) or, when
    unset, ``infrastructure.yaml``'s declared total. No row is ever recommended more GPUs than the
    cluster has; the config ``grid.total_gpus`` still applies but is capped to the cluster.
    """
    config = _load_config(config_path)
    strategy = _build_strategy(config)
    column_map = _column_map(config)
    max_gpus, gpus_per_node, max_nodes = resolve_cluster_caps(cluster_gpus)
    # Goal context for the one-line rationale (same phrasing as the API/UI/JSON).
    meta = {"preset": config["strategy"].get("preset"), "strategy_name": config["strategy"].get("name")}

    results = []
    for original, fields in _read_workloads(input_csv, column_map):
        try:
            # Build the WorkloadSpec INSIDE the per-row try: a row with a blank/invalid
            # required field raises pydantic ValidationError (a ValueError). Isolating it
            # here marks just that row feasible=False (like coastline.sdk.recommend.batch_api.recommend) instead
            # of aborting the whole job mid-stream.
            workload = WorkloadSpec(**fields)
            context = SystemContext.for_gpus(
                [workload.gpu_model], max_gpus=max_gpus, gpus_per_node=gpus_per_node, max_nodes=max_nodes
            )
            recs = strategy.recommend(workload, context)
        except (RuntimeError, ValueError, RecommenderSystemError):
            recs = []  # invalid/incomplete row, or no feasible configuration in the grid
        results.append((original, recs, meta))

    _write_output(output_csv, results)


def _load_config(path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    for section in ("strategy", "predictors", "grid"):
        config.setdefault(section, {})
    # Safeguard "performance must not get more than X times worse than the fastest
    # feasible config" maps to the engine's runtime_guard_k.
    if "max_slowdown" in config["strategy"]:
        config["strategy"]["runtime_guard_k"] = float(config["strategy"]["max_slowdown"])
    return config


def _build_strategy(config: dict[str, Any]):
    strategy = config["strategy"]
    return PolicyFactory.create_strategy(
        strategy_name=strategy.get("name", "multi_objective"),
        preset=strategy.get("preset"),
        config=config,
    )


def _column_map(config: dict[str, Any]) -> dict[str, str]:
    """CSV-column -> WorkloadSpec-field, from the canonical aliases plus any
    ``input.columns`` override in the config."""
    mapping = col_to_field_map()
    mapping.update((config.get("input") or {}).get("columns", {}))
    return mapping


def _read_workloads(input_csv, column_map) -> Iterator[tuple[dict, dict[str, Any]]]:
    """Yield ``(raw_row, workload_fields)`` per CSV row. The WorkloadSpec is built by the
    caller inside its per-row try so a row with an invalid/blank required field is isolated
    (feasible=False) instead of raising mid-iteration and aborting the whole job."""
    with open(input_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fields: dict[str, Any] = {}
            for col, value in row.items():
                field = column_map.get(col)
                if field and value not in (None, ""):
                    if field in _INT_FIELDS:
                        try:
                            fields[field] = int(float(value))
                        except (TypeError, ValueError):
                            # non-numeric: leave raw so WorkloadSpec rejects this row (the caller
                            # isolates it) rather than crashing the generator.
                            fields[field] = value
                    else:
                        fields[field] = value
            yield row, fields


def _output_row(original: dict, recs, meta) -> dict:
    row = dict(original)
    if not recs:
        row.update({field: "" for field in _OUTPUT_FIELDS})
        row["feasible"] = False
        return row
    rec = recs[0]
    runtime = rec.predicted_runtime_seconds
    row.update(
        recommended_total_gpus=rec.total_gpus,
        recommended_gpus_per_node=rec.gpus_per_node,
        recommended_number_of_nodes=rec.number_of_nodes,
        recommended_batch_size=rec.metadata.get("batch_size", ""),
        predicted_throughput=rec.predicted_throughput,
        predicted_runtime_seconds="" if runtime is None else runtime,
        predicted_power_watts=rec.metadata.get("predicted_power_watts", ""),
        tokens_per_watt=rec.metadata.get("tokens_per_watt", ""),
        feasible=True,
        rationale=recommendation_rationale(recs, meta),
    )
    return row


def _write_output(output_csv, results) -> None:
    if not results:
        return
    base = list(results[0][0].keys())
    fieldnames = base + [f for f in _OUTPUT_FIELDS if f not in base]
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_output_row(original, recs, meta) for original, recs, meta in results)
