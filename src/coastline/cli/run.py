"""Config-driven recommender CLI entry point (``python -m coastline.cli.run``)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import yaml

from coastline.cli._shared import FriendlyParser, positive_int, report_errors
from coastline.sdk.io.infrastructure import resolve_cluster_caps
from coastline.sdk.io.interface.json_output import recommendation_payload, save_recommendation_to_json
from coastline.sdk.io.run_config import default_experiment_path, load_strategy_config
from coastline.sdk.logging import setup_logging
from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.recommend import engine

UTC = timezone.utc

logger = logging.getLogger(__name__)

_JSON_JOB_SHAPE = '{"workload": {...}, "context": {...}}'

_DEFAULT_WORKLOAD = {
    "llm_model": "llama3.1-70b",
    "fine_tuning_method": "lora",
    "tokens_per_sample": 1024,
    "batch_size": 32,
    "gpus_per_node": 8,
    "number_of_nodes": 1,
}


def _workload_and_context(
    config_path: Path, raw: dict, cluster_gpus: int | None = None
) -> tuple[WorkloadSpec, SystemContext]:
    workload_cfg = raw.get("workload") or {}
    system_cfg = raw.get("system") or {}
    grid_cfg = raw.get("grid") or {}
    grid_gpus = list(grid_cfg.get("gpu_models") or [])

    # The declared job runs on its own GPU; the grid's first GPU, then system.default_gpu, then
    # the A100 are only defaults for a workload that names none.
    gpu_model = (
        workload_cfg.get("gpu_model")
        or (grid_gpus[0] if grid_gpus else None)
        or system_cfg.get("default_gpu")
        or "NVIDIA-A100-SXM4-80GB"
    )

    wl = {**_DEFAULT_WORKLOAD, **{k: v for k, v in workload_cfg.items() if k in WorkloadSpec.model_fields}}
    workload = WorkloadSpec(
        llm_model=wl.get("llm_model", _DEFAULT_WORKLOAD["llm_model"]),
        fine_tuning_method=wl.get("fine_tuning_method", _DEFAULT_WORKLOAD["fine_tuning_method"]),
        gpu_model=gpu_model,
        # "or" instead of a dict.get default, so a YAML null or 0 falls back to the default
        # instead of failing in int(None), e.g. config/coastline_functionality/experiment.yaml.
        tokens_per_sample=int(wl.get("tokens_per_sample") or _DEFAULT_WORKLOAD["tokens_per_sample"]),
        batch_size=int(wl.get("batch_size") or _DEFAULT_WORKLOAD["batch_size"]),
        gpus_per_node=int(wl.get("gpus_per_node") or _DEFAULT_WORKLOAD["gpus_per_node"]),
        number_of_nodes=int(wl.get("number_of_nodes") or _DEFAULT_WORKLOAD["number_of_nodes"]),
    )

    # Cluster budget: --cluster-gpus if given, else infrastructure.yaml. The config grid still
    # applies but is capped so no recommendation exceeds the cluster.
    max_gpus, gpus_per_node, max_nodes = resolve_cluster_caps(cluster_gpus)
    context_gpus = grid_gpus + ([] if gpu_model in grid_gpus else [gpu_model])
    context = SystemContext.for_gpus(
        context_gpus,
        max_gpus=max_gpus,
        gpus_per_node=gpus_per_node,
        max_nodes=max_nodes,
    )
    return workload, context


def _apply_max_slowdown(config: dict) -> None:
    """Map ``strategy.max_slowdown`` to the engine's ``runtime_guard_k``, as the batch CSV path does."""
    strategy = config.get("strategy") or {}
    if strategy.get("max_slowdown") is not None:
        strategy["runtime_guard_k"] = float(strategy["max_slowdown"])


def _load_json_job(path: str) -> tuple[WorkloadSpec, SystemContext]:
    """The workload and context of an --input JSON job; ValueError naming the shape otherwise."""
    with open(path, encoding="utf-8") as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"--input {path} is not JSON ({exc}). Single mode reads a JSON job {_JSON_JOB_SHAPE}; "
                "for a CSV of workloads, add --output."
            ) from exc
    workload = payload.get("workload") if isinstance(payload, dict) else None
    context = payload.get("context") if isinstance(payload, dict) else None
    if not (isinstance(workload, dict) and isinstance(context, dict)):
        raise ValueError(f"--input {path} must be a JSON object of the form {_JSON_JOB_SHAPE}")
    return WorkloadSpec(**workload), SystemContext(**context)


def _build_parser() -> FriendlyParser:
    parser = FriendlyParser(
        prog="coastline recommend-job",
        description="GPU Recommendation Engine",
        example="coastline recommend-job --config experiment.yaml",
    )
    parser.add_argument("--config", default=str(default_experiment_path()))
    parser.add_argument("--input", help=f"JSON job {_JSON_JOB_SHAPE} (overrides the config workload)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Write recommendation.json here (default: print to stdout, write nothing; "
        "OUTPUT_DIR env sets a root for per-run subdirectories).",
    )
    parser.add_argument(
        "--cluster-gpus",
        type=positive_int,
        default=None,
        help="Total cluster GPUs (default: infrastructure.yaml's total_gpus). "
        "The recommendation never exceeds the cluster.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    setup_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)
    with report_errors(parser):
        _run(args)


def _run(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    if not config_path.is_file():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    strategy_config = load_strategy_config(config_path)
    _apply_max_slowdown(strategy_config)
    run_id = os.environ.get("RUN_ID", datetime.now(UTC).strftime("%Y_%m_%d_%H_%M_%S"))
    # Files are written only with --output-dir or OUTPUT_DIR; otherwise the recommendation
    # goes to stdout.
    output_root = os.environ.get("OUTPUT_DIR")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif output_root:
        output_dir = Path(output_root) / run_id
    else:
        output_dir = None

    logger.info("Recommender starting | run_id=%s | config=%s", run_id, config_path)

    if args.input:
        workload, context = _load_json_job(args.input)
    else:
        workload, context = _workload_and_context(config_path, raw, args.cluster_gpus)

    strategy_name = strategy_config.get("strategy", {}).get("name", "multi_objective")
    preset = strategy_config.get("strategy", {}).get("preset")
    recs, _ = engine.run_request(
        engine.RecommendRequest(
            workload=workload,
            context=context,
            config=strategy_config,
            strategy_name=strategy_name,
            preset=preset,
        )
    )
    if not recs:
        logger.error("No recommendation generated")
        sys.exit(1)

    result = recs[0]
    if output_dir is None:
        print(json.dumps(recommendation_payload(result), indent=2))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "recommendation.json"
    save_recommendation_to_json(result, out_path)
    logger.info("Recommendation written to %s", out_path)


if __name__ == "__main__":
    main()
