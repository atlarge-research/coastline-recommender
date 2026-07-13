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

from coastline.sdk.io.infrastructure import resolve_cluster_caps
from coastline.sdk.io.interface.json_output import recommendation_payload, save_recommendation_to_json
from coastline.sdk.io.run_config import load_strategy_config
from coastline.sdk.logging import setup_logging
from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.recommend import engine

# 3.8+ timezone alias
UTC = timezone.utc

logger = logging.getLogger(__name__)

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

    gpu_model = system_cfg.get("default_gpu") or "NVIDIA-A100-SXM4-80GB"
    if grid_cfg.get("gpu_models"):
        gpu_model = grid_cfg["gpu_models"][0]

    wl = {**_DEFAULT_WORKLOAD, **{k: v for k, v in workload_cfg.items() if k in WorkloadSpec.model_fields}}
    workload = WorkloadSpec(
        llm_model=wl.get("llm_model", _DEFAULT_WORKLOAD["llm_model"]),
        fine_tuning_method=wl.get("fine_tuning_method", _DEFAULT_WORKLOAD["fine_tuning_method"]),
        gpu_model=gpu_model,
        # `... or default` (not dict.get default) so an explicit YAML null/0 falls
        # back to the default instead of crashing int(None) (e.g. config/coastline_functionality/experiment.yaml).
        tokens_per_sample=int(wl.get("tokens_per_sample") or _DEFAULT_WORKLOAD["tokens_per_sample"]),
        batch_size=int(wl.get("batch_size") or _DEFAULT_WORKLOAD["batch_size"]),
        gpus_per_node=int(wl.get("gpus_per_node") or _DEFAULT_WORKLOAD["gpus_per_node"]),
        number_of_nodes=int(wl.get("number_of_nodes") or _DEFAULT_WORKLOAD["number_of_nodes"]),
    )

    # Cluster budget: --cluster-gpus if given, else infrastructure.yaml. The config grid still
    # applies but is capped so no recommendation exceeds the cluster.
    max_gpus, gpus_per_node, max_nodes = resolve_cluster_caps(cluster_gpus)
    context = SystemContext.for_gpus(
        grid_cfg.get("gpu_models") or [gpu_model],
        max_gpus=max_gpus,
        gpus_per_node=gpus_per_node,
        max_nodes=max_nodes,
    )
    return workload, context


def main(argv: Sequence[str] | None = None) -> None:
    setup_logging()
    parser = argparse.ArgumentParser(prog="coastline run", description="GPU Recommendation Engine")
    parser.add_argument(
        "--config", default=os.environ.get("CONFIG_FILE", "./config/coastline_functionality/default.yaml")
    )
    parser.add_argument("--input", help="JSON input file (overrides workload/context)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Write recommendation.json here (default: print to stdout, write nothing; "
        "OUTPUT_DIR env sets a root for per-run subdirectories).",
    )
    parser.add_argument(
        "--cluster-gpus",
        type=int,
        default=None,
        help="Total cluster GPUs (default: infrastructure.yaml's total_gpus). "
        "The recommendation never exceeds the cluster.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_file():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    strategy_config = load_strategy_config(config_path)
    run_id = os.environ.get("RUN_ID", datetime.now(UTC).strftime("%Y_%m_%d_%H_%M_%S"))
    # Artifacts are opt-in: without --output-dir or OUTPUT_DIR nothing is written
    # (the recommendation prints to stdout) — no stray recommender/ dir in the CWD.
    output_root = os.environ.get("OUTPUT_DIR")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif output_root:
        output_dir = Path(output_root) / run_id
    else:
        output_dir = None

    logger.info("Recommender starting | run_id=%s | config=%s", run_id, config_path)

    if args.input:
        with open(args.input, encoding="utf-8") as f:
            payload = json.load(f)
        workload = WorkloadSpec(**payload["workload"])
        context = SystemContext(**payload["context"])
    else:
        workload, context = _workload_and_context(config_path, raw, args.cluster_gpus)

    strategy_name = strategy_config.get("strategy", {}).get("name", "min_gpu")
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
