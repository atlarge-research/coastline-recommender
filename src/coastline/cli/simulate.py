"""`coastline simulate` — predict ONE declared configuration, without ranking anything."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from coastline.cli._shared import FriendlyParser
from coastline.sdk.constants import DEFAULT_GPUS_PER_NODE


def _build_parser() -> FriendlyParser:
    p = FriendlyParser(
        prog="coastline simulate",
        description="Predict throughput, power, runtime and energy for ONE configuration you "
        "declare. No grid, no ranking: this is the recommender's simulate step on its own.",
        example="coastline simulate --model mistral-7b-v0.1 --method lora "
        "--gpu-model NVIDIA-A100-SXM4-80GB --tokens 1024 --batch-size 16 --gpus-per-node 4",
    )
    # Every flag also answers to its underscore spelling (--model_name, --tokens_per_sample, ...),
    # the form the thesis listings print; the hyphenated spellings are unchanged.
    p.add_argument("--model", "--model_name", required=True, help="LLM model name (Kavier catalog spelling).")
    p.add_argument("--method", required=True, help="Fine-tuning method: full | lora | qlora.")
    p.add_argument("--gpu-model", "--gpu_model", required=True, help="GPU model, e.g. NVIDIA-A100-SXM4-80GB.")
    p.add_argument(
        "--tokens", "--tokens_per_sample", type=int, required=True, help="Tokens per sample (sequence length)."
    )
    p.add_argument("--batch-size", "--batch_size", type=int, required=True, help="Per-device batch size.")
    # The layout is declared EITHER per node or as a total, never both: the two would contradict.
    # Both default to None so `_resolve_layout` can tell "not given" from "given as 1".
    layout = p.add_mutually_exclusive_group()
    layout.add_argument("--gpus-per-node", type=int, default=None, help="GPUs per node (default: 1).")
    layout.add_argument(
        "--number_gpus",
        type=int,
        default=None,
        help="TOTAL GPUs across every node — NOT a synonym for --gpus-per-node. Divided by --nodes "
        f"to get the per-node layout; without --nodes it must fit one node (<= {DEFAULT_GPUS_PER_NODE}).",
    )
    p.add_argument("--nodes", "--number_nodes", type=int, default=None, help="Number of nodes (default: 1).")
    p.add_argument(
        "--total-tokens",
        type=int,
        default=0,
        help="Dataset size in tokens. Required for runtime and energy: the analytical engine "
        "reports per-step time, not total runtime, so both are omitted without it.",
    )
    p.add_argument(
        "--predictor",
        default="kavier",
        help="Throughput predictor: kavier (default, analytical physics) | intelligent | cache "
        "| a named data-driven model (catboost, xgboost, ...).",
    )
    p.add_argument(
        "--feasibility",
        default="autoconf",
        help="Feasibility checker: autoconf (default, real OOM check via AutoConf) "
        "| rules (divisibility-only, works without AutoConf) | none.",
    )
    p.add_argument("--json", action="store_true", help="Emit the raw result as JSON instead of a text report.")
    return p


def _resolve_layout(parser: FriendlyParser, args: argparse.Namespace) -> tuple[int, int]:
    """The ``(gpus_per_node, nodes)`` layout behind either spelling.

    ``--gpus-per-node`` declares the per-node width directly. ``--number_gpus`` declares the
    TOTAL across the cluster, so the per-node width is derived from ``--nodes`` and must divide
    evenly. The parser makes the two mutually exclusive, so at most one is set here.
    """
    nodes = args.nodes
    if nodes is not None and nodes < 1:
        parser.error("--nodes must be at least 1")

    if args.number_gpus is None:
        gpus_per_node = 1 if args.gpus_per_node is None else args.gpus_per_node
        if gpus_per_node < 1:
            parser.error("--gpus-per-node must be at least 1")
        return gpus_per_node, 1 if nodes is None else nodes

    total = args.number_gpus
    if total < 1:
        parser.error("--number_gpus must be at least 1")
    if nodes is not None:
        if total % nodes:
            parser.error(
                f"--number_gpus {total} does not divide evenly over --nodes {nodes}: "
                "every node must hold the same number of GPUs"
            )
        return total // nodes, nodes
    if total > DEFAULT_GPUS_PER_NODE:
        parser.error(
            f"--number_gpus {total} is more than the {DEFAULT_GPUS_PER_NODE} GPUs of a single node; "
            "add --nodes N (or --number_nodes N) to declare the layout"
        )
    return total, 1


def _format_report(result: dict) -> str:
    """Human-readable report. Mirrors the JSON keys; omits what could not be derived."""
    lines = [
        f"config    {result['gpus_per_node']}x{result['number_of_nodes']} "
        f"= {result['total_gpus']} GPU(s), batch {result['batch_size']} per device",
        f"workload  {result['llm_model']} / {result['fine_tuning_method']} / "
        f"{result['gpu_model']} / {result['tokens_per_sample']} tok",
        f"predictor {result['predictor']}  feasibility={result['feasibility_mode']}",
        "",
    ]
    feasible = result["feasible"]
    lines.append(f"feasible  {'yes' if feasible else 'no'}")
    if result["error"]:
        lines.append(f"error     {result['error']}")
        return "\n".join(lines)

    lines.append(f"thr       {result['predicted_throughput']:.2f} tok/s")
    if result["predicted_power_watts"] is not None:
        lines.append(
            f"power     {result['predicted_power_watts']:.2f} W per GPU, {result['cluster_power_watts']:.2f} W total"
        )
        lines.append(f"tok/W     {result['tokens_per_watt']:.2f}")
    if result["predicted_runtime_seconds"] is not None:
        historical = result["runtime_source"] == "predictor_history"
        provenance = "  (the matched historical run, NOT this dataset)" if historical else ""
        lines.append(f"runtime   {result['predicted_runtime_seconds']:.2f} s{provenance}")
    if result["energy_kwh"] is not None:
        lines.append(f"energy    {result['energy_kwh']:.4f} kWh")
    if result["runtime_source"] != "total_tokens":
        lines.append("")
        lines.append("note      pass --total-tokens for a runtime and energy derived from YOUR dataset.")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.total_tokens < 0:
        parser.error("--total-tokens cannot be negative")

    from coastline.sdk.models.context import SystemContext
    from coastline.sdk.models.workload import WorkloadSpec
    from coastline.sdk.recommend.simulate import simulate_one

    gpus_per_node, nodes = _resolve_layout(parser, args)
    total_gpus = gpus_per_node * nodes
    workload = WorkloadSpec(
        llm_model=args.model,
        fine_tuning_method=args.method,
        gpu_model=args.gpu_model,
        tokens_per_sample=args.tokens,
        batch_size=args.batch_size,
        gpus_per_node=gpus_per_node,
        number_of_nodes=nodes,
    )
    context = SystemContext.for_gpus(
        [args.gpu_model],
        max_gpus=total_gpus,
        gpus_per_node=gpus_per_node,
        max_nodes=nodes,
    )

    result = simulate_one(
        workload,
        context,
        predictor=args.predictor,
        feasibility=args.feasibility,
        total_tokens=args.total_tokens,
    )

    print(json.dumps(result, indent=2) if args.json else _format_report(result))
    if result["error"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
