"""`coastline recommend-trace` — recommend a config for every job in a fine-tuning trace CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from coastline.cli._args import add_trace_layout_args
from coastline.cli._shared import FriendlyParser
from coastline.sdk.trace.recommend import recommend_trace


def _build_parser() -> FriendlyParser:
    p = FriendlyParser(
        prog="coastline recommend-trace",
        description="Recommend a config for every job in a fine-tuning trace CSV "
        "(adds the recommended layout + estimated duration per row).",
        example="coastline recommend-trace --input trace.csv --output recommended.csv --method kavier",
    )
    p.add_argument("--input", required=True, help="Input trace CSV.")
    p.add_argument("--output", required=True, help="Output (recommended trace) CSV.")
    p.add_argument(
        "--goal",
        default="min_gpu",
        choices=["min_gpu", "performance", "energy", "balanced"],
        help="Optimisation goal for recommendations (default: min_gpu).",
    )
    p.add_argument(
        "--feasibility",
        default="autoconf",
        help="Feasibility checker: autoconf (default, real OOM check via AutoConf) "
        "| rules (divisibility-only, works without AutoConf).",
    )
    p.add_argument(
        "--lookup",
        default=None,
        help="Measured-runs CSV for --method cache/intelligent (flat sfttrainer schema), "
        "or 'default' for the small bundled lookup DB. Default: $DATA_DIR/profiling-dataset/"
        "raw_trace.csv when set, else the bundled sample.",
    )
    p.add_argument(
        "--visual",
        action="store_true",
        help="Also render the operational cluster timeline (GPUs in use + jobs queued over "
        "time) by FIFO-scheduling the recommendations onto a fixed cluster.",
    )
    p.add_argument(
        "--visual-output", default=None, help="Path for the --visual figure (default: --output with a .pdf suffix)."
    )
    add_trace_layout_args(p)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    # Resolve the cluster budget ONCE (infrastructure.yaml default, --cluster-gpus/--node-gpus
    # override) so the recommendation cap and the timeline use the exact same cluster.
    from coastline.sdk.io.infrastructure import resolve_cluster_caps

    cluster_gpus, node_gpus, _ = resolve_cluster_caps(args.cluster_gpus, args.node_gpus)
    df = recommend_trace(
        args.input,
        args.output,
        method=args.method,
        goal=args.goal,
        feasibility=args.feasibility,
        lookup=args.lookup,
        cluster_gpus=cluster_gpus,
        node_gpus=node_gpus,
    )
    n = df[f"metadata.estimated_duration_{args.method}"].notna().sum()
    missing = len(df) - n
    print(
        f"wrote {args.output}: {len(df)} rows, {n} with an "
        f"estimated_duration_{args.method} (cluster {cluster_gpus} GPUs)"
    )
    if missing:
        print(f"note: {missing} row(s) without a duration — see warnings above and metadata.recommendation_note")
    if args.visual:
        from coastline.sdk.trace.plot import plot_trace_timeline

        viz = args.visual_output or str(Path(args.output).with_suffix(".pdf"))
        stats = plot_trace_timeline(
            args.output, viz, method=args.method, cluster_gpus=cluster_gpus, node_gpus=node_gpus
        )
        print(f"wrote {viz}: cluster timeline ({stats})")


if __name__ == "__main__":
    main()
