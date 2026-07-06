"""`coastline plot-trace` — plot an enriched trace (cluster timeline and/or impact scatter)."""

from __future__ import annotations

from typing import Optional, Sequence

from coastline.cli._args import add_trace_layout_args
from coastline.cli._shared import FriendlyParser
from coastline.sdk.trace.plot import impact_output, plot_trace_performance, plot_trace_timeline


def _build_parser() -> FriendlyParser:
    p = FriendlyParser(
        prog="coastline plot-trace",
        description="Plot a coastline-enriched trace: operational cluster timeline "
        "(default) and/or recommendation-impact scatter.",
        example="coastline plot-trace --input enriched.csv --output timeline.pdf --view timeline",
    )
    p.add_argument("--input", required=True, help="Enriched trace CSV (from coastline enrich-trace).")
    p.add_argument("--output", required=True, help="Output path (the timeline, unless --view impact).")
    p.add_argument(
        "--view",
        choices=["impact", "timeline", "both"],
        default="timeline",
        help="timeline (default): FIFO cluster timeline (GPUs in use + jobs queued). "
        "impact: recommended-vs-original duration scatter. both: write both "
        "(timeline -> --output, scatter -> <output stem>_impact).",
    )
    add_trace_layout_args(p)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.view in ("timeline", "both"):
        print(
            plot_trace_timeline(
                args.input, args.output, method=args.method, cluster_gpus=args.cluster_gpus, node_gpus=args.node_gpus
            )
        )
    if args.view in ("impact", "both"):
        impact_png = args.output if args.view == "impact" else impact_output(args.output)
        print(plot_trace_performance(args.input, impact_png, method=args.method))


if __name__ == "__main__":
    main()
