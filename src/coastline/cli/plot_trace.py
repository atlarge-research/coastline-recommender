"""`coastline plot-trace` — plot an enriched trace's operational cluster timeline."""

from __future__ import annotations

from typing import Optional, Sequence

from coastline.cli._args import add_trace_layout_args
from coastline.cli._shared import FriendlyParser
from coastline.sdk.trace.plot import plot_trace_timeline


def _build_parser() -> FriendlyParser:
    p = FriendlyParser(
        prog="coastline plot-trace",
        description="Plot a coastline-enriched trace: the operational cluster timeline "
        "(GPUs in use + jobs queued over time).",
        example="coastline plot-trace --input enriched.csv --output timeline.pdf",
    )
    p.add_argument("--input", required=True, help="Enriched trace CSV (from coastline enrich-trace).")
    p.add_argument("--output", required=True, help="Output path for the timeline figure.")
    add_trace_layout_args(p)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    print(
        plot_trace_timeline(
            args.input, args.output, method=args.method, cluster_gpus=args.cluster_gpus, node_gpus=args.node_gpus
        )
    )


if __name__ == "__main__":
    main()
