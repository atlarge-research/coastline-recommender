"""`coastline utils plot-trace` — plot a recommended trace's operational cluster timeline."""

from __future__ import annotations

from typing import Optional, Sequence

from coastline.cli._args import add_trace_layout_args
from coastline.cli._shared import FriendlyParser
from coastline.sdk.trace.plot import plot_trace_timeline


def _build_parser() -> FriendlyParser:
    p = FriendlyParser(
        prog="coastline utils plot-trace",
        description="Plot a recommended trace: the operational cluster timeline (GPUs in use + jobs queued over time).",
        example=(
            "coastline utils plot-trace --input recommended.csv --output timeline.pdf\n"
            "# baseline (raw trace, no recommendation):\n"
            "coastline utils plot-trace --input trace.csv --output baseline.pdf \\\n"
            "    --duration-col metadata.output.extrapolated_duration \\\n"
            "    --submit-col   metadata.submission_time_issue_85_rescaled \\\n"
            "    --label        baseline"
        ),
    )
    p.add_argument("--input",  required=True, help="Trace CSV (recommended or raw).")
    p.add_argument("--output", required=True, help="Output path for the timeline figure.")
    p.add_argument(
        "--duration-col",
        default=None,
        metavar="COL",
        help=(
            "Column to use as job duration [s].  "
            "Defaults to metadata.estimated_duration_<method>.  "
            "Override to plot a raw trace directly, e.g. "
            "--duration-col metadata.output.extrapolated_duration"
        ),
    )
    p.add_argument(
        "--submit-col",
        default=None,
        metavar="COL",
        help=(
            "Column to use as job submission time (numeric seconds or ISO timestamp).  "
            "Defaults to the first usable column among "
            "metadata.submission_time_issue_85_rescaled / _original / metadata.submission_time.  "
            "Override with e.g. --submit-col metadata.submission_time_issue_85_rescaled"
        ),
    )
    p.add_argument(
        "--label",
        default=None,
        metavar="TEXT",
        help="Label shown in the plot title / legend (default: the --method value).",
    )
    add_trace_layout_args(p)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    from coastline.sdk.io.infrastructure import resolve_cluster_caps

    # infrastructure.yaml default; --cluster-gpus/--node-gpus override.
    cluster_gpus, node_gpus, _ = resolve_cluster_caps(args.cluster_gpus, args.node_gpus)
    print(
        plot_trace_timeline(
            args.input,
            args.output,
            method=args.method,
            cluster_gpus=cluster_gpus,
            node_gpus=node_gpus,
            duration_col=args.duration_col,
            submit_col=args.submit_col,
            label=args.label,
        )
    )


if __name__ == "__main__":
    main()
