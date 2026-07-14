"""`coastline utils plot-trace` — plot a recommended trace's operational cluster timeline."""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from coastline.cli._args import add_trace_layout_args
from coastline.cli._shared import FriendlyParser
from coastline.sdk.trace.plot import _ORIG_GPUS, _ORIG_NODES, plot_trace_timeline

# Columns used when --baseline is set (original layout, never overwritten by recommend-trace).
_BASELINE_GPUS_COL  = _ORIG_GPUS   # "metadata.orig_number_gpus"
_BASELINE_NODES_COL = _ORIG_NODES  # "metadata.orig_num_nodes"


def _build_parser() -> FriendlyParser:
    p = FriendlyParser(
        prog="coastline utils plot-trace",
        description="Plot a recommended trace: the operational cluster timeline (GPUs in use + jobs queued over time).",
        example=(
            "coastline utils plot-trace --input recommended.csv --output timeline.pdf\n"
            "# baseline profile from a patched trace (orig layout, no recommendation):\n"
            "coastline utils plot-trace --input recommended.csv --output baseline.pdf \\\n"
            "    --baseline \\\n"
            "    --duration-col metadata.output.extrapolated_duration \\\n"
            "    --label        baseline"
        ),
    )
    p.add_argument("--input",  required=True, help="Trace CSV (recommended or raw).")
    p.add_argument("--output", required=True, help="Output path for the timeline figure.")
    p.add_argument(
        "--baseline",
        action="store_true",
        default=False,
        help=(
            "Use the original (pre-recommendation) GPU layout columns: "
            f"{_BASELINE_GPUS_COL} and {_BASELINE_NODES_COL}.  "
            "These are preserved by coastline recommend-trace, so --baseline works on both "
            "raw and patched traces.  Mutually exclusive with --gpus-per-node-col/--nodes-col."
        ),
    )
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
        "--gpus-per-node-col",
        default=None,
        metavar="COL",
        help=(
            "Column to use as GPUs-per-node.  "
            "Defaults to resources.num_gpus_per_node.  "
            "Cannot be combined with --baseline."
        ),
    )
    p.add_argument(
        "--nodes-col",
        default=None,
        metavar="COL",
        help=(
            "Column to use as number of nodes.  "
            "Defaults to resources.num_nodes.  "
            "Cannot be combined with --baseline."
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

    if args.baseline and (args.gpus_per_node_col or args.nodes_col):
        print(
            "error: --baseline is mutually exclusive with --gpus-per-node-col / --nodes-col",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if args.baseline:
        gpus_per_node_col = _BASELINE_GPUS_COL
        nodes_col         = _BASELINE_NODES_COL
        print(
            f"note: --baseline — using original layout columns:\n"
            f"  gpus-per-node : {_BASELINE_GPUS_COL}\n"
            f"  nodes         : {_BASELINE_NODES_COL}",
            file=sys.stderr,
        )
    else:
        gpus_per_node_col = args.gpus_per_node_col
        nodes_col         = args.nodes_col

    from coastline.sdk.io.infrastructure import resolve_cluster_caps

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
            gpus_per_node_col=gpus_per_node_col,
            nodes_col=nodes_col,
            label=args.label,
        )
    )


if __name__ == "__main__":
    main()
