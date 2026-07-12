"""Shared argparse builders (no parsing) reused across subcommands."""

from __future__ import annotations

import argparse


def add_trace_layout_args(parser: argparse.ArgumentParser) -> None:
    """Cluster-layout flags shared by `recommend-trace --visual` and `plot-trace`."""
    parser.add_argument(
        "--method", default="kavier", help="Predictor whose estimate to use: kavier | tabpfn | xgb (default: kavier)."
    )
    parser.add_argument(
        "--cluster-gpus",
        type=int,
        default=None,
        help="Total cluster GPUs. Caps recommendations AND sizes the timeline. "
        "Default: infrastructure.yaml's total_gpus.",
    )
    parser.add_argument(
        "--node-gpus",
        type=int,
        default=None,
        help="GPUs per node (default: infrastructure.yaml's max_gpus_per_node); num_nodes = cluster-gpus // node-gpus.",
    )
