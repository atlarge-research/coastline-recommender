"""Shared argparse builders (no parsing) reused across subcommands."""

from __future__ import annotations

import argparse


def add_trace_layout_args(parser: argparse.ArgumentParser) -> None:
    """Cluster-layout flags shared by `enrich-trace --visual` and `plot-trace`."""
    parser.add_argument(
        "--method", default="kavier", help="Predictor whose estimate to use: kavier | tabpfn | xgb (default: kavier)."
    )
    parser.add_argument("--cluster-gpus", type=int, default=16, help="Total cluster GPUs (default 16).")
    parser.add_argument(
        "--node-gpus", type=int, default=8, help="GPUs per node (default 8); num_nodes = cluster-gpus // node-gpus."
    )
