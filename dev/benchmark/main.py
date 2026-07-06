"""Benchmark suite entry point.

Usage: python -m benchmark.main [--exclude-128gpu] [--kavier-only] [--results-csv NAME]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from coastline.sdk.logging import setup_logging


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Predictor Benchmark Suite")
    parser.add_argument("--exclude-128gpu", action="store_true")
    parser.add_argument("--kavier-only", action="store_true")
    parser.add_argument(
        "--results-csv",
        type=str,
        default=None,
        help="CSV filename under benchmarks/results/ (optional).",
    )
    args = parser.parse_args()

    from .run_benchmark import run_all, run_kavier_only

    max_gpus = 32 if args.exclude_128gpu else None
    results_csv = Path(args.results_csv) if args.results_csv else None

    if args.kavier_only:
        run_kavier_only(max_gpus=max_gpus, results_csv=results_csv)
    else:
        run_all(max_gpus=max_gpus, results_csv=results_csv)


if __name__ == "__main__":
    main()
