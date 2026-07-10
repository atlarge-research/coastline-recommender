"""`coastline tune` — tune a data-driven predictor on a measured-runs CSV."""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from coastline.cli._shared import FriendlyParser


def _build_parser() -> FriendlyParser:
    p = FriendlyParser(
        prog="coastline tune",
        description="Tune a data-driven throughput/runtime predictor on your own measured-runs CSV. "
        "Run `coastline tune --format` to see what a valid dataset looks like.",
        example="coastline tune --data runs.csv --model tabpfn --train-percentage 1.0",
    )
    p.add_argument("--data", help="Measured-runs CSV (one fine-tuning run per row; see --format).")
    p.add_argument("--model", default="tabpfn", help="Model to tune (currently: tabpfn).")
    p.add_argument(
        "--train-percentage",
        type=float,
        default=1.0,
        help="Fraction of valid rows used for tuning; 1.0 (default) = all rows, no holdout. "
        "Below 1.0 the remainder becomes a test split and MdAPE is reported.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Artifact path (default: models/custom/<model>.pkl, auto-discovered "
        "by --method/<predictors.performance> = tabpfn).",
    )
    p.add_argument("--seed", type=int, default=42, help="Split seed (default 42).")
    p.add_argument("--format", action="store_true", help="Print the dataset format contract and exit.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    from coastline.sdk.predictors.performance.data_driven.tune import (
        DatasetFormatError,
        dataset_format_help,
        tune,
    )

    if args.format:
        print(dataset_format_help())
        return
    if not args.data:
        _build_parser().error("--data is required (or use --format to see the dataset contract)")

    try:
        result = tune(
            args.data,
            model=args.model,
            train_percentage=args.train_percentage,
            output=args.output,
            seed=args.seed,
            on_step=lambda msg: print(msg, flush=True),
        )
    except (DatasetFormatError, RuntimeError, ValueError) as exc:
        print(f"coastline tune: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(
        f"\ntuned {args.model} ({result['tune_id']}) on {result['rows_train']} rows "
        f"in {result['fit_seconds']}s on {result['device']}"
    )
    if result["metrics"]:
        print(
            f"holdout ({result['rows_test']} rows): "
            f"throughput MdAPE {result['metrics']['test_mdape_throughput_pct']:.1f}% · "
            f"runtime MdAPE {result['metrics']['test_mdape_runtime_pct']:.1f}%"
        )
    print(f"serve it with: coastline recommend-trace ... --method {args.model}")
    if result["warnings"]:
        print(
            "\nWARNING: Tuning may have produced poor results because valid datasets should have these properties:",
            file=sys.stderr,
        )
        for w in result["warnings"]:
            print(f"  - {w}", file=sys.stderr)


if __name__ == "__main__":
    main()
