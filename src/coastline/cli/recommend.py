"""`coastline recommend` — batch CSV recommender (a thin wrapper over recommend_csv)."""

from __future__ import annotations

from typing import Optional, Sequence

from coastline.cli._shared import FriendlyParser
from coastline.sdk.recommend.batch_csv import recommend_csv


def _build_parser() -> FriendlyParser:
    p = FriendlyParser(
        prog="coastline recommend",
        description="Recommend GPU/node configurations for a CSV of workloads (CSV in -> CSV out).",
        example="coastline recommend --config config.yaml --input workloads.csv --output recs.csv",
    )
    p.add_argument("--config", required=True, help="Config YAML (strategy, predictors, grid, safeguards).")
    p.add_argument("--input", required=True, help="Input CSV of workloads.")
    p.add_argument("--output", required=True, help="Output CSV path for the recommendations.")
    p.add_argument(
        "--cluster-gpus",
        type=int,
        default=None,
        help="Total cluster GPUs (default: infrastructure.yaml's total_gpus). "
        "No workload is recommended more GPUs than the cluster has.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    recommend_csv(args.config, args.input, args.output, cluster_gpus=args.cluster_gpus)


if __name__ == "__main__":
    main()
