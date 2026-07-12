"""`coastline trace-to-runs` — convert a fine-tuning trace CSV to the flat measured-runs schema."""

from __future__ import annotations

from typing import Optional, Sequence

from coastline.cli._shared import FriendlyParser


def _build_parser() -> FriendlyParser:
    p = FriendlyParser(
        prog="coastline trace-to-runs",
        description="Convert a fine-tuning trace CSV (metadata.*/resources.* columns) into the flat "
        "measured-runs CSV consumed by `coastline tune`, the cache/intelligent lookup, and "
        "`kavier calibrate`. An already-flat CSV is passed through unchanged.",
        example="coastline trace-to-runs --input trace.csv --output run_database.csv",
    )
    p.add_argument("--input", required=True, help="Input trace CSV (or an already-flat measured-runs CSV).")
    p.add_argument("--output", required=True, help="Output flat measured-runs CSV.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    from coastline.sdk.trace.to_runs import trace_to_runs

    df = trace_to_runs(args.input, args.output)
    valid = int(df["is_valid"].sum()) if "is_valid" in df.columns else len(df)
    print(f"wrote {args.output}: {len(df)} rows ({valid} valid) in the flat measured-runs schema")


if __name__ == "__main__":
    main()
