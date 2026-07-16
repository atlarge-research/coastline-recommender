"""`coastline recommend-trace` — recommend a config for every job in a fine-tuning trace CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from coastline.cli._args import add_trace_layout_args
from coastline.cli._shared import FriendlyParser
from coastline.sdk.trace.recommend import recommend_trace


def _build_parser() -> FriendlyParser:
    p = FriendlyParser(
        prog="coastline recommend-trace",
        description="Recommend a config for every job in a fine-tuning trace CSV "
        "(adds the recommended layout + estimated duration per row).",
        example="coastline recommend-trace --input trace.csv --output recommended.csv --method kavier",
    )
    p.add_argument("--input", required=True, help="Input trace CSV.")
    p.add_argument("--output", required=True, help="Output (recommended trace) CSV.")
    p.add_argument(
        "--goal",
        default="min_gpu",
        choices=["min_gpu", "performance", "energy", "balanced"],
        help="Optimisation goal for recommendations (default: min_gpu).",
    )
    p.add_argument(
        "--feasibility",
        default="autoconf",
        help="Feasibility checker: autoconf (default, real OOM check via AutoConf) "
        "| rules (divisibility-only, works without AutoConf).",
    )
    p.add_argument(
        "--lookup",
        default=None,
        help="Measured-runs CSV for --method cache/intelligent (flat sfttrainer schema), "
        "or 'default' for the small bundled lookup DB. Default: $DATA_DIR/profiling-dataset/"
        "raw_trace.csv when set, else the bundled sample.",
    )
    p.add_argument(
        "--visual",
        action="store_true",
        help="Also render the operational cluster timeline (GPUs in use + jobs queued over "
        "time) by FIFO-scheduling the recommendations onto a fixed cluster.",
    )
    p.add_argument(
        "--visual-output", default=None, help="Path for the --visual figure (default: --output with a .pdf suffix)."
    )
    p.add_argument(
        "--tokens-col",
        default=None,
        metavar="COL",
        help=(
            "Column to use as tokens_per_sample (sequence length) for the predictor. "
            "Defaults to metadata.tokens_per_sample. "
            "Override with e.g. --tokens-col metadata.estimated_max_seq_length to use "
            "the actual (shrunk) sequence length instead of the nominal dataset max."
        ),
    )
    p.add_argument(
        "--tot-tokens-col",
        default=None,
        metavar="COL",
        help=(
            "Column holding the pre-computed total token count per job "
            "(e.g. metadata.output.extrapolated_num_tokens). "
            "When provided, metadata.estimated_duration_<method> is written as "
            "setup_time + tot_tokens / estimated_throughput (or tot_tokens / throughput "
            "when --setup-time-col is omitted). "
            "When omitted, falls back to train_tokens_per_second × train_runtime "
            "(legacy — requires output columns in the trace)."
        ),
    )
    p.add_argument(
        "--setup-time-col",
        default=None,
        metavar="COL",
        help=(
            "Column holding the per-job setup overhead in seconds "
            "(e.g. metadata.output.setup_time). "
            "When provided together with --tot-tokens-col, the duration formula is "
            "setup_time + tot_tokens / estimated_throughput, "
            "matching the add_auxiliary_information.py identity exactly. "
            "Ignored when --tot-tokens-col is not set."
        ),
    )
    add_trace_layout_args(p)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    from coastline.sdk.io.infrastructure import resolve_cluster_caps
    from coastline.sdk.trace.recommend import _TOKENS as _DEFAULT_TOKENS

    cluster_gpus, node_gpus, _ = resolve_cluster_caps(args.cluster_gpus, args.node_gpus)
    df = recommend_trace(
        args.input,
        args.output,
        method=args.method,
        goal=args.goal,
        feasibility=args.feasibility,
        lookup=args.lookup,
        cluster_gpus=cluster_gpus,
        node_gpus=node_gpus,
        tokens_col=args.tokens_col if args.tokens_col else _DEFAULT_TOKENS,
        tot_tokens_col=args.tot_tokens_col if args.tot_tokens_col else None,
        setup_time_col=args.setup_time_col if args.setup_time_col else None,
    )
    thr_col = f"metadata.estimated_throughput_{args.method}"
    dur_col = f"metadata.estimated_duration_{args.method}"
    n_thr = df[thr_col].notna().sum() if thr_col in df.columns else 0
    n_dur = df[dur_col].notna().sum() if dur_col in df.columns else 0
    print(
        f"wrote {args.output}: {len(df)} rows, "
        f"{n_thr} with estimated_throughput, {n_dur} with estimated_duration "
        f"(cluster {cluster_gpus} GPUs)"
    )
    if n_dur < len(df):
        print(f"note: {len(df) - n_dur} row(s) without a duration — pass --tot-tokens-col to enable duration estimates")
    if args.visual:
        from coastline.sdk.trace.plot import plot_trace_timeline

        viz = args.visual_output or str(Path(args.output).with_suffix(".pdf"))
        stats = plot_trace_timeline(
            args.output, viz, method=args.method, cluster_gpus=cluster_gpus, node_gpus=node_gpus
        )
        print(f"wrote {viz}: cluster timeline ({stats})")


if __name__ == "__main__":
    main()
