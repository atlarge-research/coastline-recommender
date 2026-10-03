"""``coastline recommend-job``: recommend GPU/node configurations for one job.

Three input modes, all served by the same engine:

* ``--interactive``                          guided keyboard REPL
* ``--config CFG`` [``--output-dir DIR``]    one declared job, to ``recommendation.json`` or stdout
* ``--input CSV --output CSV --config CFG``  a batch of jobs, CSV in and CSV out

With no flags it runs the second mode on the default config
(:func:`~coastline.sdk.io.run_config.default_experiment_path`), which declares its own workload.

``recommend-trace`` handles a whole fine-tuning trace (the ``ibm_trace`` format); this command
reads the Coastline workload CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from coastline.cli._shared import FriendlyParser, positive_int, report_errors


def _require_files(parser: FriendlyParser, ns: argparse.Namespace) -> None:
    """A usage error for a --config or --input path that is not a file."""
    for flag, path in (("--config", ns.config), ("--input", ns.input)):
        if path and not Path(path).is_file():
            parser.error(f"{flag} {path}: no such file")


def _build_parser() -> FriendlyParser:
    p = FriendlyParser(
        prog="coastline recommend-job",
        description="Recommend GPU/node configurations for one job: --interactive (guided REPL), "
        "--config (one declared job), or --input/--output (a batch CSV of jobs). "
        "With no flags, the declared job of the default config.",
        example="coastline recommend-job --config config.yaml --input workloads.csv --output recs.csv",
    )
    p.add_argument("--interactive", action="store_true", help="Guided keyboard-driven REPL over the recommender.")
    p.add_argument("--config", help="Config YAML (strategy, predictors, grid, safeguards).")
    p.add_argument(
        "--input",
        help='Batch mode: input CSV of workloads. Single mode: a JSON job {"workload": {...}, "context": {...}} '
        "that overrides the config workload.",
    )
    p.add_argument("--output", help="Batch mode: output CSV path for the recommendations.")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Single mode: write recommendation.json here (default: print the JSON to stdout).",
    )
    p.add_argument(
        "--cluster-gpus",
        type=positive_int,
        default=None,
        help="Total cluster GPUs (default: infrastructure.yaml's total_gpus). "
        "No job is recommended more GPUs than the cluster has.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)

    # --interactive hands the remaining args to the guided REPL (its own flags: --top-k, --save, ...).
    if "--interactive" in args:
        from coastline.cli import interactive

        interactive.run([a for a in args if a != "--interactive"])
        return

    parser = _build_parser()
    ns = parser.parse_args(args)

    if ns.output:
        # Batch mode, CSV in and CSV out; the config carries the policy.
        if not ns.input:
            parser.error("--output needs --input (batch CSV -> CSV)")
        if not ns.config:
            parser.error("batch mode (--input/--output) needs --config for the recommendation policy")
        _require_files(parser, ns)
        from coastline.sdk.recommend.batch_csv import recommend_csv

        with report_errors(parser):
            recommend_csv(ns.config, ns.input, ns.output, cluster_gpus=ns.cluster_gpus)
        return

    if ns.input and not ns.config:
        # --input is a JSON workload override in single mode and the source CSV in batch mode.
        # Either way it needs a --config for the policy; the default config is not used.
        parser.error(
            "--input needs --config: a JSON job overriding the config workload (single mode), "
            "or --output too for a batch CSV -> CSV"
        )
    _require_files(parser, ns)

    # Single-job mode; --input is an optional JSON override. Without --config, run.py uses
    # default_experiment_path(), whose workload: block is the declared job.
    from coastline.cli.run import main as run_main

    run_argv = ["--config", ns.config] if ns.config else []
    if ns.input:
        run_argv += ["--input", ns.input]
    if ns.output_dir:
        run_argv += ["--output-dir", ns.output_dir]
    if ns.cluster_gpus is not None:
        run_argv += ["--cluster-gpus", str(ns.cluster_gpus)]
    run_main(run_argv)


if __name__ == "__main__":
    main()
