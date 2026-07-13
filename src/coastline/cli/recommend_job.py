"""``coastline recommend-job`` — recommend GPU/node configs for ONE job.

One verb, three mutually-exclusive input modes (all route into the same engine):

* ``--interactive``                       guided keyboard REPL (was ``coastline interactive``)
* ``--config CFG`` [``--output-dir DIR``] one declared job → ``recommendation.json`` / stdout (was ``coastline run``)
* ``--input CSV --output CSV --config CFG`` a batch of jobs, CSV → CSV (was ``coastline recommend``)

``recommend-trace`` is the sibling verb for a whole fine-tuning trace (the ``ibm_trace`` format);
this verb speaks the canonical Coastline workload CSV.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from coastline.cli._shared import FriendlyParser


def _build_parser() -> FriendlyParser:
    p = FriendlyParser(
        prog="coastline recommend-job",
        description="Recommend GPU/node configurations for one job: --interactive (guided REPL), "
        "--config (one declared job), or --input/--output (a batch CSV of jobs).",
        example="coastline recommend-job --config config.yaml --input workloads.csv --output recs.csv",
    )
    p.add_argument("--interactive", action="store_true", help="Guided keyboard-driven REPL over the recommender.")
    p.add_argument("--config", help="Config YAML (strategy, predictors, grid, safeguards).")
    p.add_argument(
        "--input",
        help="Batch mode: input CSV of workloads. Single mode: a JSON job that overrides the config workload.",
    )
    p.add_argument("--output", help="Batch mode: output CSV path for the recommendations.")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Single mode: write recommendation.json here (default: print the JSON to stdout).",
    )
    p.add_argument(
        "--cluster-gpus",
        type=int,
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
        # Batch CSV → CSV mode (was `coastline recommend`): the config carries the policy.
        if not ns.input:
            parser.error("--output needs --input (batch CSV -> CSV)")
        if not ns.config:
            parser.error("batch mode (--input/--output) needs --config for the recommendation policy")
        from coastline.sdk.recommend.batch_csv import recommend_csv

        recommend_csv(ns.config, ns.input, ns.output, cluster_gpus=ns.cluster_gpus)
        return

    if ns.config:
        # Single declared-job mode (was `coastline run`); --input here is an optional JSON override.
        from coastline.cli.run import main as run_main

        run_argv = ["--config", ns.config]
        if ns.input:
            run_argv += ["--input", ns.input]
        if ns.output_dir:
            run_argv += ["--output-dir", ns.output_dir]
        if ns.cluster_gpus is not None:
            run_argv += ["--cluster-gpus", str(ns.cluster_gpus)]
        run_main(run_argv)
        return

    parser.error("choose a mode: --interactive, --config <file>, or --input <csv> --output <csv> --config <file>")


if __name__ == "__main__":
    main()
