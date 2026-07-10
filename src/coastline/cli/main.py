"""The single ``coastline`` command — a lazy subcommand dispatcher.

Each handler imports its subcommand module only when invoked, so ``coastline --help``
and any one command never pull a sibling's heavy dependencies (pandas, kavier, ...).
"""

from __future__ import annotations

import sys
from typing import Callable, Optional, Sequence

from coastline import __version__

_Handler = Callable[[Optional[Sequence[str]]], None]


def _run_recommend(argv: Optional[Sequence[str]]) -> None:
    from coastline.cli.recommend import main

    main(argv)


def _run_run(argv: Optional[Sequence[str]]) -> None:
    from coastline.cli.run import main

    main(argv)


def _run_recommend_trace(argv: Optional[Sequence[str]]) -> None:
    from coastline.cli.recommend_trace import main

    main(argv)


def _run_plot(argv: Optional[Sequence[str]]) -> None:
    from coastline.cli.plot_trace import main

    main(argv)


def _run_interactive(argv: Optional[Sequence[str]]) -> None:
    from coastline.cli.interactive import run

    run(argv)


def _run_tune(argv: Optional[Sequence[str]]) -> None:
    from coastline.cli.tune import main

    main(argv)


_COMMANDS: dict[str, tuple[str, _Handler]] = {
    "recommend": ("Batch-recommend GPU/node configs for a CSV of workloads (CSV in -> CSV out).", _run_recommend),
    "run": ("Run one config-file experiment; write a recommendation.json run artifact.", _run_run),
    "recommend-trace": ("Recommend a config for every job in a fine-tuning trace CSV.", _run_recommend_trace),
    "plot-trace": ("Plot a recommended trace: cluster timeline, GPUs in use + jobs queued ([plot] extra).", _run_plot),
    "interactive": ("Guided keyboard-driven REPL over the recommender.", _run_interactive),
    "tune": ("Tune a data-driven predictor (tabpfn) on your own measured-runs CSV ([ml] extra).", _run_tune),
}


def _print_help() -> None:
    width = max(len(name) for name in _COMMANDS)
    print("usage: coastline <command> [options]\n\ncommands:")
    for name, (help_text, _) in _COMMANDS.items():
        print(f"  {name:<{width}}  {help_text}")
    print("\nRun `coastline <command> --help` for command-specific options.")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        _print_help()
        return
    if args[0] in ("-V", "--version"):
        print(f"coastline {__version__}")
        return
    command, rest = args[0], args[1:]
    if command == "enrich-trace":  # pre-rename spelling, kept working but not advertised
        print("note: `enrich-trace` is now `recommend-trace`", file=sys.stderr)
        command = "recommend-trace"
    entry = _COMMANDS.get(command)
    if entry is None:
        print(f"coastline: error: unknown command {command!r}\n", file=sys.stderr)
        _print_help()
        raise SystemExit(2)
    entry[1](rest)


if __name__ == "__main__":
    main()
