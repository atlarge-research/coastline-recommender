"""``coastline utils`` — auxiliary tooling, offloaded from the core recommend verbs.

A thin sub-dispatcher over the utilities that support (but are not) the recommender:

* ``tune``          train a data-driven predictor on a measured-runs CSV ([ml] extra)
* ``trace-to-runs`` convert a fine-tuning trace CSV → the flat measured-runs schema
* ``plot-trace``    plot a recommended trace's cluster timeline ([plot] extra)
"""

from __future__ import annotations

import sys
from typing import Callable, Optional, Sequence

_Handler = Callable[[Optional[Sequence[str]]], None]


def _run_tune(argv: Optional[Sequence[str]]) -> None:
    from coastline.cli.tune import main

    main(argv)


def _run_trace_to_runs(argv: Optional[Sequence[str]]) -> None:
    from coastline.cli.trace_to_runs import main

    main(argv)


def _run_plot_trace(argv: Optional[Sequence[str]]) -> None:
    from coastline.cli.plot_trace import main

    main(argv)


_UTILS: dict[str, tuple[str, _Handler]] = {
    "tune": ("Tune a data-driven predictor on your own measured-runs CSV ([ml] extra).", _run_tune),
    "trace-to-runs": ("Convert a fine-tuning trace CSV to the flat measured-runs schema.", _run_trace_to_runs),
    "plot-trace": (
        "Plot a recommended trace: cluster timeline, GPUs in use + jobs queued ([plot] extra).",
        _run_plot_trace,
    ),
}


def _print_help() -> None:
    width = max(len(name) for name in _UTILS)
    print("usage: coastline utils <command> [options]\n\ncommands:")
    for name, (help_text, _) in _UTILS.items():
        print(f"  {name:<{width}}  {help_text}")
    print("\nRun `coastline utils <command> --help` for command-specific options.")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        _print_help()
        return
    command, rest = args[0], args[1:]
    entry = _UTILS.get(command)
    if entry is None:
        print(f"coastline utils: error: unknown command {command!r}\n", file=sys.stderr)
        _print_help()
        raise SystemExit(2)
    entry[1](rest)


if __name__ == "__main__":
    main()
