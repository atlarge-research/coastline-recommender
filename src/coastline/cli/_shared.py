"""Shared CLI plumbing: a friendlier argparse parser used by every subcommand."""

from __future__ import annotations

import argparse
import sys
from typing import NoReturn, Optional


class FriendlyParser(argparse.ArgumentParser):
    """ArgumentParser that appends a worked example to error messages, then exits 2.

    Set ``example=`` to a one-line invocation; it is shown after the usage on error.
    """

    def __init__(self, *args: object, example: Optional[str] = None, **kwargs: object) -> None:
        self._example = example
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        prog = self.prog
        sys.stderr.write(f"{prog}: error: {message}\n")
        if self._example:
            sys.stderr.write(f"\nexample:\n  {self._example}\n")
        sys.exit(2)
