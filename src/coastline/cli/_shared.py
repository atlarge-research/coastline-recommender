"""Argument parsing and error reporting shared by the subcommands."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import NoReturn, Optional

from coastline.sdk.exceptions import RecommenderSystemError, UnsupportedGPUError


class FriendlyParser(argparse.ArgumentParser):
    """ArgumentParser that prints ``example`` after the usage on an error, then exits 2."""

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


def positive_int(text: str) -> int:
    """argparse ``type=`` for a count that must be at least 1."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def error_message(exc: BaseException) -> str:
    """One line describing ``exc`` for an ``error:`` line, without the traceback."""
    from pydantic import ValidationError

    if isinstance(exc, ValidationError):
        return "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'input'}: {error['msg']}" for error in exc.errors()
        )
    if isinstance(exc, FileNotFoundError) and exc.filename:
        return f"no such file: {exc.filename}"
    if isinstance(exc, IsADirectoryError) and exc.filename:
        return f"expected a file, got a directory: {exc.filename}"
    pandas = sys.modules.get("pandas")
    if pandas is not None and isinstance(exc, pandas.errors.EmptyDataError):
        return f"the input file is empty ({exc})"
    return str(exc) or type(exc).__name__


@contextmanager
def report_errors(parser: argparse.ArgumentParser) -> Iterator[None]:
    """Print an ``error:`` line instead of a traceback for the errors bad input raises.

    A missing file, a value the models reject or an unknown GPU is a usage error and exits 2.
    A run that finds nothing to recommend, or a predictor that cannot run, exits 1.
    """
    try:
        yield
    except (FileNotFoundError, IsADirectoryError, ValueError, UnsupportedGPUError) as exc:
        parser.error(error_message(exc))
    except (RuntimeError, RecommenderSystemError) as exc:
        parser.exit(1, f"{parser.prog}: error: {error_message(exc)}\n")
