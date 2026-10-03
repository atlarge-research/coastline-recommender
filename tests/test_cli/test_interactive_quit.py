"""The guided REPL can be left from any prompt: `q` in a menu and Ctrl-C quit, Esc goes back.

Raw mode turns Ctrl-C into a key, so the prompts raise `Quit` for it (a KeyboardInterrupt, which
the command already answers with 'bye'); Esc raises `Abort`, which restarts the flow, also at the
'Next' menu after a run. The command quiets the engine logs while it runs and hands the logging
state back when it ends. Keys are fed through a stand-in for `read_key`; one test drives the real
`read_key` through a pseudo-terminal.
"""

from __future__ import annotations

import logging
import os
import sys

import pytest
import typer

from coastline.cli import interactive
from coastline.cli._repl import prompts
from coastline.cli._repl.prompts import Abort, Quit


class _OutOfKeys(BaseException):
    """The REPL asked for a key after the test's last one (it kept prompting)."""


def _feed(monkeypatch, keys: list[str]) -> list[str]:
    """Make the prompts read ``keys`` in order; returns the list of keys still unread."""
    remaining = list(keys)

    def read_key() -> str:
        if not remaining:
            raise _OutOfKeys("the REPL kept prompting after the last key")
        key = remaining.pop(0)
        if key == "ctrl-c":
            raise Quit()
        return key

    monkeypatch.setattr(prompts, "read_key", read_key)
    return remaining


@pytest.fixture
def a_terminal(monkeypatch):
    """Let interactive.main take the REPL branch: stdin reports a terminal."""

    class _Tty:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(interactive.sys, "stdin", _Tty())


def _run_repl() -> None:
    interactive.main(interactive=True, top_k=3, save=None, verbose=False)


# Enter at every prompt of one workload: model, GPU, method, tokens, batch size, dataset size,
# epochs, max GPUs, goal and predictor.
_DEFAULT_ANSWERS = ["enter"] * 10


@pytest.fixture
def logging_off_below_info():
    """A logging state the command must hand back unchanged: DEBUG records off."""
    previous = logging.root.manager.disable
    logging.disable(logging.DEBUG)
    yield logging.DEBUG
    logging.disable(previous)


@pytest.mark.parametrize(("key", "error"), [("q", Quit), ("esc", Abort)])
def test_q_in_a_menu_quits_and_esc_cancels(monkeypatch, key, error) -> None:
    _feed(monkeypatch, [key])

    with pytest.raises(error):
        prompts.menu("pick", [prompts.Choice("a", "a"), prompts.Choice("b", "b")])


def test_quit_is_a_keyboard_interrupt() -> None:
    """So a caller that already handles Ctrl-C (and `except Exception` does not) handles it too."""
    assert issubclass(Quit, KeyboardInterrupt)
    assert not issubclass(Quit, Exception)


def test_q_at_the_method_menu_leaves_the_repl(monkeypatch, capsys, a_terminal) -> None:
    # Model and GPU take their defaults, then 'q' at the fine-tuning method menu.
    remaining = _feed(monkeypatch, ["enter", "enter", "q"])

    with pytest.raises(typer.Exit) as excinfo:
        _run_repl()

    assert excinfo.value.exit_code == 0
    assert remaining == []
    out = capsys.readouterr().out
    assert "bye" in out
    assert "back to start" not in out


def test_ctrl_c_at_the_first_prompt_leaves_the_repl(monkeypatch, capsys, a_terminal) -> None:
    _feed(monkeypatch, ["ctrl-c"])

    with pytest.raises(typer.Exit) as excinfo:
        _run_repl()

    assert excinfo.value.exit_code == 0
    assert "bye" in capsys.readouterr().out


def test_esc_goes_back_to_the_start_and_q_still_quits(monkeypatch, capsys, a_terminal) -> None:
    remaining = _feed(monkeypatch, ["esc", "enter", "enter", "q"])

    with pytest.raises(typer.Exit):
        _run_repl()

    assert remaining == []
    out = capsys.readouterr().out
    assert out.count("back to start") == 1
    assert "bye" in out


def test_esc_at_the_next_menu_goes_back_to_the_start(monkeypatch, capsys, a_terminal) -> None:
    runs: list[dict] = []

    def run_and_show(answers, top_k):
        runs.append(answers)
        return [], {}

    monkeypatch.setattr(interactive, "_run_and_show", run_and_show)
    # One workload, Esc at the 'Next' menu, then Ctrl-C at the model prompt of the fresh start.
    remaining = _feed(monkeypatch, [*_DEFAULT_ANSWERS, "esc", "ctrl-c"])

    with pytest.raises(typer.Exit):
        _run_repl()

    assert remaining == []
    assert len(runs) == 1
    out = capsys.readouterr().out
    assert out.count("back to start") == 1
    assert "bye" in out


def test_q_at_the_next_menu_still_quits(monkeypatch, capsys, a_terminal) -> None:
    monkeypatch.setattr(interactive, "_run_and_show", lambda answers, top_k: ([], {}))
    remaining = _feed(monkeypatch, [*_DEFAULT_ANSWERS, "q"])

    with pytest.raises(typer.Exit) as excinfo:
        _run_repl()

    assert excinfo.value.exit_code == 0
    assert remaining == []
    out = capsys.readouterr().out
    assert "bye" in out
    assert "back to start" not in out


def test_leaving_the_repl_restores_the_logging_state(monkeypatch, a_terminal, logging_off_below_info) -> None:
    _feed(monkeypatch, ["ctrl-c"])

    with pytest.raises(typer.Exit):
        _run_repl()

    assert logging.root.manager.disable == logging_off_below_info


def test_a_one_shot_run_restores_the_logging_state(monkeypatch, logging_off_below_info) -> None:
    seen: list[int] = []
    monkeypatch.setattr(
        interactive, "_run_noninteractive", lambda top_k, save: seen.append(logging.root.manager.disable)
    )

    interactive.main(interactive=False, top_k=3, save=None, verbose=False)

    assert seen == [logging.WARNING]  # quiet while it runs
    assert logging.root.manager.disable == logging_off_below_info


@pytest.mark.skipif(sys.platform == "win32", reason="raw terminal keys need a POSIX pseudo-terminal")
def test_read_key_turns_ctrl_c_into_quit(monkeypatch) -> None:
    import pty
    import tty
    from types import SimpleNamespace

    leader, follower = pty.openpty()
    # Raw before the keys arrive: a cooked terminal turns the Ctrl-C byte into a signal instead
    # of input. read_key's own setraw would then flush the queued key (TCSAFLUSH), so it is a
    # no-op here; the terminal is already raw.
    tty.setraw(follower)
    monkeypatch.setattr(prompts, "tty", SimpleNamespace(setraw=lambda fd: None))

    class _Stdin:
        def fileno(self) -> int:
            return follower

    monkeypatch.setattr(prompts.sys, "stdin", _Stdin())
    try:
        os.write(leader, b"\x03")
        with pytest.raises(Quit):
            prompts.read_key()
        os.write(leader, b"\x1b")
        assert prompts.read_key() == "esc"
    finally:
        os.close(leader)
        os.close(follower)
