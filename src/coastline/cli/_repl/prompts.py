"""Keyboard-driven interactive widgets (arrow-nav + fuzzy selector); cancellable via Ctrl-C/Esc (raises Abort)."""

from __future__ import annotations

import os
import select
import sys
import termios
import tty
from collections.abc import Sequence
from dataclasses import dataclass

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from coastline.cli._repl.theme import console


class Abort(Exception):
    """Raised when the user cancels a prompt (Esc / Ctrl-C)."""


def read_key() -> str:
    """Return one logical keypress: a char or 'up'/'down'/'left'/'right'/'enter'/'backspace'/'esc'."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = os.read(fd, 1)
        if not ch:  # EOF — treat as cancel
            return "esc"
        if ch == b"\x1b":  # escape — arrow suffix only if actually pending
            if select.select([fd], [], [], 0.05)[0]:
                seq = os.read(fd, 2)
                return {b"[A": "up", b"[B": "down", b"[C": "right", b"[D": "left"}.get(seq, "esc")
            return "esc"
        if ch in (b"\r", b"\n"):
            return "enter"
        if ch == b"\x03":  # Ctrl-C
            raise Abort()
        if ch in (b"\x7f", b"\x08"):
            return "backspace"
        return ch.decode("utf-8", "ignore")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


@dataclass
class Choice:
    """A selectable option: value returned on select, label displayed, hint dimmed."""

    value: object
    label: str
    hint: str = ""


def _coerce(choices: Sequence[Choice | str]) -> list[Choice]:
    return [c if isinstance(c, Choice) else Choice(c, str(c)) for c in choices]


def menu(
    title: str,
    choices: Sequence[Choice | str],
    *,
    accent: str = "cyan",
    default: int = 0,
    footer: str = "↑↓ move · enter select · q quit",
) -> object:
    """Arrow-key menu; returns chosen Choice.value; 'q'/Esc raises Abort."""
    items = _coerce(choices)
    idx = max(0, min(default, len(items) - 1))

    def render() -> Panel:
        table = Table(box=None, pad_edge=False, show_header=False)
        table.add_column(width=3)
        table.add_column()
        table.add_column(justify="right", style="dim")
        for i, c in enumerate(items):
            sel = i == idx
            table.add_row(
                Text("❯" if sel else " ", style=f"bold {accent}"),
                Text(c.label, style=f"bold {accent}" if sel else "white"),
                Text(c.hint, style="dim"),
            )
        body = Group(table, Text(), Text(f"  {footer}", style="dim"))
        return Panel(body, title=f"[bold {accent}]{title}[/]", border_style=accent, padding=(1, 2))

    with Live(render(), console=console, auto_refresh=False, screen=False) as live:
        while True:
            key = read_key()
            if key in ("q", "esc"):
                raise Abort()
            if key == "up":
                idx = (idx - 1) % len(items)
            elif key == "down":
                idx = (idx + 1) % len(items)
            elif key == "enter":
                return items[idx].value
            live.update(render(), refresh=True)


def fuzzy_select(
    title: str,
    choices: Sequence[Choice | str],
    *,
    accent: str = "cyan",
    default: object | None = None,
    max_rows: int = 9,
) -> object:
    """Type-to-filter selector: substring match, arrows to move, enter to pick, Esc cancels."""
    items = _coerce(choices)
    query = ""
    idx = 0
    if default is not None:
        for i, c in enumerate(items):
            if c.value == default:
                idx = i
                break

    def filtered() -> list[Choice]:
        if not query:
            return items
        q = query.lower()
        return [c for c in items if q in c.label.lower()]

    def render() -> Panel:
        rows = filtered()
        nonlocal idx
        idx = max(0, min(idx, len(rows) - 1)) if rows else 0
        win_start = max(0, min(idx - max_rows // 2, max(0, len(rows) - max_rows)))
        view = rows[win_start : win_start + max_rows]

        table = Table(box=None, pad_edge=False, show_header=False)
        table.add_column(width=3)
        table.add_column()
        table.add_column(justify="right", style="dim")
        for i, c in enumerate(view, start=win_start):
            sel = i == idx
            table.add_row(
                Text("❯" if sel else " ", style=f"bold {accent}"),
                Text(c.label, style=f"bold {accent}" if sel else "white"),
                Text(c.hint, style="dim"),
            )
        if not rows:
            table.add_row(Text(" "), Text("(no match)", style="dim italic"), Text(""))

        qline = Text.assemble(
            ("  search ", "dim"),
            (query or "type to filter…", "white" if query else "dim italic"),
            ("▏", f"bold {accent}"),
        )
        count = Text(f"  {len(rows)}/{len(items)} · ↑↓ move · enter select · esc cancel", style="dim")
        body = Group(qline, Text(), table, Text(), count)
        return Panel(body, title=f"[bold {accent}]{title}[/]", border_style=accent, padding=(1, 2))

    with Live(render(), console=console, auto_refresh=False, screen=False) as live:
        while True:
            key = read_key()
            if key == "esc":
                raise Abort()
            rows = filtered()
            if key == "up":
                idx = (idx - 1) % len(rows) if rows else 0
            elif key == "down":
                idx = (idx + 1) % len(rows) if rows else 0
            elif key == "enter":
                if rows:
                    return rows[idx].value
            elif key == "backspace":
                query = query[:-1]
                idx = 0
            elif len(key) == 1 and key.isprintable():
                query += key
                idx = 0
            live.update(render(), refresh=True)


def text_prompt(message: str, *, default: str = "", accent: str = "cyan") -> str:
    """Single-line text entry with an editable default. Enter accepts, Esc cancels."""
    buf = default

    def render() -> Text:
        return Text.assemble(
            ("  ? ", f"bold {accent}"), (message + "  ", "white"), (buf, "bold white"), ("▏", f"bold {accent}")
        )

    with Live(render(), console=console, auto_refresh=False, screen=False) as live:
        while True:
            key = read_key()
            if key == "esc":
                raise Abort()
            if key == "enter":
                return buf
            if key == "backspace":
                buf = buf[:-1]
            elif len(key) == 1 and key.isprintable():
                buf += key
            live.update(render(), refresh=True)


def number_prompt(
    message: str,
    *,
    default: float | int,
    accent: str = "cyan",
    minimum: float | None = None,
    maximum: float | None = None,
    integer: bool = True,
) -> float | int:
    """Numeric entry with validation + re-prompt on bad input. Empty == default."""
    is_int = integer and isinstance(default, int)
    while True:
        raw = text_prompt(message, default=str(default), accent=accent).strip()
        if raw == "":
            return default
        try:
            val: float | int = int(raw) if is_int else float(raw)
        except ValueError:
            console.print(f"[red]  ✗ '{raw}' is not a valid number[/]")
            continue
        if minimum is not None and val < minimum:
            console.print(f"[red]  ✗ must be ≥ {minimum}[/]")
            continue
        if maximum is not None and val > maximum:
            console.print(f"[red]  ✗ must be ≤ {maximum}[/]")
            continue
        return val


def confirm(message: str, *, default: bool = True, accent: str = "cyan") -> bool:
    """Yes/No prompt. Enter takes the default; Esc cancels."""
    hint = "[Y/n]" if default else "[y/N]"
    line = Text.assemble(("  ? ", f"bold {accent}"), (f"{message} ", "white"), (hint, "dim"))
    console.print(line)
    key = read_key()
    if key == "esc":
        raise Abort()
    if key == "enter":
        return default
    return key.lower() == "y"
