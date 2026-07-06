"""Shared console, palette and banner for the COASTLINE interactive UI."""

from __future__ import annotations

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

# One accent for the advisor; green is reserved for the chosen recommendation,
# yellow for energy — matching the per-domain "channel" idea in kavier_ui.
ACCENT = "cyan"
WINNER = "green"
ENERGY = "yellow"

_LOGO = r"""
  ____                _   _ _
 / ___|___   __ _ ___| |_| (_)_ __   ___
| |   / _ \ / _` / __| __| | | '_ \ / _ \
| |__| (_) | (_| \__ \ |_| | | | | |  __/
 \____\___/ \__,_|___/\__|_|_|_| |_|\___|
""".strip("\n")


def banner() -> Panel:
    logo = Text(_LOGO, style="bold cyan")
    sub = Text("GPU configuration advisor for LLM fine-tuning", style="dim")
    tag = Text("throughput · runtime · energy, ranked", style="cyan")
    body = Align.center(Text("\n").join([logo, Text(), sub, tag]))
    return Panel(body, border_style="cyan", padding=(1, 4), title="[bold]interactive[/]", title_align="right")


def rule(text: str, accent: str = ACCENT) -> Text:
    return Text.assemble(("  ", ""), (text, f"bold {accent}"))
