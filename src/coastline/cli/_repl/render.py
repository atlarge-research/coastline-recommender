"""Rich rendering for the recommender UI — spec cards, ranked table, recommendation panel."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional

from rich.columns import Columns
from rich.console import Group
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from coastline.cli._repl.theme import ACCENT, ENERGY, WINNER, console
from coastline.sdk.models.recommendation import Recommendation
from coastline.sdk.recommend.engine import runtime_energy


@contextmanager
def spinner(message: str, accent: str = ACCENT) -> Iterator[None]:
    with Progress(
        SpinnerColumn(style=accent),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as prog:
        prog.add_task(f"[{accent}]{message}", total=None)
        yield


def _kv(rows: list[tuple[str, str]]) -> Table:
    t = Table.grid(padding=(0, 2))
    t.add_column(justify="right", style="dim")
    t.add_column(style="white")
    for k, v in rows:
        t.add_row(k, v)
    return t


def _fmt_runtime(seconds: Optional[float]) -> str:
    if not seconds or seconds <= 0:
        return "—"
    if seconds < 90:
        return f"{seconds:.0f}s"
    minutes = seconds / 60.0
    return f"{minutes:.1f}m" if minutes < 90 else f"{minutes / 60.0:.1f}h"


def _fmt_energy(wh: Optional[float]) -> str:
    if not wh or wh <= 0:
        return "—"
    return f"{wh:,.0f} Wh" if wh < 1000 else f"{wh / 1000:,.2f} kWh"


def _model_card(model: str, accent: str) -> Panel:
    try:  # the physics library knows most fine-tuning models
        from kavier.sdk.library import get_llm

        llm = get_llm(model)
        body = _kv(
            [
                ("params", f"{llm.m_params / 1e9:,.1f} B"),
                ("active", f"{llm.active_params / 1e9:,.1f} B"),
                ("layers", str(llm.n_layers)),
                ("d_model", f"{llm.d_model:,}"),
                ("heads", str(llm.n_heads)),
            ]
        )
    except Exception:
        body = _kv([("spec", "[dim]not in physics library[/]")])
    return Panel(body, title=f"[bold]{model}[/]", border_style=accent, padding=(0, 2))


def _gpu_card(gpu: str, accent: str) -> Panel:
    try:
        from kavier.sdk.library import get_gpu

        g = get_gpu(gpu)
        body = _kv(
            [
                ("memory", f"{g.memory_gb:,.0f} GB"),
                ("FP16 TC", f"{g.fp_16_tensor_core_tflops:,} TFLOPs"),
                ("bandwidth", f"{g.bandwidth_bps / 1e9:,.0f} GB/s"),
                ("max power", f"{g.max_power_w:,.0f} W"),
            ]
        )
    except Exception:
        try:  # fall back to coastline's own hardware table (memory only)
            from coastline.sdk.library.hardware import get_gpu_memory

            body = _kv([("memory", f"{get_gpu_memory(gpu):,.0f} GB")])
        except Exception:
            body = _kv([("spec", "[dim]unknown[/]")])
    return Panel(body, title=f"[bold]{gpu}[/]", border_style=accent, padding=(0, 2))


def specs_panel(model: str, gpu: str, accent: str = ACCENT) -> Panel:
    """Side-by-side model + GPU spec cards."""
    return Panel(
        Columns([_model_card(model, accent), _gpu_card(gpu, accent)], equal=True, expand=True),
        title="[bold]selected specs[/]",
        border_style="dim",
        padding=(0, 1),
    )


def workload_panel(answers: dict[str, Any], accent: str = ACCENT) -> Panel:
    epochs = answers.get("epochs", 1)
    dataset = f"{answers.get('dataset_size', 0):,} samples · {epochs:g} epoch{'s' if epochs != 1 else ''}"
    body = _kv(
        [
            ("method", answers["fine_tuning_method"]),
            ("tokens/sample", str(answers["tokens_per_sample"])),
            ("batch size", str(answers["batch_size"])),
            ("dataset", dataset),
            ("optimise for", answers["goal_label"]),
        ]
    )
    return Panel(body, title="[bold]workload[/]", border_style="dim", padding=(1, 2))


def ranked_table(recs: list[Recommendation], total_tokens: int, accent: str = ACCENT) -> Table:
    t = Table(
        title="[bold]ranked configurations[/]",
        title_justify="left",
        header_style=f"bold {accent}",
        border_style=accent,
        expand=True,
    )
    t.add_column(" ", justify="center", style="dim", no_wrap=True)
    t.add_column("config", no_wrap=True)
    t.add_column("gpus", justify="right", no_wrap=True)
    t.add_column("batch", justify="right", no_wrap=True)
    t.add_column("tok/s", justify="right", style="green")
    t.add_column("runtime", justify="right")
    t.add_column("energy", justify="right", style=ENERGY)
    for i, r in enumerate(recs, start=1):
        best = i == 1
        runtime, energy = runtime_energy(r, total_tokens)
        t.add_row(
            "★" if best else f"[dim]{i}[/]",
            f"{r.gpus_per_node}×{r.number_of_nodes}",
            str(r.total_gpus),
            str((r.metadata or {}).get("batch_size", "—")),
            f"{r.predicted_throughput:,.0f}" if r.predicted_throughput else "—",
            _fmt_runtime(runtime),
            _fmt_energy(energy),
            style=f"bold {WINNER}" if best else None,
        )
    return t


def recommendation_panel(rec: Recommendation, meta: dict[str, Any]) -> Panel:
    """Render the chosen configuration as a green hero card."""
    runtime, energy = runtime_energy(rec, meta.get("total_tokens", 0))
    plural = "s" if rec.total_gpus != 1 else ""
    config = _kv(
        [
            ("layout", f"{rec.gpus_per_node}×{rec.number_of_nodes}"),
            ("batch", str((rec.metadata or {}).get("batch_size", "—"))),
            ("predictor", str(meta.get("predictor", "—"))),
        ]
    )
    metrics = _kv(
        [
            ("throughput", f"[green]{rec.predicted_throughput:,.0f}[/] tok/s" if rec.predicted_throughput else "—"),
            ("runtime", _fmt_runtime(runtime)),
            ("energy", f"[{ENERGY}]{_fmt_energy(energy)}[/]"),
        ]
    )
    headline = Text.from_markup(f"[bold green]{rec.total_gpus} GPU{plural}[/]   [green]·   full run on your dataset[/]")
    body = Group(headline, Text(""), Columns([config, metrics], expand=True))
    return Panel(
        body, title="[bold green]✓ recommendation[/]", title_align="left", border_style="green", padding=(1, 2)
    )
