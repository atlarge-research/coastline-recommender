"""COASTLINE interactive CLI — guided REPL over the GPU-configuration recommender."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import typer
from rich.panel import Panel

from coastline.cli._repl import render
from coastline.cli._repl.prompts import Abort, Choice, fuzzy_select, menu, number_prompt, text_prompt
from coastline.cli._repl.theme import ACCENT, banner, console
from coastline.sdk.recommend import engine

app = typer.Typer(add_completion=False, help="Interactive COASTLINE recommender — guided GPU-configuration advisor.")


def _index(values: list, target: Any) -> int:
    try:
        return list(values).index(target)
    except ValueError:
        return 0


def _spec_hint_model(name: str) -> str:
    try:
        from kavier.sdk.library import get_llm

        return f"{get_llm(name).m_params / 1e9:,.0f}B"
    except Exception:
        return ""


def _spec_hint_gpu(name: str) -> str:
    try:
        from kavier.sdk.library import get_gpu

        return f"{get_gpu(name).memory_gb:,.0f}GB"
    except Exception:
        return ""


def _recommend_inputs(seed: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Guided prompt flow → a plain answers dict (re-seeded by 'tweak')."""
    s = seed or {}
    opts = engine.resolve_options()
    d = engine.defaults(opts)

    model = str(
        fuzzy_select(
            "Model",
            [Choice(m, m, _spec_hint_model(m)) for m in opts["models"]],
            default=s.get("llm_model", d["llm_model"]),
        )
    )
    gpu = str(
        fuzzy_select(
            "GPU",
            [Choice(g, g, _spec_hint_gpu(g)) for g in opts["gpus"]],
            default=s.get("gpu_model", d["gpu_model"]),
        )
    )
    console.print(render.specs_panel(model, gpu, ACCENT))

    method = str(
        menu(
            "Fine-tuning method",
            [Choice(m, m) for m in opts["methods"]],
            default=_index(opts["methods"], s.get("fine_tuning_method", d["fine_tuning_method"])),
        )
    )
    tokens = menu(
        "Tokens per sample (sequence length)",
        [Choice(t, str(t)) for t in opts["tokens_per_sample"]],
        default=_index(opts["tokens_per_sample"], s.get("tokens_per_sample", d["tokens_per_sample"])),
    )
    batch = number_prompt("Batch size:", default=int(s.get("batch_size", d["batch_size"])), minimum=1)
    dataset_size = number_prompt(
        "Dataset size (training samples):", default=int(s.get("dataset_size", 50_000)), minimum=1
    )
    epochs = number_prompt("Epochs:", default=float(s.get("epochs", 1)), minimum=0, integer=False)
    max_gpus = number_prompt("Max GPUs to consider:", default=int(s.get("max_gpus", 8)), minimum=1)
    goal = str(
        menu(
            "Optimise for",
            [Choice(g, g) for g in engine.GOALS],
            default=_index(list(engine.GOALS), s.get("goal_label", d["goal_label"])),
        )
    )
    top_keys = [k for k, _ in engine.PREDICTOR_CHOICES]
    seed_pred = s.get("predictor", d["predictor"])
    if seed_pred in top_keys:
        top_default = seed_pred
    elif seed_pred in dict(engine.ML_MODELS):
        top_default = "ml"  # a trained-ML model seed opens on the ML submenu
    else:
        top_default = "intelligent"
    predictor = str(
        menu(
            "Performance predictor",
            [Choice(k, label) for k, label in engine.PREDICTOR_CHOICES],
            default=_index(top_keys, top_default),
        )
    )
    if predictor == "ml":  # "trained ML model · you pick" → open a second list
        ml_keys = [k for k, _ in engine.ML_MODELS]
        predictor = str(
            menu(
                "Trained ML model",
                [Choice(k, label) for k, label in engine.ML_MODELS],
                default=_index(ml_keys, seed_pred if seed_pred in ml_keys else "catboost"),
            )
        )
    return {
        "llm_model": model,
        "fine_tuning_method": method,
        "gpu_model": gpu,
        "tokens_per_sample": int(tokens),
        "batch_size": int(batch),
        "dataset_size": int(dataset_size),
        "epochs": float(epochs),
        "max_gpus": int(max_gpus),
        "goal_label": goal,
        "predictor": predictor,
    }


def _run_and_show(answers: dict[str, Any], top_k: int) -> tuple[list, dict[str, Any]]:
    with render.spinner("Predicting throughput & energy, ranking configurations…"):
        recs, meta = engine.run_pipeline(answers, top_k)
    console.print()
    console.print(render.workload_panel(answers))
    if not recs:
        console.print(
            Panel(
                "No feasible configuration in the search space — raise 'Max GPUs' or lower the batch size.",
                title="[bold yellow]no recommendations[/]",
                border_style="yellow",
                padding=(1, 2),
            )
        )
        return recs, meta
    console.print()
    console.print(render.ranked_table(recs, meta["total_tokens"]))
    console.print()
    console.print(render.recommendation_panel(recs[0], meta))
    console.print(f"[dim]  why: {engine.recommendation_rationale(recs, meta)}[/]")
    console.print(
        f"[dim]  {len(recs)} options ranked in {meta['elapsed_s']:.2f}s · energy = full run on your dataset[/]\n"
    )
    return recs, meta


def _save_to(rec: Any, path: Path, rationale: Optional[str] = None) -> None:
    try:
        from coastline.sdk.io.interface.json_output import save_recommendation_to_json

        path.parent.mkdir(parents=True, exist_ok=True)
        save_recommendation_to_json(rec, path, rationale=rationale)
        console.print(f"[green]  ✓ saved[/] [cyan]{path}[/]")
    except Exception as exc:  # noqa: BLE001 — surface any IO/serializer error gracefully
        console.print(f"[red]  ✗ could not save: {exc}[/]")


def _save_top(recs: list, rationale: Optional[str] = None) -> None:
    if not recs:
        console.print("[yellow]  nothing to save.[/]")
        return
    try:
        path = text_prompt("Save to:", default="recommendation.json")
    except Abort:
        return
    _save_to(recs[0], Path(path), rationale)


def _followups(answers: dict[str, Any], recs: list, meta: dict[str, Any], top_k: int) -> str:
    """Loop after a run. Returns 'new' (fresh workload) or 'quit'."""
    while True:
        try:
            action = str(
                menu(
                    "Next",
                    [
                        Choice("objective", "Change objective & re-rank", "balanced / runtime / energy / fewest"),
                        Choice("tweak", "Tweak inputs & re-run", "edit the workload"),
                        Choice("save", "Save top recommendation", "write JSON"),
                        Choice("new", "New workload", ""),
                        Choice("quit", "Quit", ""),
                    ],
                )
            )
        except Abort:
            return "quit"
        if action in ("new", "quit"):
            return action
        if action == "tweak":
            answers = _recommend_inputs(seed=answers)
            recs, meta = _run_and_show(answers, top_k)
        elif action == "objective":
            goal = str(
                menu(
                    "Optimise for",
                    [Choice(g, g) for g in engine.GOALS],
                    default=_index(list(engine.GOALS), answers["goal_label"]),
                )
            )
            answers = {**answers, "goal_label": goal}
            recs, meta = _run_and_show(answers, top_k)
        elif action == "save":
            _save_top(recs, engine.recommendation_rationale(recs, meta))


def _repl(top_k: int) -> None:
    seed: Optional[dict[str, Any]] = None
    while True:
        try:
            answers = _recommend_inputs(seed)
            recs, meta = _run_and_show(answers, top_k)
            action = _followups(answers, recs, meta, top_k)
        except Abort:
            console.print("[dim]  cancelled — back to start[/]")
            seed = None
            continue
        except Exception as exc:  # noqa: BLE001 — never let one bad run kill the REPL
            console.print(f"[red]  ✗ {exc}[/]")
            seed = None
            continue
        if action == "quit":
            break
        seed = None  # "new" → fresh workload
    console.print("\n[cyan]  thanks for using Coastline 👋[/]\n")


def _run_noninteractive(top_k: int, save: Optional[Path]) -> None:
    answers = engine.defaults(engine.resolve_options())
    console.print("[dim]  non-interactive defaults[/]")
    recs, meta = _run_and_show(answers, top_k)
    if save and recs:
        _save_to(recs[0], save, engine.recommendation_rationale(recs, meta))


@app.command()
def main(
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive",
        help="Run the guided REPL (default) or a one-shot run with defaults.",
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", min=1, max=20, help="How many configurations to rank."),
    save: Optional[Path] = typer.Option(
        None, "--save", help="Write the top recommendation to this JSON file.", dir_okay=False, writable=True
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show engine INFO/WARNING logs."),
) -> None:
    """Guided, interactive GPU-configuration recommender for LLM fine-tuning."""
    if not verbose:
        logging.disable(logging.WARNING)
    # Raw-key prompts need a real terminal; fall back to a one-shot defaults run
    # when piped / redirected (CI, scripts, `< /dev/null`).
    if interactive and not sys.stdin.isatty():
        interactive = False
        console.print(
            "[yellow]stdin is not a terminal; using non-interactive defaults (--no-interactive to silence).[/]"
        )
    console.print(banner())
    try:
        if interactive:
            _repl(top_k)
        else:
            _run_noninteractive(top_k, save)
    except (KeyboardInterrupt, Abort):
        console.print("\n[cyan]  bye 👋[/]\n")
        raise typer.Exit(code=0)


def run(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for ``coastline interactive`` (dispatched from the unified CLI)."""
    app(args=list(argv) if argv is not None else [], prog_name="coastline interactive")


if __name__ == "__main__":
    run()
