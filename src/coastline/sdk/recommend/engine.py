"""Pure recommend logic for the interactive UI — no Rich, no prompts; shared by REPL and non-interactive path."""

from __future__ import annotations

import time
from typing import Any, Optional

from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.recommendation import Recommendation
from coastline.sdk.models.workload import WorkloadSpec

try:
    from coastline.sdk.io.options_loader import load_available_options
except Exception:  # pragma: no cover - defensive: loader should always import
    load_available_options = None  # type: ignore[assignment]


# Static fallbacks: used when the curated dataset is unavailable, and for knobs
# that are not part of the dataset-derived options.
FALLBACK_MODELS = [
    "granite-3.1-3b-a800m-instruct",
    "granite-3.3-8b",
    "mistral-7b-v0.1",
    "mixtral-8x7b-instruct-v0.1",
]
FALLBACK_METHODS = ["full", "lora", "gptq-lora", "qlora"]
FALLBACK_GPUS = ["NVIDIA-A100-SXM4-80GB", "NVIDIA-A100-80GB-PCIe", "L40S"]
FALLBACK_TOKENS = [512, 1024, 2048, 4096, 8192]
FALLBACK_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
GPU_BUDGETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)

# Optimisation goal -> (strategy_name, preset).
GOALS: dict[str, tuple[str, Optional[str]]] = {
    "Multi-objective balanced": ("multi_objective", "balanced"),
    "Multi-objective lowest runtime": ("multi_objective", "performance"),
    "Multi-objective energy-saver": ("multi_objective", "energy"),
    "Fewest GPUs that fit": ("min_gpu", None),
}

# Top-level performance-predictor choices for the UI. "ml" is a sentinel that
# opens the trained-ML submenu (ML_MODELS); the rest are engine predictor keys.
PREDICTOR_CHOICES: list[tuple[str, str]] = [
    ("intelligent", "intelligent  ·  exact cache match, else Kavier physics"),
    ("kavier", "physics simulator  ·  Kavier"),
    ("ml", "trained ML model  ·  you pick"),
    ("cache", "exact match  ·  measured past runs only"),
]

# The data-driven models, surfaced as a submenu under "trained ML model".
ML_MODELS: list[tuple[str, str]] = [
    ("catboost", "CatBoost"),
    ("xgboost", "XGBoost"),
    ("lightgbm", "LightGBM"),
    ("random_forest", "Random Forest"),
    ("gaussian_process", "Gaussian Process"),
    ("bayesian_ridge", "Bayesian Ridge"),
    ("knn", "k-Nearest Neighbours"),
    ("svr", "Support Vector Regression"),
    ("tabpfn", "TabPFN"),
    ("deep_learning", "Deep Learning (MLP)"),
]


def resolve_options() -> dict[str, list]:
    """Option lists for the prompts, from the curated dataset when available."""
    opts: dict[str, list] = {
        "models": list(FALLBACK_MODELS),
        "methods": list(FALLBACK_METHODS),
        "gpus": list(FALLBACK_GPUS),
        "tokens_per_sample": list(FALLBACK_TOKENS),
        "batch_sizes": list(FALLBACK_BATCH_SIZES),
    }
    if load_available_options is None:
        return opts
    try:
        loaded = load_available_options()
        for key in opts:
            if loaded.get(key):
                opts[key] = list(loaded[key])
    except Exception:  # pragma: no cover - defensive
        pass
    return opts


def defaults(opts: dict[str, list]) -> dict[str, Any]:
    """Non-interactive defaults (for the scripted / no-TTY path)."""

    def pick(values: list, preferred: list, fallback: Any) -> Any:
        for p in preferred:
            if p in values:
                return p
        return values[0] if values else fallback

    return {
        "llm_model": pick(opts["models"], ["mistral-7b-v0.1", "granite-3.3-8b"], "mistral-7b-v0.1"),
        "fine_tuning_method": pick(opts["methods"], ["lora", "full"], "lora"),
        "gpu_model": pick(opts["gpus"], ["NVIDIA-A100-SXM4-80GB"], "NVIDIA-A100-SXM4-80GB"),
        "tokens_per_sample": pick(opts["tokens_per_sample"], [1024, 2048], 1024),
        "batch_size": pick(opts["batch_sizes"], [16, 8, 32], 16),
        "dataset_size": 50_000,
        "epochs": 1,
        "max_gpus": 16,
        "goal_label": "Multi-objective balanced",
        "predictor": "intelligent",
    }


def build_config(
    answers: dict[str, Any],
    top_k: int,
    max_slowdown: Optional[float] = None,
    feasibility: str = "autoconf",
) -> tuple[dict, str, Optional[str]]:
    """Build strategy-config dict for PolicyFactory; max_slowdown maps to runtime_guard_k.

    ``feasibility`` selects the checker (``autoconf`` | ``rules`` | ``none``); the
    answers dict may override it via a ``feasibility`` key.
    """
    strategy_name, preset = GOALS[answers["goal_label"]]
    predictor = answers["predictor"]
    strategy: dict[str, Any] = {"name": strategy_name, "preset": preset or "balanced"}
    if max_slowdown is not None:
        strategy["runtime_guard_k"] = float(max_slowdown)
    predictors: dict[str, Any] = {
        "performance": predictor,
        "energy": "kavier_power",
        "feasibility": answers.get("feasibility", feasibility),
    }
    if answers.get("lookup"):
        predictors["lookup"] = str(answers["lookup"])  # measured-runs CSV for cache/intelligent
    config: dict[str, Any] = {
        "strategy": strategy,
        "predictors": predictors,
        "grid": {
            # The chosen batch size plus its neighbours, so the ranked table
            # shows real trade-offs rather than a single row.
            "batch_sizes": sorted(
                {answers["batch_size"], max(1, answers["batch_size"] // 2), answers["batch_size"] * 2}
            ),
            "total_gpus": [g for g in GPU_BUDGETS if g <= answers["max_gpus"]],
            "top_k": top_k,
        },
    }
    return config, strategy_name, preset


def build_workload(answers: dict[str, Any]) -> WorkloadSpec:
    return WorkloadSpec(
        llm_model=answers["llm_model"],
        fine_tuning_method=answers["fine_tuning_method"],
        gpu_model=answers["gpu_model"],
        tokens_per_sample=int(answers["tokens_per_sample"]),
        batch_size=int(answers["batch_size"]),
        gpus_per_node=min(8, answers["max_gpus"]),
        number_of_nodes=1,
    )


def build_context(answers: dict[str, Any]) -> SystemContext:
    max_gpus = int(answers["max_gpus"])
    return SystemContext.for_gpus([answers["gpu_model"]], max_gpus=max_gpus, gpus_per_node=min(8, max_gpus))


def run_pipeline(
    answers: dict[str, Any],
    top_k: int,
    max_slowdown: Optional[float] = None,
    feasibility: str = "autoconf",
) -> tuple[list[Recommendation], dict[str, Any]]:
    """Run PolicyFactory + strategy.recommend; return (recs, meta).

    ``feasibility`` (``autoconf`` | ``rules`` | ``none``) picks the feasibility
    checker; an answers ``feasibility`` key takes precedence (see ``build_config``).
    """
    from coastline.sdk.policies import PolicyFactory

    config, strategy_name, preset = build_config(answers, top_k, max_slowdown, feasibility)
    workload = build_workload(answers)
    context = build_context(answers)

    t0 = time.perf_counter()
    strategy = PolicyFactory.create_strategy(strategy_name=strategy_name, preset=preset, config=config)
    recs = strategy.recommend(workload, context)
    elapsed = time.perf_counter() - t0

    total_tokens = int(answers["dataset_size"] * answers["epochs"] * answers["tokens_per_sample"])
    meta = {
        "strategy_name": strategy_name,
        "preset": preset,
        "predictor": config["predictors"]["performance"],
        "elapsed_s": elapsed,
        "grid": config["grid"],
        "workload": workload,
        "total_tokens": total_tokens,
    }
    return recs, meta


def runtime_energy(rec: Recommendation, total_tokens: int) -> tuple[Optional[float], Optional[float]]:
    """Return (runtime_s, energy_wh) for the user's dataset."""
    thr = rec.predicted_throughput or 0.0
    power = (rec.metadata or {}).get("predicted_power_watts") or 0.0
    runtime = total_tokens / thr if (thr > 0 and total_tokens > 0) else rec.predicted_runtime_seconds
    energy = (power * rec.total_gpus * runtime) / 3600.0 if (runtime and power) else None
    return runtime, energy


# Goal -> a short phrase for the recommendation rationale.
_GOAL_RATIONALE = {
    "balanced": "the best throughput-vs-energy balance",
    "performance": "the highest throughput",
    "energy": "the lowest energy",
    "min_gpu": "the fewest GPUs that fit",
}


def recommendation_rationale(recs: list[Recommendation], meta: dict[str, Any]) -> str:
    """One-line rationale for the top recommendation vs runner-up."""
    if not recs:
        return "No feasible configuration in the search space."
    top = recs[0]
    goal = (
        _GOAL_RATIONALE.get(meta.get("preset"))
        or _GOAL_RATIONALE.get(meta.get("strategy_name"))
        or "the best throughput/energy trade-off"
    )
    plural = "s" if top.total_gpus != 1 else ""
    top_batch = (top.metadata or {}).get("batch_size")
    config = f"{top.gpus_per_node}×{top.number_of_nodes}" + (f", batch {top_batch}" if top_batch else "")
    line = f"{top.total_gpus} GPU{plural} ({config}) picked for {goal}"
    if len(recs) > 1 and top.predicted_throughput and recs[1].predicted_throughput:
        runner = recs[1]
        gap = (top.predicted_throughput - runner.predicted_throughput) / runner.predicted_throughput * 100.0
        if abs(gap) >= 0.5:
            direction = "faster" if gap > 0 else "slower"
            rb = (runner.metadata or {}).get("batch_size")
            rplural = "s" if runner.total_gpus != 1 else ""
            runner_desc = f"{runner.total_gpus} GPU{rplural}" + (f", batch {rb}" if rb else "")
            line += f", {abs(gap):.0f}% {direction} than the runner-up ({runner_desc})"
    return line + "."
