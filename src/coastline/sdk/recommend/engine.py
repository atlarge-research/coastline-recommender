"""Pure recommend logic for the interactive UI — no Rich, no prompts; shared by REPL and non-interactive path."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.recommendation import Recommendation
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.recommend import _goals

if TYPE_CHECKING:  # avoid importing the heavy policies package at module load
    from coastline.sdk.policies.base import BaseStrategy

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

# Optimisation goal (display label) -> (strategy_name, preset). Derived from the single
# objective vocabulary in `_goals`; the REPL enumerates these keys as menu choices.
GOALS: dict[str, tuple[str, Optional[str]]] = _goals.engine_goals()

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


@dataclass
class RecommendRequest:
    """The one shape the engine consumes. Every door (facade, batch CSV, config-driven
    ``run``, UI, and the answers-driven ``run_pipeline``) builds one of these its own way,
    then hands it to :func:`run_request`. Input-building and output-serialization stay in
    the caller; only the strategy-create → recommend core lives here."""

    workload: WorkloadSpec
    context: SystemContext
    config: dict[str, Any]  # fully-formed PolicyFactory config: strategy / predictors / grid
    strategy_name: str
    preset: Optional[str] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    total_tokens: int = 0  # for runtime/energy meta; 0 == "not applicable" (facade, run.py)


def build_strategy(
    config: dict[str, Any],
    strategy_name: str,
    preset: Optional[str] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
) -> "BaseStrategy":
    """The ONE place ``PolicyFactory.create_strategy`` is called. Split out from
    :func:`execute_strategy` so a caller (``batch_csv``) can build the strategy once and
    reuse it across many rows — predictors + the AutoConf feasibility model load a single
    time instead of per row."""
    from coastline.sdk.policies import PolicyFactory

    return PolicyFactory.create_strategy(
        strategy_name=strategy_name, preset=preset, alpha=alpha, beta=beta, config=config
    )


def execute_strategy(
    strategy: "BaseStrategy",
    workload: WorkloadSpec,
    context: SystemContext,
    *,
    strategy_name: str,
    preset: Optional[str],
    grid: dict[str, Any],
    predictor: Optional[str],
    total_tokens: int = 0,
) -> tuple[list[Recommendation], dict[str, Any]]:
    """Run ``strategy.recommend``, time it, normalize ``None``/single/list → list, build meta.
    Accepts a pre-built strategy so it can be called repeatedly with the same one."""
    t0 = time.perf_counter()
    recs = strategy.recommend(workload, context)
    elapsed = time.perf_counter() - t0

    if recs is None:
        recs = []
    elif isinstance(recs, Recommendation):
        recs = [recs]
    else:
        recs = list(recs)

    meta = {
        "strategy_name": strategy_name,
        "preset": preset,
        "predictor": predictor,
        "elapsed_s": elapsed,
        "grid": grid,
        "workload": workload,
        "total_tokens": total_tokens,
    }
    return recs, meta


def run_request(request: RecommendRequest) -> tuple[list[Recommendation], dict[str, Any]]:
    """The single workflow: build the strategy, run it, return (recs, meta)."""
    strategy = build_strategy(
        request.config, request.strategy_name, request.preset, request.alpha, request.beta
    )
    return execute_strategy(
        strategy,
        request.workload,
        request.context,
        strategy_name=request.strategy_name,
        preset=request.preset,
        grid=request.config.get("grid", {}),
        predictor=(request.config.get("predictors") or {}).get("performance"),
        total_tokens=request.total_tokens,
    )


def run_pipeline(
    answers: dict[str, Any],
    top_k: int,
    max_slowdown: Optional[float] = None,
    feasibility: str = "autoconf",
) -> tuple[list[Recommendation], dict[str, Any]]:
    """Answers-driven entry (interactive REPL, no-TTY path, and ``batch_api``): derive a
    ``RecommendRequest`` from an ``answers`` dict and run it. Signature and return are
    unchanged — this is a thin wrapper over the shared :func:`run_request` seam.

    ``feasibility`` (``autoconf`` | ``rules`` | ``none``) picks the feasibility
    checker; an answers ``feasibility`` key takes precedence (see ``build_config``).
    """
    config, strategy_name, preset = build_config(answers, top_k, max_slowdown, feasibility)
    total_tokens = int(answers["dataset_size"] * answers["epochs"] * answers["tokens_per_sample"])
    return run_request(
        RecommendRequest(
            workload=build_workload(answers),
            context=build_context(answers),
            config=config,
            strategy_name=strategy_name,
            preset=preset,
            total_tokens=total_tokens,
        )
    )


def runtime_energy(rec: Recommendation, total_tokens: int) -> tuple[Optional[float], Optional[float]]:
    """Return (runtime_s, energy_wh) for the user's dataset."""
    thr = rec.predicted_throughput or 0.0
    power = (rec.metadata or {}).get("predicted_power_watts") or 0.0
    runtime = total_tokens / thr if (thr > 0 and total_tokens > 0) else rec.predicted_runtime_seconds
    energy = (power * rec.total_gpus * runtime) / 3600.0 if (runtime and power) else None
    return runtime, energy


def recommendation_rationale(recs: list[Recommendation], meta: dict[str, Any]) -> str:
    """One-line rationale for the top recommendation vs runner-up."""
    if not recs:
        return "No feasible configuration in the search space."
    top = recs[0]
    # The phrase keys off the preset (balanced/performance/energy) or, for min_gpu, the
    # strategy name — both are canonical goals in the single `_goals` vocabulary.
    goal = (
        _goals.rationale_phrase(meta.get("preset"))
        or _goals.rationale_phrase(meta.get("strategy_name"))
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
