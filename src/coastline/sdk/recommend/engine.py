"""Recommendation logic without Rich or prompts, shared by the REPL, the no-TTY path and the APIs."""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from coastline.sdk.constants import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_GPUS_PER_NODE,
    DEFAULT_TOKENS_PER_SAMPLE,
    GPU_BUDGETS,
    Method,
)
from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.recommendation import Recommendation
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.pipeline.parallel import RUNTIME_SECTION, WORKERS_KEY
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
FALLBACK_METHODS = [m.value for m in Method]
FALLBACK_GPUS = ["NVIDIA-A100-SXM4-80GB", "NVIDIA-A100-80GB-PCIe", "L40S"]
FALLBACK_TOKENS = DEFAULT_TOKENS_PER_SAMPLE
FALLBACK_BATCH_SIZES = DEFAULT_BATCH_SIZES

# Goal display label to (strategy_name, preset), built from `_goals`. The REPL lists these keys
# as menu choices.
GOALS: dict[str, tuple[str, Optional[str]]] = _goals.engine_goals()

# Top-level performance-predictor choices for the UI. "ml" is a sentinel that
# opens the trained-ML submenu (ML_MODELS); the rest are engine predictor keys.
PREDICTOR_CHOICES: list[tuple[str, str]] = [
    ("intelligent", "intelligent  |  exact cache match, else Kavier physics"),
    ("kavier", "physics simulator  |  Kavier"),
    ("ml", "trained ML model  |  you pick"),
    ("cache", "exact match  |  measured past runs only"),
]

# Data-driven models, listed in the "trained ML model" submenu.
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
    workers: Optional[int] = None,
) -> tuple[dict, str, Optional[str]]:
    """Build strategy-config dict for PolicyFactory; max_slowdown maps to runtime_guard_k.

    ``feasibility`` selects the checker (``autoconf`` | ``rules`` | ``none``); the
    answers dict may override it via a ``feasibility`` key.
    ``workers`` sets ``runtime.parallel_workers``, the per-stage worker count the pipeline
    forks candidates across; None leaves the block out and the pipeline runs sequentially.
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
            # The chosen batch size with half and double of it, so the table has more than one
            # row. A given batch grid, such as the trace's per-device sweep, replaces them.
            "batch_sizes": (
                list(answers["batch_sizes"])
                if answers.get("batch_sizes")
                else sorted({answers["batch_size"], max(1, answers["batch_size"] // 2), answers["batch_size"] * 2})
            ),
            "total_gpus": [g for g in GPU_BUDGETS if g <= answers["max_gpus"]],
            "top_k": top_k,
        },
    }
    if workers is not None:
        config[RUNTIME_SECTION] = {WORKERS_KEY: int(workers)}
    return config, strategy_name, preset


def _gpus_per_node(answers: dict[str, Any]) -> int:
    """The node width the grid may use: the ``max_gpus_per_node`` cap (default 8), within max_gpus."""
    cap = answers.get("max_gpus_per_node") or DEFAULT_GPUS_PER_NODE
    return min(int(cap), int(answers["max_gpus"]))


def build_workload(answers: dict[str, Any]) -> WorkloadSpec:
    return WorkloadSpec(
        llm_model=answers["llm_model"],
        fine_tuning_method=answers["fine_tuning_method"],
        gpu_model=answers["gpu_model"],
        tokens_per_sample=int(answers["tokens_per_sample"]),
        batch_size=int(answers["batch_size"]),
        gpus_per_node=_gpus_per_node(answers),
        number_of_nodes=1,
    )


def build_context(answers: dict[str, Any]) -> SystemContext:
    max_gpus = int(answers["max_gpus"])
    return SystemContext.for_gpus([answers["gpu_model"]], max_gpus=max_gpus, gpus_per_node=_gpus_per_node(answers))


@dataclass
class RecommendRequest:
    """Input to :func:`run_request`. The facade, the config-driven ``run`` command, the UI and
    :func:`run_pipeline` each build one; input parsing and output formatting stay with them."""

    workload: WorkloadSpec
    context: SystemContext
    config: dict[str, Any]  # full PolicyFactory config: strategy, predictors, grid
    strategy_name: str
    preset: Optional[str] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    total_tokens: int = 0  # for runtime and energy in meta; 0 means unused (facade, run.py)


def build_strategy(
    config: dict[str, Any],
    strategy_name: str,
    preset: Optional[str] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
) -> "BaseStrategy":
    """Build a strategy with ``PolicyFactory.create_strategy``. Separate from
    :func:`execute_strategy` so a caller such as ``batch_csv`` can build it once for all rows,
    loading the predictors and the AutoConf model once."""
    from coastline.sdk.policies import PolicyFactory

    return PolicyFactory.create_strategy(
        strategy_name=strategy_name, preset=preset, alpha=alpha, beta=beta, config=config
    )


class StrategyCache:
    """Reuse a built strategy across calls with the same config.

    :func:`build_strategy` constructs every predictor and the feasibility checker. The config
    can differ between trace rows (:func:`build_config` derives ``grid.batch_sizes`` from the
    row's batch size unless a sweep is given) and the grid is fixed when the pipeline is built,
    so the cache is keyed on the full config. A config that cannot be serialized is built
    without caching.
    """

    def __init__(self, capacity: int = 128) -> None:
        self._capacity = capacity
        self._entries: "OrderedDict[str, BaseStrategy]" = OrderedDict()
        self.builds = 0  # strategies constructed (for logging and tests)
        self.hits = 0

    @staticmethod
    def _key(
        config: dict[str, Any],
        strategy_name: str,
        preset: Optional[str],
        alpha: Optional[float],
        beta: Optional[float],
    ) -> Optional[str]:
        try:
            return json.dumps([config, strategy_name, preset, alpha, beta], sort_keys=True, default=repr)
        except (TypeError, ValueError):
            return None

    def get(
        self,
        config: dict[str, Any],
        strategy_name: str,
        preset: Optional[str] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> "BaseStrategy":
        key = self._key(config, strategy_name, preset, alpha, beta)
        if key is None:
            self.builds += 1
            return build_strategy(config, strategy_name, preset, alpha, beta)
        cached = self._entries.get(key)
        if cached is not None:
            self.hits += 1
            self._entries.move_to_end(key)
            return cached
        strategy = build_strategy(config, strategy_name, preset, alpha, beta)
        self.builds += 1
        self._entries[key] = strategy
        if len(self._entries) > self._capacity:
            self._entries.popitem(last=False)
        return strategy


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
    """Run a pre-built strategy on one workload, time it and return ``(recs, meta)``.
    A ``None`` or single result becomes a list."""
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


def run_request(
    request: RecommendRequest, strategy_cache: Optional[StrategyCache] = None
) -> tuple[list[Recommendation], dict[str, Any]]:
    """Build the strategy, run it and return ``(recs, meta)``.

    ``strategy_cache`` lets a batch caller (a trace, a CSV) reuse one strategy across rows that
    share a config. With ``None`` the strategy is built on every call.
    """
    args = (request.config, request.strategy_name, request.preset, request.alpha, request.beta)
    strategy = build_strategy(*args) if strategy_cache is None else strategy_cache.get(*args)
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
    strategy_cache: Optional[StrategyCache] = None,
    workers: Optional[int] = None,
) -> tuple[list[Recommendation], dict[str, Any]]:
    """Build a ``RecommendRequest`` from an ``answers`` dict and run it with :func:`run_request`.
    Used by the interactive REPL, the no-TTY path and ``batch_api``.

    ``feasibility`` (``autoconf``, ``rules`` or ``none``) picks the feasibility checker; a
    ``feasibility`` key in ``answers`` takes precedence (see ``build_config``).
    """
    config, strategy_name, preset = build_config(answers, top_k, max_slowdown, feasibility, workers)
    total_tokens = int(answers["dataset_size"] * answers["epochs"] * answers["tokens_per_sample"])
    return run_request(
        RecommendRequest(
            workload=build_workload(answers),
            context=build_context(answers),
            config=config,
            strategy_name=strategy_name,
            preset=preset,
            total_tokens=total_tokens,
        ),
        strategy_cache=strategy_cache,
    )


def runtime_energy(rec: Recommendation, total_tokens: int) -> tuple[Optional[float], Optional[float]]:
    """Return (runtime_s, energy_wh) for the user's dataset."""
    thr = rec.predicted_throughput or 0.0
    power = (rec.metadata or {}).get("predicted_power_watts") or 0.0
    runtime = total_tokens / thr if (thr > 0 and total_tokens > 0) else rec.predicted_runtime_seconds
    energy = (power * rec.total_gpus * runtime) / 3600.0 if (runtime and power) else None
    return runtime, energy


def flatten_recommendation(rec: Recommendation, total_tokens: int = 0) -> dict[str, Any]:
    """Raw values of one Recommendation; each output format renames the keys to its own columns.
    Runtime and energy come from :func:`runtime_energy`; with ``total_tokens=0`` the runtime is
    the predictor's ``predicted_runtime_seconds``."""
    runtime, energy_wh = runtime_energy(rec, total_tokens)
    meta = rec.metadata or {}
    return {
        "total_gpus": rec.total_gpus,
        "gpus_per_node": rec.gpus_per_node,
        "number_of_nodes": rec.number_of_nodes,
        "batch_size": meta.get("batch_size"),
        "throughput": rec.predicted_throughput,
        "runtime_s": runtime,
        "power_w": meta.get("predicted_power_watts"),
        "energy_wh": energy_wh,
        "energy_kwh": None if energy_wh is None else energy_wh / 1000.0,
        "tokens_per_watt": meta.get("tokens_per_watt"),
        "combined_score": meta.get("combined_score"),
    }


def recommendation_rationale(recs: list[Recommendation], meta: dict[str, Any]) -> str:
    """One-line rationale for the top recommendation vs runner-up."""
    if not recs:
        return "No feasible configuration in the search space."
    top = recs[0]
    # The phrase comes from the preset (balanced, performance, energy) or, for min_gpu, the
    # strategy name; both are goals in `_goals`.
    goal = (
        _goals.rationale_phrase(meta.get("preset"))
        or _goals.rationale_phrase(meta.get("strategy_name"))
        or "the best throughput/energy trade-off"
    )
    plural = "s" if top.total_gpus != 1 else ""
    top_batch = (top.metadata or {}).get("batch_size")
    config = f"{top.gpus_per_node}x{top.number_of_nodes}" + (f", batch {top_batch}" if top_batch else "")
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
