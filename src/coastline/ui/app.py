#!/usr/bin/env python3
"""Coastline — FastAPI Web Interface."""

from __future__ import annotations

import os

# Allow multiple OpenMP runtimes in one process. Selecting several ML models
# (catboost + xgboost + lightgbm + torch) loads native libs that each bundle
# libomp, which otherwise crashes on macOS. Must be set before those libs load.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import copy
import logging
import math
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, NamedTuple, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator

from coastline.sdk.constants import Strategy
from coastline.sdk.exceptions import UnsupportedGPUError
from coastline.sdk.io.infrastructure import Infrastructure, load_infrastructure
from coastline.sdk.io.options_loader import load_available_options
from coastline.sdk.io.run_config import (
    builtin_default_config,
    default_experiment_path,
    load_strategy_config,
)
from coastline.sdk.logging import setup_logging
from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.recommend import engine

from . import workload_queue

setup_logging()
logger = logging.getLogger(__name__)

_MODULE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(_MODULE_DIR / "templates"))

OPTIONS: dict[str, list] = {}
INFRA: Optional[Infrastructure] = None
STRATEGY_CONFIG: dict[str, Any] = {}

# Selectable prediction models for the UI (id → display name). Only Kavier
# produces results today; the ML models populate once the featv3 models exist.
# Order mirrors the thesis design-section model-mapping table (tab:exp1:model_mapping):
# retrieval (PR) → analytical (PA) → data-driven (PD) in the same sequence.
_PREDICTORS = [
    {"id": "cache", "name": "Cache lookup"},
    {"id": "kavier", "name": "Kavier (analytical)"},
    {"id": "random_forest", "name": "Random Forest"},
    {"id": "xgboost", "name": "XGBoost"},
    {"id": "lightgbm", "name": "LightGBM"},
    {"id": "catboost", "name": "CatBoost"},
    {"id": "bayesian_ridge", "name": "Bayesian Ridge"},
    {"id": "svr", "name": "SVR"},
    {"id": "knn", "name": "KNN"},
    {"id": "gaussian_process", "name": "Gaussian Process"},
    {"id": "deep_learning", "name": "Deep Learning"},
    {"id": "tabpfn", "name": "TabPFN"},
]

# The config-less fallback — the one built-in default, sourced from the bundled
# default_experiment.yaml (not a third hardcoded copy). autoconf degrades to rules under
# COASTLINE_ALLOW_RULES_FALLBACK=1.
_DEFAULT_STRATEGY_CONFIG: dict[str, Any] = builtin_default_config()


def _load_strategy_config() -> dict[str, Any]:
    """Resolve the one recommendation-policy config and load it.

    Discovery (env override → the repo's ``experiment.yaml``) is the shared
    :func:`coastline.sdk.io.run_config.default_experiment_path`; the file load is the shared
    ``load_strategy_config``. Both are shared with the CLI so every door resolves the same
    config. Merged over the one built-in default; falls back to it when no file is found.
    """
    path = default_experiment_path()
    if path.is_file():
        try:
            config = load_strategy_config(path, default=_DEFAULT_STRATEGY_CONFIG)
            logger.info("Loaded strategy config from %s", path)
            return config
        except Exception as exc:  # a malformed config must not abort startup
            logger.warning("Could not read strategy config %s: %s", path, exc)
    logger.warning("Using built-in default strategy config")
    return copy.deepcopy(_DEFAULT_STRATEGY_CONFIG)


class RecommendRequest(BaseModel):
    llm_model: str
    fine_tuning_method: str
    gpu_model: str
    tokens_per_sample: int = Field(gt=0)
    batch_size: int = Field(default=32, gt=0)
    training_epochs: int = Field(default=3, gt=0)
    dataset_size: int = Field(default=10000, gt=0)
    hardware_mode: Literal["total", "nodes"] = "total"
    total_gpus: int = Field(default=16, gt=0)
    num_nodes: int = Field(default=2, gt=0)
    gpus_per_node: int = Field(default=8, gt=0)
    prediction_model: str = "kavier"
    strategy: Literal["min_gpu", "multi_objective"] = "multi_objective"
    preset: Literal["energy", "balanced", "performance"] = "balanced"

    @field_validator("llm_model", "fine_tuning_method", "gpu_model")
    @classmethod
    def _strip_required_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("prediction_model")
    @classmethod
    def _known_predictor(cls, value: str) -> str:
        valid = {p["id"] for p in _PREDICTORS}
        if value not in valid:
            raise ValueError(f"unknown predictor {value!r}; valid ids: {sorted(valid)}")
        return value


_MAX_BATCH_WORKLOADS = int(os.environ.get("COASTLINE_MAX_BATCH_WORKLOADS", "200"))


class BatchRecommendRequest(BaseModel):
    """A batch of workloads (each a flexible dict), matching ``coastline.recommend``."""

    workloads: list[dict[str, Any]] = Field(..., min_length=1, max_length=_MAX_BATCH_WORKLOADS)
    top_k: int = Field(default=1, gt=0, le=20)
    goal: str = "balanced"
    predictor: str = "kavier"
    max_gpus: Optional[int] = Field(default=None, gt=0)
    max_slowdown: Optional[float] = Field(default=None, gt=0)
    # Feasibility checker (autoconf | rules | none), mirroring the single /api/recommend
    # path and the python API. 'rules' is the divisibility-only path that needs no AutoConf.
    feasibility: str = "autoconf"


class QueueAddRequest(BaseModel):
    """One workload added to the FIFO queue. The scheduler only consumes arrival_time +
    num_gpus + predicted_duration_s; the optional workload-config fields query Kavier at
    add-time for per-GPU power and (when complete) predicted runtime, which overwrites any
    supplied predicted_duration_s. predicted_power_watts_per_gpu overrides the Kavier power
    lookup; there is no equivalent override on the duration side."""

    request_id: Optional[str] = None
    arrival_time: Optional[float] = Field(default=None, ge=0.0)
    num_gpus: int = Field(..., ge=1)
    predicted_duration_s: Optional[float] = Field(default=None, gt=0.0)
    llm_model: Optional[str] = None
    fine_tuning_method: Optional[str] = None
    gpu_model: Optional[str] = None
    tokens_per_sample: Optional[int] = Field(default=None, gt=0)
    batch_size: Optional[int] = Field(default=None, gt=0)
    training_epochs: Optional[int] = Field(default=None, gt=0)
    dataset_size: Optional[int] = Field(default=None, gt=0)
    gpus_per_node: Optional[int] = Field(default=None, ge=1)
    number_of_nodes: Optional[int] = Field(default=None, ge=1)
    predicted_power_watts_per_gpu: Optional[float] = Field(default=None, gt=0)


class ImportCSVRequest(BaseModel):
    """CSV upload payload (sent as JSON to avoid a python-multipart dependency)."""

    csv: str = Field(..., max_length=5_000_000, description="Raw CSV text (max ~5 MB)")
    predict_durations: bool = Field(
        default=False, description="Replace CSV durations with Kavier predictions where possible."
    )


_kavier_predictor: Any = None
# Guards the lazy init of _kavier_predictor. The queue/import endpoints run in
# FastAPI's threadpool, so concurrent first calls could otherwise each construct a
# KavierPredictor (double-init race, wasted load). Double-checked locking keeps the
# steady-state fast path lock-free.
_kavier_predictor_lock = threading.Lock()


class _KavierEstimate(NamedTuple):
    """Per-GPU power and total runtime from one Kavier call. Either field may be
    None when Kavier can't predict for this workload."""

    power_watts_per_gpu: Optional[float]
    duration_seconds: Optional[float]


_KAVIER_EMPTY = _KavierEstimate(None, None)


def _kavier_predict(
    *,
    model: Optional[str],
    method: Optional[str],
    gpu_model: Optional[str],
    tokens_per_sample: Optional[int],
    batch_size: Optional[int],
    num_gpus: int,
    dataset_size: Optional[int] = None,
    training_epochs: Optional[int] = None,
    gpus_per_node: Optional[int] = None,
    number_of_nodes: Optional[int] = None,
) -> _KavierEstimate:
    """Single Kavier engine call for a queued workload, returning per-GPU power
    and total runtime. Either field may be None:

    * power requires the five base fields (model, method, gpu_model,
      tokens_per_sample, batch_size); the engine returns None for an
      uncalibrated (model, GPU) pair.
    * duration additionally requires dataset_size, training_epochs, and a
      positive throughput; runtime = ``dataset_size × training_epochs ×
      tokens_per_sample / throughput`` — the same formula the recommender uses.

    Returns ``_KAVIER_EMPTY`` on any missing-field gate, engine exception
    (logged), or null/non-positive output. Callers treat None as "fall back"."""
    if not (model and method and gpu_model and tokens_per_sample and batch_size):
        return _KAVIER_EMPTY
    per_node_cap = INFRA.max_gpus_per_node if INFRA is not None else 8
    if not gpus_per_node or not number_of_nodes:
        gpus_per_node = min(num_gpus, per_node_cap)
        number_of_nodes = max(1, math.ceil(num_gpus / max(gpus_per_node, 1)))
    try:
        global _kavier_predictor
        if _kavier_predictor is None:
            with _kavier_predictor_lock:
                # Re-check inside the lock: another thread may have built it while
                # we waited (double-checked locking).
                if _kavier_predictor is None:
                    from coastline.sdk.predictors.performance.physics.kavier_predictor import (  # noqa: E501
                        KavierPredictor,
                    )

                    _kavier_predictor = KavierPredictor()
        wl = WorkloadSpec(
            llm_model=model,
            fine_tuning_method=method,
            gpu_model=gpu_model,
            tokens_per_sample=int(tokens_per_sample),
            batch_size=int(batch_size),
            gpus_per_node=int(gpus_per_node),
            number_of_nodes=int(number_of_nodes),
        )
        ctx = SystemContext.for_gpus(
            [gpu_model],
            max_gpus=wl.total_gpus,
            gpus_per_node=int(gpus_per_node),
            max_nodes=int(number_of_nodes),
        )
        pred = _kavier_predictor.predict(wl, ctx)
        if not pred:
            return _KAVIER_EMPTY
        throughput = (
            float(pred.predicted_throughput) if pred.predicted_throughput and pred.predicted_throughput > 0 else None
        )
        power = float(pred.predicted_power) if pred.predicted_power and pred.predicted_power > 0 else None
        duration: Optional[float] = None
        if throughput and dataset_size and training_epochs:
            total_tokens = int(dataset_size) * int(training_epochs) * int(tokens_per_sample)
            duration = float(total_tokens) / throughput
        return _KavierEstimate(power_watts_per_gpu=power, duration_seconds=duration)
    except Exception as exc:
        logger.warning(
            "Kavier lookup failed for %s/%s on %s: %s",
            model,
            method,
            gpu_model,
            exc,
        )
    return _KAVIER_EMPTY


def _serialize_candidate(rank: int, rec: Any, total_tokens: int = 0) -> dict[str, Any]:
    """The web-response shape over the shared flattener (runtime/energy via the same
    engine.runtime_energy the facade/CLI use, so every door reports identical numbers)."""
    f = engine.flatten_recommendation(rec, total_tokens)
    return {
        "rank": rank,
        "gpus_per_node": f["gpus_per_node"],
        "workers": f["number_of_nodes"],
        "number_of_nodes": f["number_of_nodes"],
        "total_gpus": f["total_gpus"],
        "batch_size": f["batch_size"],
        "predicted_throughput": f["throughput"],
        "predicted_runtime_seconds": f["runtime_s"],
        "power_watts": f["power_w"],
        "energy_kwh": f["energy_kwh"],
        "tokens_per_watt": f["tokens_per_watt"],
        "score": f["combined_score"],
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global OPTIONS, INFRA, STRATEGY_CONFIG
    try:
        STRATEGY_CONFIG = _load_strategy_config()
    except Exception as exc:
        logger.error("Failed to load strategy config: %s", exc, exc_info=True)
        STRATEGY_CONFIG = dict(_DEFAULT_STRATEGY_CONFIG)

    try:
        OPTIONS = load_available_options()
        OPTIONS.setdefault("recommendation_policies", ["min_gpu", "multi_objective"])
        OPTIONS.setdefault("presets", ["energy", "balanced", "performance"])
        OPTIONS.setdefault("predictors", _PREDICTORS)
        logger.info(
            "Options loaded: %d models, %d GPUs",
            len(OPTIONS.get("models", [])),
            len(OPTIONS.get("gpus", [])),
        )
    except Exception as exc:
        logger.error("Failed to load options: %s", exc, exc_info=True)
        OPTIONS = {
            "models": ["llama3.1-70b"],
            "methods": ["full", "lora"],
            "gpus": ["NVIDIA-A100-SXM4-80GB", "NVIDIA-H100-PCIe", "L40S"],
            "tokens_per_sample": [512, 1024, 2048],
            "batch_sizes": [8, 16, 32],
            "recommendation_policies": ["min_gpu", "multi_objective"],
            "presets": ["energy", "balanced", "performance"],
            "predictors": _PREDICTORS,
        }

    # Sysadmin-declared cluster capacity (component F). Cached + surfaced to the UI;
    # enforced on /api/recommend. Falls back to conservative defaults if the file is
    # absent (a warning is logged so a sysadmin notices).
    INFRA = load_infrastructure()
    logger.info(
        "Infrastructure loaded: %d GPUs, %d max nodes, %d GPUs/node, %d GPU types",
        INFRA.total_gpus,
        INFRA.max_nodes,
        INFRA.max_gpus_per_node,
        len(INFRA.gpu_models),
    )

    yield


def _resolve_version() -> str:
    """The package version (single source of truth), with a source-checkout fallback."""
    try:
        from importlib.metadata import version

        return version("coastline")
    except Exception:  # not pip-installed (editable / source checkout)
        return "0.1.0"


API_VERSION = _resolve_version()

_OPENAPI_TAGS = [
    {"name": "recommend", "description": "GPU/node configuration recommendation (single, batch, CSV)."},
    {"name": "jobs", "description": "Async submit + poll for slow recommends (e.g. ML predictors)."},
    {"name": "predict", "description": "Run individual predictors on one exact config."},
    {"name": "queue", "description": "Workload queue management."},
    {"name": "admin", "description": "Queue simulation + bulk CSV import."},
    {"name": "meta", "description": "Health, version, available options, cluster infrastructure."},
    {"name": "dashboard", "description": "The bundled web UI."},
]

app = FastAPI(
    title="Coastline",
    description="Context-aware recommender system for LLM fine-tuning workloads.",
    version=API_VERSION,
    openapi_tags=_OPENAPI_TAGS,
    lifespan=lifespan,
)

# CORS: restrict to known origins (override via COASTLINE_CORS_ORIGINS, comma-separated).
# Defaults to localhost so the bundled dashboard works without exposing the
# state-mutating /api/admin/* routes to arbitrary cross-origin pages.
_cors_origins = [
    o.strip()
    for o in os.environ.get("COASTLINE_CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(_MODULE_DIR / "static")), name="static")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    message = errors[0].get("msg", "Invalid request") if errors else "Invalid request"
    return JSONResponse(
        status_code=422,
        content={"success": False, "error": message, "detail": message},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": detail, "detail": detail},
    )


# Typed response envelopes (for OpenAPI spec / client codegen). The dynamic
# payloads (recommendation/candidates/workload_summary) stay dict-typed so nothing
# is filtered; the API tests assert the exact fields, guarding against drops.
class HealthResponse(BaseModel):
    success: bool
    status: str
    options_loaded: bool
    config_loaded: bool
    strategy_config_loaded: bool
    infrastructure_loaded: bool


class VersionResponse(BaseModel):
    success: bool
    name: str
    version: str


class RecommendResponse(BaseModel):
    success: bool
    recommendation: dict[str, Any]
    candidates: list[dict[str, Any]]
    rationale: Optional[str] = None
    strategy: str
    preset: Optional[str] = None
    workload_summary: dict[str, Any]


class BatchRecommendResponse(BaseModel):
    success: bool
    count: int
    results: list[dict[str, Any]]


class CSVRecommendResponse(BaseModel):
    success: bool
    count: int
    csv: str


@app.get("/", tags=["dashboard"])
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"options": OPTIONS, "infra": INFRA.model_dump() if INFRA else None},
    )


@app.get("/api/health", tags=["meta"], response_model=HealthResponse)
async def health():
    return {
        "success": True,
        "status": "healthy",
        "options_loaded": bool(OPTIONS),
        "config_loaded": True,
        "strategy_config_loaded": bool(STRATEGY_CONFIG),
        "infrastructure_loaded": INFRA is not None,
    }


@app.get("/api/infrastructure", tags=["meta"])
async def get_infrastructure():
    """Sysadmin-declared cluster capacity (component F).

    The UI surfaces these caps to the user ("Available: N GPUs, up to M nodes, …")
    and the backend enforces them on inbound recommendations — requests beyond the
    advertised capacity get a 400 from /api/recommend.
    """
    if INFRA is None:
        raise HTTPException(status_code=503, detail="Infrastructure config not loaded")
    return {"success": True, **INFRA.model_dump()}


@app.get("/api/options", tags=["meta"])
async def get_options():
    """Discoverable inputs for /api/recommend and /api/predict: the available models,
    methods, GPUs, sequence lengths, batch sizes, recommendation_policies, presets and predictors.
    A programmatic consumer reads this instead of scraping the dashboard HTML."""
    if not OPTIONS:
        raise HTTPException(status_code=503, detail="Options not loaded")
    return {"success": True, **OPTIONS}


@app.get("/api/version", tags=["meta"], response_model=VersionResponse)
async def get_version():
    """API + package version, so a consumer (e.g. ado) can gate against a contract."""
    return {"success": True, "name": "coastline", "version": API_VERSION}


@app.post("/api/recommend", tags=["recommend"], response_model=RecommendResponse)
def recommend(body: RecommendRequest):
    """Generate an optimised GPU configuration recommendation.

    Plain ``def`` ON PURPOSE: the body is synchronous CPU-bound work (grid
    simulation; ML predictors can block >10s — TabPFN ~minutes). As ``async def``
    it ran ON the single-worker event loop and froze the entire app (the live-demo
    spinner-of-death). FastAPI runs ``def`` routes in the threadpool instead.
    """
    try:
        # Enforce the sysadmin-declared cluster capacity (component F).
        if INFRA is not None:
            if body.hardware_mode == "nodes":
                if body.gpus_per_node > INFRA.max_gpus_per_node:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Requested {body.gpus_per_node} GPUs/node exceeds the cluster's {INFRA.max_gpus_per_node}."
                        ),
                    )
                if body.num_nodes > INFRA.max_nodes:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Requested {body.num_nodes} nodes exceeds the cluster's {INFRA.max_nodes}.",
                    )
                if body.num_nodes * body.gpus_per_node > INFRA.total_gpus:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Requested {body.num_nodes * body.gpus_per_node} GPUs exceeds the "
                            f"cluster's {INFRA.total_gpus}."
                        ),
                    )
            elif body.total_gpus > INFRA.total_gpus:
                raise HTTPException(
                    status_code=400,
                    detail=f"Requested {body.total_gpus} GPUs exceeds the cluster's {INFRA.total_gpus}.",
                )

        # Resolve the hardware budget into grid caps (max_gpus, gpus_per_node, max_nodes).
        if body.hardware_mode == "nodes":
            gpus_per_node = body.gpus_per_node
            max_nodes = body.num_nodes
            max_gpus = body.num_nodes * body.gpus_per_node
        else:  # "total" — recommender derives the node layout
            max_gpus = body.total_gpus
            gpus_per_node = min(8, body.total_gpus)
            max_nodes = max(1, math.ceil(body.total_gpus / gpus_per_node))

        workload = WorkloadSpec(
            llm_model=body.llm_model,
            fine_tuning_method=body.fine_tuning_method,
            gpu_model=body.gpu_model,
            tokens_per_sample=body.tokens_per_sample,
            batch_size=body.batch_size,
            gpus_per_node=gpus_per_node,
            number_of_nodes=1,
        )

        gpu_model = body.gpu_model
        context = SystemContext.for_gpus(
            [gpu_model],
            max_gpus=max_gpus,
            gpus_per_node=gpus_per_node,
            max_nodes=max_nodes,
        )

        # Deep-copy the shared module-level STRATEGY_CONFIG before mutating it for
        # this request. /api/recommend runs in FastAPI's threadpool, so several
        # requests touch STRATEGY_CONFIG concurrently; a shallow spread copies the
        # top level but leaves nested sub-dicts (e.g. ``grid``) shared by reference,
        # so a downstream mutation in one request could race another. The override
        # below already rebuilds ``predictors``; deepcopy protects ``grid``/``strategy``.
        req_config = copy.deepcopy(STRATEGY_CONFIG)
        req_config["predictors"] = {
            **req_config.get("predictors", {}),
            "performance": body.prediction_model,
        }
        preset = body.preset if body.strategy == Strategy.MULTI_OBJECTIVE else None
        total_tokens = body.dataset_size * body.training_epochs * body.tokens_per_sample
        # Route through the single engine seam; INFRA caps + hardware-mode resolution
        # (above) and serialization (below) stay UI-specific.
        recs, meta = engine.run_request(
            engine.RecommendRequest(
                workload=workload,
                context=context,
                config=req_config,
                strategy_name=body.strategy,
                preset=preset,
                total_tokens=total_tokens,
            )
        )

        if not recs:
            raise HTTPException(
                status_code=404,
                detail="No recommendation could be generated for this configuration.",
            )

        candidates = [_serialize_candidate(i + 1, rec, total_tokens) for i, rec in enumerate(recs)]

        return {
            "success": True,
            "recommendation": candidates[0],
            "candidates": candidates,
            "rationale": engine.recommendation_rationale(recs, meta),
            "strategy": body.strategy,
            "preset": body.preset if body.strategy == Strategy.MULTI_OBJECTIVE else None,
            "workload_summary": {
                "llm_model": body.llm_model,
                "fine_tuning_method": body.fine_tuning_method,
                "gpu_model": body.gpu_model,
                "tokens_per_sample": body.tokens_per_sample,
                "training_epochs": body.training_epochs,
                "dataset_size": body.dataset_size,
                "batch_size": body.batch_size,
            },
        }

    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning("Recommendation validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        # The grid pipeline raises RuntimeError("no feasible candidates ...") when
        # Kavier knows nothing about the (model, GPU, method) — e.g. a typo'd or
        # uncalibrated model. That is "nothing to recommend" (404, friendly toast
        # in the UI), not a server fault (500, raw red error).
        logger.warning("No feasible candidates: %s", exc)
        raise HTTPException(
            status_code=404,
            detail="No feasible configuration found for this workload (model/GPU/method may be unsupported).",
        ) from exc
    except UnsupportedGPUError as exc:
        logger.warning("Unknown GPU model: %s", exc)
        raise HTTPException(
            status_code=404,
            detail=f"Unknown GPU model: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Recommendation failed")
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {exc}",
        ) from exc


class PredictRequest(BaseModel):
    llm_model: str
    fine_tuning_method: str
    gpu_model: str
    tokens_per_sample: int = Field(gt=0)
    batch_size: int = Field(default=8, gt=0)
    gpus_per_node: int = Field(default=8, gt=0)
    number_of_nodes: int = Field(default=1, gt=0)
    training_epochs: int = Field(default=3, gt=0)
    dataset_size: int = Field(default=10000, gt=0)
    models: list[str] = Field(default_factory=lambda: ["kavier"])

    @field_validator("llm_model", "fine_tuning_method", "gpu_model")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


@app.post("/api/recommend/batch", tags=["recommend"], response_model=BatchRecommendResponse)
def recommend_batch(body: BatchRecommendRequest):
    """Batch recommend via the canonical ``coastline.recommend`` facade, so the numbers and
    columns (including ``rationale``) match the python API / CLI exactly. Per-row isolation:
    a bad workload yields a feasible=False + error row, never failing the whole batch."""
    import pandas as pd

    import coastline

    try:
        df = coastline.recommend(
            body.workloads,
            top_k=body.top_k,
            goal=body.goal,
            predictor=body.predictor,
            max_gpus=body.max_gpus,
            max_slowdown=body.max_slowdown,
            feasibility=body.feasibility,
        )
    except (ValueError, TypeError) as exc:  # unknown goal, bad workload shape, etc.
        raise HTTPException(status_code=422, detail=str(exc))
    records = df.where(pd.notna(df), None).to_dict(orient="records")
    return {"success": True, "count": len(records), "results": records}


class RecommendCSVRequest(BaseModel):
    """CSV text in -> recommendations as CSV text out (the IBM file-pipeline shape over HTTP)."""

    csv: str = Field(..., max_length=5_000_000, description="Input CSV (one workload per row).")
    goal: str = "balanced"
    predictor: str = "kavier"
    max_gpus: Optional[int] = Field(default=None, gt=0)
    # Feasibility checker (autoconf | rules | none), mirroring /api/recommend and the API.
    feasibility: str = "autoconf"


@app.post("/api/recommend/csv", tags=["recommend"], response_model=CSVRecommendResponse)
def recommend_csv_endpoint(body: RecommendCSVRequest):
    """Recommend for a CSV of workloads, returning a CSV — same flexible columns and
    rationale as ``coastline.recommend`` / the CLI, with no file upload needed."""
    import csv as _csv
    import io

    import coastline

    rows = list(_csv.DictReader(io.StringIO(body.csv)))
    if not rows:
        raise HTTPException(status_code=400, detail="empty CSV (no data rows)")
    if len(rows) > _MAX_BATCH_WORKLOADS:
        raise HTTPException(status_code=413, detail=f"too many rows ({len(rows)}); max {_MAX_BATCH_WORKLOADS}.")
    try:
        df = coastline.recommend(
            rows,
            goal=body.goal,
            predictor=body.predictor,
            max_gpus=body.max_gpus,
            top_k=1,
            feasibility=body.feasibility,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    out = io.StringIO()
    df.to_csv(out, index=False)
    return {"success": True, "count": len(df), "csv": out.getvalue()}


# Async jobs: submit a batch recommend and poll for the result, so slow ML predictors
# (e.g. TabPFN, ~minutes) don't tie up a request. In-process store (single-replica).
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _set_job(job_id: str, status: str, result: Any, error: Optional[str]) -> None:
    with _jobs_lock:
        _jobs[job_id] = {"status": status, "result": result, "error": error}


def _run_recommend_job(job_id: str, body: BatchRecommendRequest) -> None:
    try:
        import pandas as pd

        import coastline

        df = coastline.recommend(
            body.workloads,
            top_k=body.top_k,
            goal=body.goal,
            predictor=body.predictor,
            max_gpus=body.max_gpus,
            max_slowdown=body.max_slowdown,
            feasibility=body.feasibility,
        )
        results = df.where(pd.notna(df), None).to_dict(orient="records")
        _set_job(job_id, "done", {"count": len(results), "results": results}, None)
    except Exception as exc:  # record the failure on the job rather than crash the worker
        _set_job(job_id, "error", None, str(exc)[:500])


@app.post("/api/jobs", status_code=202, tags=["jobs"])
def submit_job(body: BatchRecommendRequest):
    """Submit a batch recommend to run in the background; returns a job id immediately.
    Poll GET /api/jobs/{job_id} for the result. Use this for slow predictors (TabPFN ~min)."""
    job_id = uuid.uuid4().hex
    _set_job(job_id, "pending", None, None)
    threading.Thread(target=_run_recommend_job, args=(job_id, body), daemon=True).start()
    return {"success": True, "job_id": job_id, "status": "pending"}


@app.get("/api/jobs/{job_id}", tags=["jobs"])
def get_job(job_id: str):
    """Poll a submitted job: status is pending | done | error; result/error filled accordingly."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
    return {"success": True, "job_id": job_id, **job}


@app.post("/api/predict", tags=["predict"])
def predict(body: PredictRequest):
    """Playground: run each selected predictor on one exact configuration.

    Each model runs in its own spawned subprocess so several native ML runtimes
    never coexist in one process (which crashes on macOS).
    """
    import json
    import subprocess
    import sys

    total_tokens = body.dataset_size * body.training_epochs * body.tokens_per_sample
    name_by_id = {p["id"]: p["name"] for p in _PREDICTORS}
    models = body.models or ["kavier"]
    # Cap models-per-request: each runs in its own (up-to-timeout) subprocess, so an
    # unbounded list ties up a threadpool slot for minutes and can starve the API.
    _max_models = int(os.environ.get("COASTLINE_MAX_PREDICT_MODELS", "6"))
    if len(models) > _max_models:
        raise HTTPException(
            status_code=400,
            detail=f"Too many models requested ({len(models)}); max {_max_models} per request.",
        )

    results = []
    for model_id in models:
        label = name_by_id.get(model_id, model_id)
        payload = {
            "model_id": model_id,
            "label": label,
            "llm_model": body.llm_model,
            "fine_tuning_method": body.fine_tuning_method,
            "gpu_model": body.gpu_model,
            "tokens_per_sample": body.tokens_per_sample,
            "batch_size": body.batch_size,
            "gpus_per_node": body.gpus_per_node,
            "number_of_nodes": body.number_of_nodes,
            "total_tokens": total_tokens,
        }
        # One subprocess per model: a single native ML runtime per process, and a
        # crash in one model is isolated (it just marks that model unavailable).
        try:
            # coastline is an installed package, so the worker resolves on the
            # subprocess's sys.path with no PYTHONPATH juggling. Inherit the env so
            # KMP_DUPLICATE_LIB_OK / DATA_DIR / PORTFOLIO_DIR pass through.
            proc = subprocess.run(
                [sys.executable, "-m", "coastline.ui.prediction_worker"],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=60,
                env=os.environ.copy(),
            )
        except Exception as exc:
            # Launch/timeout failure (subprocess never produced a clean result) —
            # isolate it as "this model is unavailable" rather than failing the batch.
            logger.warning("Predict worker for %s errored: %s", model_id, exc)
            results.append({"model": model_id, "label": label, "available": False})
            continue

        if proc.returncode != 0 or not proc.stdout.strip():
            # The worker exited non-zero (e.g. a missing ML artifact / native crash);
            # that model is simply unavailable for this config.
            logger.warning("Predict worker for %s exited rc=%s", model_id, proc.returncode)
            results.append({"model": model_id, "label": label, "available": False})
            continue

        # rc==0 with output: the worker claims success, so a JSON parse failure is a
        # contract violation (corrupt worker output), not "model unavailable". Surface
        # it as a 500 carrying the worker's stderr so the cause is visible.
        try:
            results.append(json.loads(proc.stdout))
        except json.JSONDecodeError as exc:
            logger.error(
                "Predict worker for %s returned rc=0 but unparseable stdout: %s; stderr=%s",
                model_id,
                exc,
                proc.stderr.strip(),
            )
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Prediction worker for '{model_id}' returned malformed output: "
                    f"{exc}. Worker stderr: {proc.stderr.strip() or '<empty>'}"
                ),
            ) from exc

    return {
        "success": True,
        "config": {
            "llm_model": body.llm_model,
            "fine_tuning_method": body.fine_tuning_method,
            "gpu_model": body.gpu_model,
            "tokens_per_sample": body.tokens_per_sample,
            "batch_size": body.batch_size,
            "total_gpus": body.gpus_per_node * body.number_of_nodes,
            "gpus_per_node": body.gpus_per_node,
            "number_of_nodes": body.number_of_nodes,
            "dataset_size": body.dataset_size,
            "training_epochs": body.training_epochs,
        },
        "results": results,
    }


# Workload queue + admin (component I — FIFO scheduler harness). Independent from
# the recommend path: a user adds/removes jobs, the FIFO discrete-event simulator
# "runs" the queue on the cluster (component F's total_gpus), and reports per-job +
# cluster-wide metrics. CSV import accepts flexible trace schemas (column aliases).


@app.post("/api/queue", tags=["queue"])
async def queue_add(body: QueueAddRequest):
    """Add a job to the FIFO queue. When the optional workload-config is given,
    Kavier supplies per-GPU power and (if the full duration config is present)
    total runtime, which overwrites any supplied predicted_duration_s. The
    response's duration_source ∈ {"kavier", "user"} reports which path won."""
    if INFRA is not None and body.num_gpus > INFRA.total_gpus:
        raise HTTPException(
            status_code=400,
            detail=f"num_gpus={body.num_gpus} exceeds the cluster's {INFRA.total_gpus}.",
        )
    kavier = _kavier_predict(
        model=body.llm_model,
        method=body.fine_tuning_method,
        gpu_model=body.gpu_model,
        tokens_per_sample=body.tokens_per_sample,
        batch_size=body.batch_size,
        dataset_size=body.dataset_size,
        training_epochs=body.training_epochs,
        num_gpus=body.num_gpus,
        gpus_per_node=body.gpus_per_node,
        number_of_nodes=body.number_of_nodes,
    )
    # Explicit power override > Kavier > simulator fallback constant.
    power = (
        body.predicted_power_watts_per_gpu
        if body.predicted_power_watts_per_gpu is not None
        else kavier.power_watts_per_gpu
    )
    # Kavier duration wins when its config is complete; else the caller's is required.
    if kavier.duration_seconds is not None and kavier.duration_seconds > 0:
        duration = kavier.duration_seconds
        duration_source = "kavier"
    elif body.predicted_duration_s is not None and body.predicted_duration_s > 0:
        duration = float(body.predicted_duration_s)
        duration_source = "user"
    else:
        raise HTTPException(
            status_code=422,
            detail=(
                "Provide either predicted_duration_s, or a complete workload "
                "config (llm_model, fine_tuning_method, gpu_model, "
                "tokens_per_sample, batch_size, dataset_size, training_epochs) "
                "so Kavier can predict it."
            ),
        )
    job = workload_queue.add_job(
        workload_queue.QueueJob(
            request_id=body.request_id or workload_queue.generate_id(),
            arrival_time=body.arrival_time if body.arrival_time is not None else time.time(),
            num_gpus=body.num_gpus,
            predicted_duration_s=duration,
            predicted_power_watts_per_gpu=power,
            llm_model=body.llm_model,
            fine_tuning_method=body.fine_tuning_method,
            gpu_model=body.gpu_model,
            tokens_per_sample=body.tokens_per_sample,
            batch_size=body.batch_size,
            dataset_size=body.dataset_size,
            training_epochs=body.training_epochs,
            gpus_per_node=body.gpus_per_node,
            number_of_nodes=body.number_of_nodes,
        )
    )
    return {
        "success": True,
        "job": job.model_dump(),
        "duration_source": duration_source,
        "queue_size": len(workload_queue.list_jobs()),
    }


@app.delete("/api/queue/{request_id}", tags=["queue"])
async def queue_remove(request_id: str):
    if not workload_queue.remove_job(request_id):
        raise HTTPException(status_code=404, detail=f"Job {request_id} not in queue")
    return {"success": True, "queue_size": len(workload_queue.list_jobs())}


@app.get("/api/queue", tags=["queue"])
async def queue_list():
    return {"success": True, "jobs": [j.model_dump() for j in workload_queue.list_jobs()]}


@app.post("/api/admin/run", tags=["admin"])
async def admin_run():
    """FIFO-schedule the queue on the cluster; return per-job + cluster-wide metrics."""
    if INFRA is None:
        raise HTTPException(status_code=503, detail="Infrastructure not loaded")
    jobs = workload_queue.list_jobs()
    if not jobs:
        return {
            "success": True,
            "totals": None,
            "jobs": [],
            "message": "Queue is empty.",
            "cluster_gpus": INFRA.total_gpus,
        }
    result = workload_queue.simulate_fifo(jobs, INFRA.total_gpus)
    # Step-series for the cluster figure (GPUs allocated + queue depth over time).
    # Derived from the same finished run — no second simulation — and fed to the
    # dashboard's cluster-timeline chart.
    timeline = workload_queue.build_cluster_timeline(result.jobs, INFRA.total_gpus)
    return {
        "success": True,
        "totals": {
            "makespan_s": result.makespan_s,
            "avg_resource_occupation": result.avg_resource_occupation,
            "goodput_jobs_per_s": result.goodput_jobs_per_s,
            "avg_waiting_time_s": result.avg_waiting_time_s,
            "avg_job_completion_time_s": result.avg_job_completion_time_s,
            "total_energy_kwh": result.total_energy_kwh,
            "n_jobs": result.n_jobs,
            "cluster_gpus": INFRA.total_gpus,
        },
        "timeline": {
            "t": timeline.t,
            "gpus_used": timeline.gpus_used,
            "queue_depth": timeline.queue_depth,
            "cluster_gpus": timeline.cluster_gpus,
            "makespan_s": timeline.makespan_s,
            "peak_gpus": timeline.peak_gpus,
            "peak_queue": timeline.peak_queue,
        },
        "jobs": [
            {
                "request_id": j.request_id,
                "num_gpus": j.num_gpus,
                "predicted_duration_s": j.predicted_duration_s,
                "arrival_time": j.arrival_time,
                "start_time": j.start_time,
                "end_time": j.end_time,
                "wait_time_s": j.wait_time_s,
                "completion_time_s": j.completion_time_s,
                "energy_kwh": j.energy_kwh,
                "llm_model": j.llm_model,
                "fine_tuning_method": j.fine_tuning_method,
                "batch_size": j.batch_size,
                "training_epochs": j.training_epochs,
            }
            for j in result.jobs
        ],
    }


@app.post("/api/admin/import", tags=["admin"])
async def admin_import(body: ImportCSVRequest):
    """Bulk-import jobs from a CSV (multiple trace schemas tolerated —
    see workload_queue.parse_csv for the recognised column aliases).

    For each row with a complete workload config the handler queries Kavier for
    per-GPU power and, when ``predict_durations`` is on, the predicted runtime.
    The text is POSTed as JSON so the api image needs no python-multipart."""
    try:
        jobs = workload_queue.parse_csv(body.csv)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"CSV parse error: {e}")
    _max_rows = int(os.environ.get("COASTLINE_MAX_IMPORT_ROWS", "1000"))
    if len(jobs) > _max_rows:
        raise HTTPException(
            status_code=413,
            detail=f"CSV has {len(jobs)} rows; max {_max_rows} per import.",
        )
    durations_predicted = 0
    accepted = 0
    for j in jobs:
        if INFRA is not None and j.num_gpus > INFRA.total_gpus:
            continue  # silently drop rows beyond the cluster cap
        # One Kavier call covers both power and duration; the duration leg only
        # matters when predict_durations is on, but the call shape is the same.
        kavier = _kavier_predict(
            model=j.llm_model,
            method=j.fine_tuning_method,
            gpu_model=j.gpu_model,
            tokens_per_sample=j.tokens_per_sample,
            batch_size=j.batch_size,
            dataset_size=j.dataset_size,
            training_epochs=j.training_epochs,
            num_gpus=j.num_gpus,
            gpus_per_node=j.gpus_per_node,
            number_of_nodes=j.number_of_nodes,
        )
        # Honour any explicit per-GPU power the CSV row already carries.
        if j.predicted_power_watts_per_gpu is None and kavier.power_watts_per_gpu is not None:
            j.predicted_power_watts_per_gpu = kavier.power_watts_per_gpu
        if body.predict_durations and kavier.duration_seconds is not None and kavier.duration_seconds > 0:
            j.predicted_duration_s = kavier.duration_seconds
            durations_predicted += 1
        workload_queue.add_job(j)
        accepted += 1
    return {
        "success": True,
        "imported": accepted,
        "durations_predicted": durations_predicted,
        "queue_size": len(workload_queue.list_jobs()),
    }


@app.post("/api/admin/clear", tags=["admin"])
async def admin_clear():
    n = workload_queue.clear_jobs()
    return {"success": True, "cleared": n}
