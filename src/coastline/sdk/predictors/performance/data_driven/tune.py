"""Tune a data-driven predictor on any measured-runs CSV (``coastline tune``).

The dataset is validated loudly: missing required columns hard-fail with the full
schema spelled out; quality problems (too few rows, one config, models unknown to
Kavier's library, ...) are reported after tuning as "Tuning may have produced poor
results because valid datasets should have these properties: ...".

The artifact written is a featv3 pickle with the same shape as ``dev/trainer``'s,
so a tuned model is served immediately by ``--method tabpfn`` /
``predictors.performance: tabpfn``.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from coastline.sdk.predictors.performance.data_driven.ml_common import (
    _torch_dtype_category,
    custom_models_dir,
    extract_model_family,
    extract_model_size_bucket,
    get_feature_lists,
    gpu_spec_features,
    llm_spec_features,
)

logger = logging.getLogger(__name__)

# One measured fine-tuning run per row; these columns are the format contract.
REQUIRED_COLUMNS: dict[str, str] = {
    "model_name": "HF model id (best results when known to Kavier's model library)",
    "method": "fine-tuning method (lora, full, ...)",
    "gpu_model": "GPU type (e.g. NVIDIA-A100-SXM4-80GB)",
    "number_nodes": "nodes the run used",
    "number_gpus": "GPUs per node",
    "tokens_per_sample": "sequence length",
    "batch_size": "per-device batch size",
    "dataset_tokens_per_second": "measured throughput — tuning target",
    "train_runtime": "measured runtime in seconds — tuning target",
}
OPTIONAL_COLUMNS: dict[str, str] = {
    "is_valid": "1.0 keeps the row; anything else drops it",
    "torch_dtype": "training dtype (bfloat16, ...); 'unknown' when absent",
    "enable_roce": "0/1 fast-interconnect flag; 'unknown' when absent",
}

MIN_ROWS = 20  # below this, warn that the tuned model is likely poor
TUNABLE_MODELS = ("tabpfn",)


class DatasetFormatError(ValueError):
    """The dataset cannot be tuned on at all (missing columns / no usable rows)."""


def dataset_format_help() -> str:
    """The tuning-dataset contract, printed whenever validation fails."""
    lines = ["A valid tuning dataset is a CSV with one measured fine-tuning run per row and columns:"]
    width = max(len(c) for c in (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS))
    for col, meaning in REQUIRED_COLUMNS.items():
        lines.append(f"  {col:<{width}}  {meaning}")
    for col, meaning in OPTIONAL_COLUMNS.items():
        lines.append(f"  {col:<{width}}  (optional) {meaning}")
    return "\n".join(lines)


def _quality_warnings(clean: pd.DataFrame, dropped: int) -> list[str]:
    """Violated properties of a good tuning dataset, phrased as the property itself."""
    warnings: list[str] = []
    if len(clean) < MIN_ROWS:
        warnings.append(f"at least {MIN_ROWS} valid rows (this dataset has {len(clean)} after filtering)")
    if dropped:
        warnings.append(f"rows with is_valid=1.0 and positive targets ({dropped} row(s) were dropped by these filters)")
    config_cols = [
        "model_name",
        "method",
        "gpu_model",
        "number_nodes",
        "number_gpus",
        "tokens_per_sample",
        "batch_size",
    ]
    if len(clean) and len(clean.drop_duplicates(config_cols)) < 2:
        warnings.append("at least 2 distinct configurations (this dataset has 1)")
    for col, why in (("number_gpus", "GPU-scaling"), ("batch_size", "batch-scaling")):
        if len(clean) and clean[col].nunique() < 2:
            warnings.append(f"multiple {col} values, or the model cannot learn {why} behaviour")
    unknown_models = sorted(
        {str(m) for m in clean["model_name"].unique() if np.isnan(llm_spec_features(m)["llm_n_layers"])}
    )
    if unknown_models:
        warnings.append(
            "model names known to Kavier's library — predictions are refused for unknown models "
            f"(unknown here: {', '.join(unknown_models[:5])}{', ...' if len(unknown_models) > 5 else ''})"
        )
    unknown_gpus = sorted(
        {str(g) for g in clean["gpu_model"].unique() if np.isnan(gpu_spec_features(g)["gpu_fp16_tflops"])}
    )
    if unknown_gpus:
        warnings.append(f"GPU models known to Kavier's library (unknown here: {', '.join(unknown_gpus[:5])})")
    return warnings


def validate_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (usable rows, quality warnings); raise DatasetFormatError when untunable."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetFormatError(
            f"cannot tune: the dataset is missing required column(s): {', '.join(missing)}\n\n{dataset_format_help()}"
        )
    clean = df.copy()
    if "is_valid" in clean.columns:
        clean = clean[pd.to_numeric(clean["is_valid"], errors="coerce") == 1.0]
    for col in REQUIRED_COLUMNS:
        if col in ("model_name", "method", "gpu_model"):
            clean = clean[clean[col].notna()]
        else:
            clean = clean[pd.to_numeric(clean[col], errors="coerce") > 0]
    dropped = len(df) - len(clean)
    if clean.empty:
        raise DatasetFormatError(
            f"cannot tune: no usable rows — all {len(df)} row(s) were dropped "
            f"(is_valid != 1.0, or non-positive/missing values in a required column)\n\n{dataset_format_help()}"
        )
    return clean.reset_index(drop=True), _quality_warnings(clean, dropped)


def _roce_category(v: Any) -> str:
    if v is None or pd.isna(v):
        return "unknown"
    return "1" if float(v) else "0"


def _feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """featv3 feature frame (categoricals as str, numericals as float) from clean rows."""
    cat_features, num_features = get_feature_lists()
    rows = []
    for _, r in df.iterrows():
        feat: dict[str, Any] = {
            "method": str(r["method"]),
            "gpu_model": str(r["gpu_model"]),
            "model_type": extract_model_family(r["model_name"]),
            "torch_dtype": _torch_dtype_category(r.get("torch_dtype")),
            "enable_roce": _roce_category(r.get("enable_roce")),
            "model_size_bucket": extract_model_size_bucket(str(r["model_name"])),
            "number_nodes": float(r["number_nodes"]),
            "number_gpus": float(r["number_gpus"]),
            "total_gpus": float(r["number_nodes"]) * float(r["number_gpus"]),
            "tokens_per_sample": float(r["tokens_per_sample"]),
            "batch_size": float(r["batch_size"]),
        }
        feat.update(llm_spec_features(r["model_name"]))
        feat.update(gpu_spec_features(r["gpu_model"]))
        rows.append(feat)
    X = pd.DataFrame(rows)[cat_features + num_features]
    for c in cat_features:
        X[c] = X[c].astype(str)
    return X, cat_features, num_features


def _as_model_input(X: pd.DataFrame, num_features: list[str]) -> np.ndarray:
    """Mixed object array (str categoricals, float64 numericals) — what the predictor feeds TabPFN."""
    arr = X.values.astype(object)
    for i, col in enumerate(X.columns):
        if col in num_features:
            arr[:, i] = arr[:, i].astype(np.float64)
    return arr


def _mdape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.median(np.abs(predicted - actual) / actual) * 100.0)


def tune(
    data_csv: str,
    *,
    model: str = "tabpfn",
    train_percentage: float = 1.0,
    output: Optional[str] = None,
    seed: int = 42,
    on_step: Optional[Any] = None,
) -> dict[str, Any]:
    """Tune ``model`` on ``data_csv``; return {tune_id, path, rows_*, fit_seconds, metrics, warnings}.

    ``train_percentage=1.0`` uses every valid row (no holdout); below 1.0 the rest
    becomes a test split and MdAPE for both targets is reported. ``on_step`` (a
    ``str -> None`` callable, e.g. ``print``) receives live progress lines so long
    fits are never silent.
    """
    step = on_step or logger.info
    if model not in TUNABLE_MODELS:
        raise ValueError(f"only {TUNABLE_MODELS} can be tuned here; for other models use dev/trainer")
    if not 0.0 < train_percentage <= 1.0:
        raise ValueError(f"--train-percentage must be in (0, 1], got {train_percentage}")
    try:
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
    except ImportError as exc:
        raise RuntimeError(
            "tabpfn is not installed — install the ML extras first: uv sync --extra ml "
            '(or pip install "coastline-recommender[ml]")'
        ) from exc

    raw = pd.read_csv(data_csv, low_memory=False)
    clean, warnings_list = validate_dataset(raw)
    step(f"loaded {data_csv}: {len(raw)} rows -> {len(clean)} valid ({len(raw) - len(clean)} dropped by filters)")
    X, cat_features, num_features = _feature_frame(clean)
    y = clean[["dataset_tokens_per_second", "train_runtime"]].astype(float)
    y_log = np.log1p(y)

    if train_percentage < 1.0:
        from sklearn.model_selection import train_test_split

        X_train, X_test, ylog_train, _, _, y_test = train_test_split(
            X, y_log, y, test_size=1.0 - train_percentage, random_state=seed
        )
    else:
        X_train, X_test, ylog_train, y_test = X, None, y_log, None

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    tune_id = f"{model}-{time.strftime('%Y%m%d-%H%M%S')}"
    # tuned artifacts land in models/custom/ — they shadow the coastline-bundled portfolio
    path = Path(output) if output else custom_models_dir() / f"{model}.pkl"
    step(
        f"tune id {tune_id} · train {len(X_train)} rows / holdout {0 if X_test is None else len(X_test)} rows "
        f"(train-percentage {train_percentage}) · device {device}"
    )
    step(f"output: {path}" + (" (will OVERWRITE the existing artifact)" if path.exists() else " (new file)"))

    # One regressor per target (TabPFN is single-output); v2 weights are the only
    # redistributable ones — see dev/trainer/train_performance_tabpfn.py.
    def make_regressor():
        return TabPFNRegressor.create_default_for_version(
            ModelVersion.V2, device=device, ignore_pretraining_limits=True
        )

    X_train_arr = _as_model_input(X_train, num_features)
    t0 = time.perf_counter()
    step("[1/3] fitting throughput regressor ...")
    model_throughput = make_regressor()
    model_throughput.fit(X_train_arr, ylog_train["dataset_tokens_per_second"].to_numpy())
    step(f"[1/3] done in {time.perf_counter() - t0:.1f}s")
    t1 = time.perf_counter()
    step("[2/3] fitting runtime regressor ...")
    model_runtime = make_regressor()
    model_runtime.fit(X_train_arr, ylog_train["train_runtime"].to_numpy())
    step(f"[2/3] done in {time.perf_counter() - t1:.1f}s")
    fit_seconds = time.perf_counter() - t0

    metrics: dict[str, float] = {}
    if X_test is not None and len(X_test):
        step(f"evaluating holdout ({len(X_test)} rows) ...")
        X_test_arr = _as_model_input(X_test, num_features)
        pred_thr = np.expm1(model_throughput.predict(X_test_arr))
        pred_rt = np.expm1(model_runtime.predict(X_test_arr))
        metrics = {
            "test_mdape_throughput_pct": _mdape(y_test["dataset_tokens_per_second"].to_numpy(), pred_thr),
            "test_mdape_runtime_pct": _mdape(y_test["train_runtime"].to_numpy(), pred_rt),
        }

    step("[3/3] saving artifact ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "model": {"throughput": model_throughput, "runtime": model_runtime},
        "cat_features": cat_features,
        "num_features": num_features,
        "tune_id": tune_id,
        "tuned_on": str(data_csv),
        "train_percentage": train_percentage,
        "test_metrics": metrics,
    }
    with open(path, "wb") as f:
        pickle.dump(artifacts, f)
    step(f"[3/3] wrote {path} ({path.stat().st_size / 1e6:.0f} MB)")

    return {
        "tune_id": tune_id,
        "path": str(path),
        "device": device,
        "rows_train": len(X_train),
        "rows_test": 0 if X_test is None else len(X_test),
        "fit_seconds": round(fit_seconds, 2),
        "metrics": metrics,
        "warnings": warnings_list,
    }
