"""Shared ML inference utilities for data-driven performance predictors."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from coastline.sdk.models.recommendation import Prediction

logger = logging.getLogger(__name__)

PERFORMANCE_MODEL_ARTIFACT_SUFFIX = "_featv3"

BASE_CATEGORICAL = ["method", "gpu_model"]
ENGINEERED_CATEGORICAL = ["model_type", "torch_dtype", "enable_roce", "model_size_bucket"]
BASE_NUMERICAL = ["number_nodes", "number_gpus", "tokens_per_sample", "batch_size", "total_gpus"]

# Must stay in sync with dev/trainer/common.py feature schema (featv3).
# If kavier.sdk.library is absent, spec-dependent predictors return None.
try:
    from kavier.sdk.library import GPU_SPEC_LIBRARY as _GPU_LIB
    from kavier.sdk.library import LLM_SPEC_LIBRARY as _LLM_LIB
except ImportError:
    logger.warning(
        "kavier.sdk.library not importable; deep LLM/GPU specs unavailable, so "
        'data-driven predictors will return None. Install kavier: pip install "kavier>=0.5,<0.6"'
    )
    _LLM_LIB, _GPU_LIB = {}, {}

LLM_SPEC_NUMERICAL = [
    "llm_n_layers",
    "llm_d_model",
    "llm_n_heads",
    "llm_d_head",
    "llm_m_params",
    "llm_active_params",
    "llm_num_experts",
    "llm_active_experts",
]
GPU_SPEC_NUMERICAL = [
    "gpu_fp16_tflops",
    "gpu_mem_bw_gbps",
    "gpu_cores",
    "gpu_mem_gb",
    "gpu_clock_mhz",
    "gpu_tdp_w",
    "gpu_net_bw_gbps",
]
SPEC_NUMERICAL = LLM_SPEC_NUMERICAL + GPU_SPEC_NUMERICAL


def feature_row_has_unknown_specs(row_or_df: Any) -> bool:
    """True if any SPEC_NUMERICAL feature is NaN (model/GPU absent from Kavier library).

    Non-tree sklearn models use this to bail out instead of raising ValueError on NaN.
    Accepts either a feature dict or a single-row DataFrame.
    """
    if isinstance(row_or_df, pd.DataFrame):
        present = [c for c in SPEC_NUMERICAL if c in row_or_df.columns]
        if not present:
            return False
        return bool(row_or_df[present].isna().to_numpy().any())
    return any(pd.isna(row_or_df.get(c)) for c in SPEC_NUMERICAL)


def llm_spec_features(model_name: Any) -> Dict[str, float]:
    s = _LLM_LIB.get(str(model_name))
    if s is None:
        return {k: np.nan for k in LLM_SPEC_NUMERICAL}
    return {
        "llm_n_layers": float(s.n_layers),
        "llm_d_model": float(s.d_model),
        "llm_n_heads": float(s.n_heads),
        "llm_d_head": float(s.d_head),
        "llm_m_params": float(s.m_params),
        "llm_active_params": float(s.active_params),
        # Hardcoded 1.0: Kavier dropped MoE tracking and training had zero MoE rows.
        # Keep for featv3 pickle compatibility; remove on a featv4 retrain.
        "llm_num_experts": 1.0,
        "llm_active_experts": 1.0,
    }


def gpu_spec_features(gpu_model: Any) -> Dict[str, float]:
    s = _GPU_LIB.get(str(gpu_model))
    if s is None:
        return {k: np.nan for k in GPU_SPEC_NUMERICAL}
    return {
        "gpu_fp16_tflops": float(s.fp_16_tensor_core_tflops),
        "gpu_mem_bw_gbps": float(s.bandwidth_bps) / 1e9,
        "gpu_cores": float(s.cores),
        "gpu_mem_gb": float(s.memory_gb),
        "gpu_clock_mhz": float(s.core_max_mhz),
        "gpu_tdp_w": float(s.base_power_w),
        "gpu_net_bw_gbps": float(s.network_bandwidth_gbps),
    }


# Model artifacts are resolved in precedence order:
#   1. PORTFOLIO_DIR/custom/  — user-tuned models (`coastline utils tune` writes here)
#   2. PORTFOLIO_DIR/         — the bundled portfolio (all 10 in a dev checkout)
#   3. the packaged portfolio/ next to this file — fallback when PORTFOLIO_DIR is overridden
# By default PORTFOLIO_DIR *is* the packaged portfolio, so every model (bundled and user-tuned)
# has one home inside the SDK. A read-only (pip) deployment sets PORTFOLIO_DIR to a writable dir.
_BUNDLED_PORTFOLIO_DIR = Path(__file__).resolve().parent / "portfolio"
PORTFOLIO_DIR = Path(os.environ.get("PORTFOLIO_DIR", str(_BUNDLED_PORTFOLIO_DIR)))


def custom_models_dir() -> Path:
    """Where user-tuned artifacts live (highest resolution precedence)."""
    return PORTFOLIO_DIR / "custom"


def _resolve_artifact(*names: str) -> Path:
    """First existing of custom/ > flat PORTFOLIO_DIR > packaged portfolio, trying each name
    spelling per directory; falls back to the custom/ path of the first (canonical) name for a
    clear not-found error."""
    directories = (PORTFOLIO_DIR / "custom", PORTFOLIO_DIR, _BUNDLED_PORTFOLIO_DIR)
    for directory in directories:
        for name in names:
            candidate = directory / name
            if candidate.exists():
                return candidate
    return PORTFOLIO_DIR / "custom" / names[0]


def performance_trained_model_path(model_stem: str) -> Path:
    """Path to a trained sklearn-style pickle (e.g. model_stem='xgboost') — ``<stem>.pkl``,
    with the pre-rename ``performance_<stem>_featv3.pkl`` spelling as a legacy fallback."""
    return _resolve_artifact(f"{model_stem}.pkl", f"performance_{model_stem}{PERFORMANCE_MODEL_ARTIFACT_SUFFIX}.pkl")


def performance_deep_learning_model_dir() -> Path:
    """Directory holding DL weights + artifacts (``deep_learning/``; legacy spelling falls back)."""
    return _resolve_artifact("deep_learning", f"performance_deep_learning{PERFORMANCE_MODEL_ARTIFACT_SUFFIX}")


def extract_model_family(name: str) -> str:
    full = str(name).lower()
    for family in ["llama", "granite", "mistral", "mixtral", "allam"]:
        if family in full:
            return family
    return full.split("/")[0].split("-")[0]


def extract_model_size_bucket(name: str) -> str:
    raw = str(name).strip().lower()
    norm = raw.replace("_", "-").replace(" ", "")

    if "mixtral" in norm:
        m = re.search(r"(\d+)x(\d+)b?", norm)
        if m:
            return f"mixtral{m.group(1)}x{m.group(2)}b"

    # Prefer a "<n>b" token (e.g. 7b); otherwise fall back to a "-<n>" suffix.
    size_b = list(re.finditer(r"(\d+)\s*b\b", norm))
    size_dash = re.search(r"-(\d+)\b", norm)
    if size_b:
        size = float(size_b[-1].group(1))
    elif size_dash:
        size = float(size_dash.group(1))
    else:
        size = None

    family = extract_model_family(name)
    if size is None:
        return f"{family}_unknown"
    if abs(size - int(size)) < 1e-9:
        return f"{family}{int(size)}b"
    return f"{family}{str(size).replace('.', 'p')}b"


def _torch_dtype_category(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return "unknown"
    s = str(v).strip().lower()
    if s in ("", "nan", "none"):
        return "unknown"
    return s


def get_feature_lists():
    """Return categorical and numerical feature names after engineering."""
    cat_features = BASE_CATEGORICAL + ENGINEERED_CATEGORICAL
    num_features = BASE_NUMERICAL + SPEC_NUMERICAL
    return cat_features, num_features


def workload_to_ml_feature_row(workload: Any) -> Dict[str, Any]:
    """Build the ML feature dict from a WorkloadSpec."""
    nn = int(workload.number_of_nodes or 1)
    ng = int(workload.gpus_per_node or 1)
    td = getattr(workload, "torch_dtype", None)
    dtype_str = _torch_dtype_category(td)
    roce = getattr(workload, "enable_roce", None)
    if roce is None:
        roce_str = "unknown"
    else:
        roce_str = "1" if roce else "0"
    feat = {
        "method": str(workload.fine_tuning_method),
        "gpu_model": str(workload.gpu_model),
        "model_type": extract_model_family(workload.llm_model),
        "torch_dtype": dtype_str,
        "enable_roce": roce_str,
        "model_size_bucket": extract_model_size_bucket(str(workload.llm_model)),
        "number_nodes": nn,
        "number_gpus": ng,
        "total_gpus": float(nn * ng),
        "tokens_per_sample": int(workload.tokens_per_sample),
        "batch_size": int(workload.batch_size),
    }
    feat.update(llm_spec_features(workload.llm_model))
    feat.update(gpu_spec_features(workload.gpu_model))
    return feat


def encode_categoricals(row: Dict[str, Any], encoders: Dict[str, Any], cat_features: List[str]) -> pd.DataFrame:
    """Map categoricals through fitted LabelEncoders; unseen values fall back to 'unknown' or 0.

    Used by xgboost, lightgbm, random_forest, svr, knn. CatBoost uses native categoricals.
    """
    X_cat = pd.DataFrame()
    for col in cat_features:
        encoder = encoders[col]
        val = row.get(col, "unknown")
        if val in encoder.classes_:
            X_cat[col] = [encoder.transform([val])[0]]
        elif "unknown" in encoder.classes_:
            X_cat[col] = [encoder.transform(["unknown"])[0]]
        else:
            X_cat[col] = [0]
            logger.warning(f"Unknown value '{val}' for feature '{col}', using fallback")
    return X_cat


def build_encoded_features(
    row: Dict[str, Any],
    encoders: Dict[str, Any],
    cat_features: List[str],
    num_features: List[str],
) -> pd.DataFrame:
    """Assemble the model input row: LabelEncoded categoricals + raw numericals."""
    X_cat = encode_categoricals(row, encoders, cat_features)
    X_num = pd.DataFrame([{f: row[f] for f in num_features}])
    return pd.concat([X_cat.reset_index(drop=True), X_num.reset_index(drop=True)], axis=1)


def invert_log_targets(y_log_pred: Any) -> Tuple[float, Optional[float]]:
    """Invert log1p output into (throughput, runtime_seconds); runtime is None for single-output models."""
    if np.ndim(y_log_pred) > 1:
        throughput = float(np.expm1(y_log_pred[0][0]))
        runtime_seconds = float(np.expm1(y_log_pred[0][1]))
    else:
        throughput = float(np.expm1(y_log_pred[0]))
        runtime_seconds = None
    return throughput, runtime_seconds


def finalize_ml_prediction(workload: Any, *, throughput, runtime_seconds, metadata) -> "Prediction | None":
    """Build a Prediction or return None if throughput is missing/non-finite.

    Negative-finite throughput is clamped to 0; non-finite runtime becomes None.
    (Non-finite values are not clamped to 0 — they'd silently corrupt downstream scoring.)
    """
    if throughput is None or not np.isfinite(throughput):
        return None
    throughput = max(float(throughput), 0.0)
    if runtime_seconds is not None:
        runtime_seconds = float(runtime_seconds)
        runtime_seconds = max(runtime_seconds, 0.0) if np.isfinite(runtime_seconds) else None
    gpus_per_node = int(workload.gpus_per_node or 1)
    number_of_nodes = int(workload.number_of_nodes or 1)
    return Prediction(
        gpus_per_node=gpus_per_node,
        number_of_nodes=number_of_nodes,
        total_gpus=gpus_per_node * number_of_nodes,
        predicted_throughput=throughput,
        predicted_runtime_seconds=runtime_seconds,
        metadata=metadata,
    )
