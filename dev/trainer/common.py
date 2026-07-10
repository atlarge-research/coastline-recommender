"""Shared utilities for performance model training: data loading, feature engineering, preprocessing, metrics."""

import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, QuantileTransformer

BASE_DIR = Path(__file__).parent
# Honour DATA_DIR (Docker/CI); otherwise resolve the shared trace-archive relative to this
# file so lookup never depends on the CWD. The trace-archive sits in the superproject umbrella,
# one level above the coastline repo root; this file lives at dev/trainer/common.py, so
# parents[2] is the repo root and parents[3] the umbrella that holds trace-archive.
DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).resolve().parents[3] / "trace-archive")))
DATA_PATH = DATA_DIR / "profiling-dataset" / "curated_trace.csv"
# Official (re)trained artifacts live in <repo>/models/coastline-bundled/, tracked in git and
# independent of DATA_DIR. User-tuned models go to models/custom/ via `coastline tune`. Honour
# PORTFOLIO_DIR (Docker/CI/pip-install): a pip-installed deployment points this env var at its
# own models dir rather than the absent site-packages/models.
PORTFOLIO_DIR = Path(
    os.environ.get("PORTFOLIO_DIR", str(Path(__file__).resolve().parents[2] / "models" / "coastline-bundled"))
)

# Bump when categorical / numeric schema changes so old pickles are not loaded by mistake.
PERFORMANCE_MODEL_ARTIFACT_SUFFIX = "_featv3"


def performance_trained_model_path(model_stem: str) -> Path:
    """Path to a trained sklearn-style pickle (e.g. model_stem='xgboost')."""
    return PORTFOLIO_DIR / f"performance_{model_stem}{PERFORMANCE_MODEL_ARTIFACT_SUFFIX}.pkl"


def performance_deep_learning_model_dir() -> Path:
    """Directory holding Deep Learning weights + artifacts for the current feature schema."""
    return PORTFOLIO_DIR / f"performance_deep_learning{PERFORMANCE_MODEL_ARTIFACT_SUFFIX}"


SEED = 42
TARGET_COLUMNS = {
    "throughput": "dataset_tokens_per_second",
    "runtime_seconds": "train_runtime",
}

# Train / Val / Test split ratios
# 70% train, ~15% val, 15% test
TEST_SIZE = 0.15
VAL_SIZE = 0.176  # 0.176 of remaining 85% ≈ 15% of total

# Categorical: method, gpu_model, model_type (from llm_model), torch_dtype,
#   enable_roce (0/1/unknown), model_size_bucket (coarse size from llm_model).
# Numerical: number_nodes, number_gpus (ADO CSV / trained artifact schema),
#   tokens_per_sample, batch_size, total_gpus

BASE_CATEGORICAL = ["method", "gpu_model"]
ENGINEERED_CATEGORICAL = ["model_type", "torch_dtype", "enable_roce", "model_size_bucket"]
BASE_NUMERICAL = ["number_nodes", "number_gpus", "tokens_per_sample", "batch_size", "total_gpus"]

# ---------------------------------------------------------------------------
# Feature parity with Kavier: the analytical model consumes deep LLM/GPU specs;
# we attach the SAME specs (sourced from Kavier's libraries) so both predictor
# families see identical inputs. Tuned calibration knobs (mfu_factor,
# calibration_factor) are deliberately excluded — they are not raw inputs.
# ---------------------------------------------------------------------------
# Kavier is installed separately (pip install "kavier>=0.4,<0.5"). Use kavier.sdk.library's
# top-level re-exports (its public surface), not the .llm/.gpu submodules.
try:
    from kavier.sdk.library import GPU_SPEC_LIBRARY as _GPU_LIB
    from kavier.sdk.library import LLM_SPEC_LIBRARY as _LLM_LIB
except ImportError:  # specs unavailable -> features fall back to NaN (median-filled)
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


def llm_spec_features(model_name: Any) -> Dict[str, float]:
    """LLM architecture specs from Kavier's LLM_SPEC_LIBRARY (NaN if unknown)."""
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
        # llm_num_experts / llm_active_experts: hardcoded 1.0 — Kavier dropped MoE; kept for _featv3 pickle compat.
        "llm_num_experts": 1.0,
        "llm_active_experts": 1.0,
    }


def gpu_spec_features(gpu_model: Any) -> Dict[str, float]:
    """GPU hardware specs from Kavier's GPU_SPEC_LIBRARY, raw specs only (NaN if unknown)."""
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


def extract_model_family(name: str) -> str:
    """Extract the base model family (e.g. 'llama3.1-70b' -> 'llama', 'meta-llama/Llama-3.1-8B' -> 'llama')."""
    full = str(name).lower()
    for family in ["llama", "granite", "mistral", "mixtral", "allam"]:
        if family in full:
            return family
    return full.split("/")[0].split("-")[0]


def extract_model_size_bucket(name: str) -> str:
    """Coarse size bucket: llama3.1-70b → llama70b, mixtral-8x7b → mixtral8x7b."""
    raw = str(name).strip().lower()
    norm = raw.replace("_", "-").replace(" ", "")

    if "mixtral" in norm:
        m = re.search(r"(\d+)x(\d+)b?", norm)
        if m:
            return f"mixtral{m.group(1)}x{m.group(2)}b"

    mo_b = list(re.finditer(r"(\d+)\s*b\b", norm))
    mo_plain = (
        float(mo_b[-1].group(1))
        if mo_b
        else (float(re.search(r"-(\d+)\b", norm).group(1)) if re.search(r"-(\d+)\b", norm) else None)
    )

    family = extract_model_family(name)
    if mo_plain is not None:
        v = mo_plain
        if abs(v - int(v)) < 1e-9:
            return f"{family}{int(v)}b"
        sv = str(v).replace(".", "p")
        return f"{family}{sv}b"
    return f"{family}_unknown"


def _enable_roce_category(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return "unknown"
    try:
        return "1" if float(v) != 0.0 else "0"
    except (TypeError, ValueError):
        return "unknown"


def _torch_dtype_category(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return "unknown"
    s = str(v).strip().lower()
    if s in ("", "nan", "none"):
        return "unknown"
    return s


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns: model_type, model_size_bucket, torch_dtype, enable_roce, total_gpus, and Kavier specs."""
    df = df.copy()
    if "model_name" in df.columns:
        mn = df["model_name"].astype(str)
        df["model_type"] = mn.apply(extract_model_family)
        df["model_size_bucket"] = mn.apply(extract_model_size_bucket)
    else:
        df["model_type"] = "unknown"
        df["model_size_bucket"] = "unknown"

    nodes = pd.to_numeric(df["number_nodes"], errors="coerce").fillna(1).clip(lower=1)
    gpn = pd.to_numeric(df["number_gpus"], errors="coerce").fillna(1).clip(lower=1)
    df["total_gpus"] = (nodes * gpn).astype(float)

    if "torch_dtype" in df.columns:
        df["torch_dtype"] = df["torch_dtype"].apply(_torch_dtype_category)
    else:
        df["torch_dtype"] = "unknown"

    if "enable_roce" in df.columns:
        df["enable_roce"] = df["enable_roce"].apply(_enable_roce_category)
    else:
        df["enable_roce"] = "unknown"

    # Feature parity: attach Kavier's deep LLM + GPU specs as numeric columns.
    mn = df["model_name"].astype(str) if "model_name" in df.columns else pd.Series([""] * len(df), index=df.index)
    gm = df["gpu_model"].astype(str) if "gpu_model" in df.columns else pd.Series([""] * len(df), index=df.index)
    llm_df = mn.map(llm_spec_features).apply(pd.Series)
    gpu_df = gm.map(gpu_spec_features).apply(pd.Series)
    df = pd.concat([df, llm_df, gpu_df], axis=1)

    return df


def get_feature_lists() -> Tuple[List[str], List[str]]:
    """Return (cat_features, num_features) for the featv3 schema."""
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


def curated_series_to_ml_feature_row(row: pd.Series) -> Dict[str, Any]:
    """Build ML features from one row of the curated ADO CSV."""
    nn = int(row.get("number_nodes", 1) or 1)
    ng = int(row.get("number_gpus", 1) or 1)
    mn = row.get("model_name", "unknown")
    feat = {
        "method": str(row["method"]),
        "gpu_model": str(row["gpu_model"]),
        "model_type": extract_model_family(str(mn)),
        "torch_dtype": _torch_dtype_category(row.get("torch_dtype")),
        "enable_roce": _enable_roce_category(row.get("enable_roce")),
        "model_size_bucket": extract_model_size_bucket(str(mn)),
        "number_nodes": nn,
        "number_gpus": ng,
        "total_gpus": float(nn * ng),
        "tokens_per_sample": int(row["tokens_per_sample"]),
        "batch_size": int(row["batch_size"]),
    }
    feat.update(llm_spec_features(mn))
    feat.update(gpu_spec_features(row["gpu_model"]))
    return feat


def load_and_preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Load and preprocess the curated CSV; return (X_cat, X_num, y, cat_features, num_features)."""
    df = pd.read_csv(DATA_PATH)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected pandas DataFrame from curated CSV")

    if "is_valid" in df.columns:
        df = df.loc[df["is_valid"] == 1.0].copy()

    required_targets = list(TARGET_COLUMNS.values())
    for target_col in required_targets:
        if target_col not in df.columns:
            raise ValueError(f"Missing required target column: {target_col}")

    throughput_col = TARGET_COLUMNS["throughput"]
    runtime_col = TARGET_COLUMNS["runtime_seconds"]
    valid_mask = df[throughput_col].notna() & (df[throughput_col] > 0) & df[runtime_col].notna() & (df[runtime_col] > 0)
    df = df.loc[valid_mask].copy()

    df = engineer_features(df)

    cat_features, num_features = get_feature_lists()

    available_cat = [f for f in cat_features if f in df.columns]
    available_num = [f for f in num_features if f in df.columns]

    X_cat = pd.DataFrame(df.loc[:, available_cat]).copy()
    X_num = pd.DataFrame(df.loc[:, available_num]).copy()
    y = pd.DataFrame(df.loc[:, list(TARGET_COLUMNS.values())]).copy()

    for col in X_cat.columns:
        X_cat[col] = pd.Series(X_cat[col]).fillna("unknown")
    for col in X_num.columns:
        X_num[col] = pd.to_numeric(X_num[col], errors="coerce")
        X_num[col] = X_num[col].fillna(X_num[col].median())

    return X_cat, X_num, y, available_cat, available_num


def split_data(*arrays, test_size=TEST_SIZE, val_size=VAL_SIZE, seed=SEED):
    """Split arrays into (train_splits, val_splits, test_splits); val_size is a fraction of the post-test remainder."""
    temp_and_test = train_test_split(*arrays, test_size=test_size, random_state=seed)
    # train_test_split interleaves: [temp_0, test_0, temp_1, test_1, ...]
    n = len(arrays)
    temps = [temp_and_test[2 * i] for i in range(n)]
    tests = [temp_and_test[2 * i + 1] for i in range(n)]

    train_and_val = train_test_split(*temps, test_size=val_size, random_state=seed)
    trains = [train_and_val[2 * i] for i in range(n)]
    vals = [train_and_val[2 * i + 1] for i in range(n)]

    return tuple(trains), tuple(vals), tuple(tests)


def as_dataframes(*objs: Any) -> Tuple[pd.DataFrame, ...]:
    """Type-narrow split outputs to DataFrame (runtime no-op; aids static typing)."""
    return tuple(cast(pd.DataFrame, o) for o in objs)


def encode_categorical_features(
    X_cat_train: pd.DataFrame,
    X_cat_val: pd.DataFrame,
    X_cat_test: pd.DataFrame,
    return_numpy: bool = False,
) -> Tuple:
    """Encode categoricals via LabelEncoder (with explicit 'unknown' class); returns (enc_train, enc_val, enc_test, encoders, vocab_sizes)."""
    encoders: Dict[str, LabelEncoder] = {}
    vocab_sizes: Dict[str, int] = {}

    if return_numpy:
        X_train_enc = np.zeros((len(X_cat_train), len(X_cat_train.columns)), dtype=int)
        X_val_enc = np.zeros((len(X_cat_val), len(X_cat_val.columns)), dtype=int)
        X_test_enc = np.zeros((len(X_cat_test), len(X_cat_test.columns)), dtype=int)
    else:
        X_train_enc = pd.DataFrame()
        X_val_enc = pd.DataFrame()
        X_test_enc = pd.DataFrame()

    for i, col in enumerate(X_cat_train.columns):
        encoder = LabelEncoder()
        uniq = pd.Series(X_cat_train[col]).astype(str).unique().tolist()
        train_vals = list(uniq)
        if "unknown" not in train_vals:
            train_vals.append("unknown")
        encoder.fit(train_vals)

        cls_set = {str(c) for c in encoder.classes_}
        unknown_idx = int(encoder.transform(["unknown"])[0])

        def _safe_transform(values):
            return [
                int(encoder.transform([str(v)])[0]) if str(v) in cls_set else unknown_idx
                for v in pd.Series(values).astype(str)
            ]

        train_ids = encoder.transform(pd.Series(X_cat_train[col]).astype(str))
        train_encoded = [int(v) for v in train_ids]

        val_encoded = _safe_transform(X_cat_val[col])
        test_encoded = _safe_transform(X_cat_test[col])

        if return_numpy:
            X_train_enc[:, i] = train_encoded
            X_val_enc[:, i] = val_encoded
            X_test_enc[:, i] = test_encoded
        else:
            X_train_enc[col] = train_encoded
            X_val_enc[col] = val_encoded
            X_test_enc[col] = test_encoded

        encoders[col] = encoder
        vocab_sizes[col] = len(encoder.classes_)

    return X_train_enc, X_val_enc, X_test_enc, encoders, vocab_sizes


def scale_numerical_features(
    X_num_train: pd.DataFrame,
    X_num_val: pd.DataFrame,
    X_num_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, QuantileTransformer]:
    """Scale numerics via QuantileTransformer (normal output); returns (scaled_train, scaled_val, scaled_test, scaler)."""
    scaler = QuantileTransformer(output_distribution="normal", random_state=SEED)

    cols = X_num_train.columns
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_num_train), columns=cols)
    X_val_scaled = pd.DataFrame(scaler.transform(X_num_val), columns=cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_num_test), columns=cols)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_log_true: Optional[np.ndarray] = None,
    y_log_pred: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute prediction metrics in original space (always) and log space (when log inputs provided)."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    median_ae = median_absolute_error(y_true, y_pred)
    max_err = float(np.max(np.abs(y_true - y_pred)))

    # Percentage-based metrics (filter zeros)
    mask = y_true > 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    mdape = float(np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    within_20_pct = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) <= 0.20) * 100)

    result = {
        "original_space": {
            "mae": float(mae),
            "rmse": rmse,
            "r2": float(r2),
            "mape": mape,
            "mdape": mdape,
            "median_ae": float(median_ae),
            "max_error": max_err,
            "within_20_pct": within_20_pct,
        }
    }

    if y_log_true is not None and y_log_pred is not None:
        y_log_true = np.asarray(y_log_true, dtype=float).reshape(-1)
        y_log_pred = np.asarray(y_log_pred, dtype=float).reshape(-1)
        result["log_space"] = {
            "mae": float(mean_absolute_error(y_log_true, y_log_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_log_true, y_log_pred))),
            "r2": float(r2_score(y_log_true, y_log_pred)),
        }

    return result


# Conditional artifact save: retain the model with the best test throughput MdAPE.


def training_force_save() -> bool:
    """True when ``TRAIN_FORCE_SAVE`` is set (1 / true / yes): always overwrite artifacts."""
    return os.environ.get("TRAIN_FORCE_SAVE", "").strip().lower() in ("1", "true", "yes")


def get_stored_test_throughput_mdape(artifact_path: Path) -> Optional[float]:
    """Read the stored test throughput MdAPE from a portfolio pickle, or None if absent/unreadable."""
    artifact_path = Path(artifact_path)
    if not artifact_path.is_file():
        return None
    try:
        with open(artifact_path, "rb") as f:
            blob = pickle.load(f)
    except Exception:
        return None
    if not isinstance(blob, dict):
        return None
    tm = blob.get("test_metrics")
    if isinstance(tm, dict) and "original_space" in tm:
        try:
            return float(tm["original_space"]["mdape"])
        except (KeyError, TypeError, ValueError):
            pass
    by_tgt = blob.get("test_metrics_by_target") or {}
    tput = by_tgt.get("throughput")
    if isinstance(tput, dict) and "original_space" in tput:
        try:
            return float(tput["original_space"]["mdape"])
        except (KeyError, TypeError, ValueError):
            pass
    return None


def save_pickled_artifact_if_better(
    artifact_path: Path,
    artifacts: Any,
    new_throughput_mdape: float,
    *,
    force: bool = False,
) -> Tuple[bool, str]:
    """Pickle artifacts only when new test throughput MdAPE is strictly better than on disk; returns (saved, message)."""
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    force = force or training_force_save()
    old = None if force else get_stored_test_throughput_mdape(artifact_path)
    if old is not None and new_throughput_mdape >= old:
        return False, (
            f"Kept existing model: on-disk test throughput MdAPE {old:.4f}% "
            f"≤ new {new_throughput_mdape:.4f}%. Not writing {artifact_path}"
        )
    with open(artifact_path, "wb") as f:
        pickle.dump(artifacts, f)
    if old is None:
        return True, f"Saved model to {artifact_path} (no comparable prior MdAPE on disk)."
    return True, (f"Replaced model at {artifact_path}: test throughput MdAPE {old:.4f}% → {new_throughput_mdape:.4f}%")


def save_deep_learning_bundle_if_better(
    model_dir: Path,
    *,
    new_throughput_mdape: float,
    torch_save_dict: dict,
    sklearn_artifacts: dict,
    force: bool = False,
) -> Tuple[bool, str]:
    """Save DL .pth + .pkl bundle atomically only when MdAPE improves; returns (saved, message)."""
    import torch

    model_dir = Path(model_dir)
    model_path = model_dir / "performance_deep_learning.pth"
    artifacts_path = model_dir / "performance_deep_learning_artifacts.pkl"
    force = force or training_force_save()
    old = None if force else get_stored_test_throughput_mdape(artifacts_path)
    if old is not None and new_throughput_mdape >= old:
        return False, (
            f"Kept existing Deep Learning bundle: on-disk test throughput MdAPE {old:.4f}% "
            f"≤ new {new_throughput_mdape:.4f}%. Not writing {model_dir}"
        )
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch_save_dict, model_path)
    with open(artifacts_path, "wb") as f:
        pickle.dump(sklearn_artifacts, f)
    if old is None:
        return True, f"Saved Deep Learning bundle under {model_dir} (no prior MdAPE on disk)."
    return True, (
        f"Replaced Deep Learning bundle under {model_dir}: test throughput MdAPE "
        f"{old:.4f}% → {new_throughput_mdape:.4f}%"
    )


def print_metrics(metrics: Dict, dataset_name: str = "", unit: str = "tokens/sec") -> None:
    """Pretty-print training metrics."""
    if dataset_name:
        print(f"\n{'=' * 70}")
        print(f"{dataset_name.upper()} METRICS")
        print(f"{'=' * 70}")

    orig = metrics["original_space"]

    print("\n📊 Original Space:")
    print(f"  MAE:           {orig['mae']:>12,.2f} {unit}")
    print(f"  RMSE:          {orig['rmse']:>12,.2f} {unit}")
    print(f"  R²:            {orig['r2']:>12.4f}")
    print(f"  Max Error:     {orig['max_error']:>12,.2f} {unit}")
    print(f"  Median AE:     {orig['median_ae']:>12,.2f} {unit}")

    print("\n🎯 Key Performance Indicators:")
    print(f"  MdAPE:         {orig['mdape']:>12.2f}%")
    print(f"  MAPE:          {orig['mape']:>12.2f}%")
    print(f"  Within 20%:    {orig['within_20_pct']:>12.1f}%")

    if "log_space" in metrics:
        log = metrics["log_space"]
        print("\n📈 Log Space:")
        print(f"  MAE:           {log['mae']:>12.4f}")
        print(f"  RMSE:          {log['rmse']:>12.4f}")
        print(f"  R²:            {log['r2']:>12.4f}")


def transform_targets(y: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p to all target columns."""
    return pd.DataFrame(
        np.log1p(y.to_numpy(dtype=float)),
        columns=y.columns,
        index=y.index,
    )


def inverse_transform_targets(y_log: Any) -> np.ndarray:
    """Transform model outputs from log space back to original space."""
    return np.expm1(np.asarray(y_log, dtype=float))


def get_target_column_names() -> List[str]:
    """Return target column names in stable order."""
    return list(TARGET_COLUMNS.values())


def get_primary_target_name() -> str:
    """Return the primary user-facing target name."""
    return TARGET_COLUMNS["throughput"]
