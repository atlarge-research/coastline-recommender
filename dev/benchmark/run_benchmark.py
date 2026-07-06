#!/usr/bin/env python3
"""Unified benchmarking suite: evaluate all 12 predictors on throughput + latency (MdAPE, ms/100, Within-20%)."""

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# dev/benchmark -> parents[2] == the superproject umbrella (siblings: trace-archive/, kavier/).
BENCHMARKS_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARKS_DIR.parents[2]
KAVIER_SRC = REPO_ROOT / "kavier" / "src"

from trainer.common import (  # noqa: E402
    SEED,
    curated_series_to_ml_feature_row,
    get_feature_lists,
    load_and_preprocess_data,
    performance_trained_model_path,
    split_data,
)

from benchmark.metrics import compute_metrics, ms_per_100_predictions, throughput_to_latency  # noqa: E402
from coastline.sdk.models.context import Constraints, SystemContext  # noqa: E402
from coastline.sdk.models.workload import WorkloadSpec  # noqa: E402

# Predictors are imported lazily inside per-model subprocesses so native ML backends
# never co-load — co-loading several segfaults on macOS (duplicate OpenMP runtimes).


CONTEXT = SystemContext(
    available_gpu_models=["NVIDIA-A100-SXM4-80GB"],
    max_gpus=128,
    gpu_memory={"NVIDIA-A100-SXM4-80GB": 80},
    constraints=Constraints(max_gpus=128, gpus_per_node=8, max_nodes=16),
)


def prepare_ml_data(max_gpus: Optional[int] = None) -> dict:
    """Load curated data and return the test split; optionally filter to ``max_gpus`` total GPUs."""
    X_cat, X_num, y_df, _, _ = load_and_preprocess_data()
    y_throughput = y_df["dataset_tokens_per_second"].values
    y_runtime = y_df["train_runtime"].values

    # full_df must apply the same filters/index as load_and_preprocess_data().
    data_dir = Path(os.environ.get("DATA_DIR", str(REPO_ROOT / "trace-archive")))
    full_df = pd.read_csv(data_dir / "profiling-dataset" / "curated_trace.csv")
    if "is_valid" in full_df.columns:
        full_df = full_df.loc[full_df["is_valid"] == 1.0].copy()
    valid_mask = (
        full_df["dataset_tokens_per_second"].notna()
        & (full_df["dataset_tokens_per_second"] > 0)
        & full_df["train_runtime"].notna()
        & (full_df["train_runtime"] > 0)
    )
    full_df = full_df.loc[valid_mask].copy()

    # Deterministic 70/15/15 split — indices must align with load_and_preprocess_data().
    # Only the test split is benchmarked; train/val are discarded here.
    _train, _val, (X_cat_test, X_num_test, y_thr_test, y_rt_test, full_test) = split_data(
        X_cat,
        X_num,
        y_throughput,
        y_runtime,
        full_df,
    )

    if max_gpus is not None:
        total = pd.to_numeric(full_test["number_gpus"], errors="coerce").fillna(1) * pd.to_numeric(
            full_test["number_nodes"], errors="coerce"
        ).fillna(1)
        keep = total <= max_gpus
        idx = np.where(keep.values)[0]
        X_cat_test = (
            X_cat_test.iloc[idx].reset_index(drop=True) if isinstance(X_cat_test, pd.DataFrame) else X_cat_test[idx]
        )
        X_num_test = (
            X_num_test.iloc[idx].reset_index(drop=True) if isinstance(X_num_test, pd.DataFrame) else X_num_test[idx]
        )
        y_thr_test = y_thr_test[idx]
        y_rt_test = y_rt_test[idx]
        full_test = full_test.iloc[idx].reset_index(drop=True)

    return {
        "X_cat_test": X_cat_test,
        "X_num_test": X_num_test,
        "y_throughput_test": y_thr_test,
        "y_runtime_test": y_rt_test,
        "full_test": full_test,
    }


def evaluate_kavier(ml_data: dict) -> dict:
    """Evaluate Kavier on the shared test split (live engine only — no CSV fallback; stale fallbacks silently mis-reported MdAPE)."""
    full_test = ml_data["full_test"]
    y_true = np.asarray(ml_data["y_throughput_test"], dtype=np.float64)

    from kavier.sdk.training.core.engine import simulate_training_step  # noqa: PLC0415

    predictions = []
    failures = []
    predict_time_s = 0.0
    for _idx in full_test.index:
        row = full_test.loc[_idx]
        t_pred = time.perf_counter()
        try:
            out = simulate_training_step(
                model_name=str(row["model_name"]),
                gpu_model=str(row["gpu_model"]),
                tokens_per_sample=int(row["tokens_per_sample"]),
                batch_size=int(row["batch_size"]),
                method=str(row["method"]),
                num_gpus=int(row["number_gpus"]) * int(row["number_nodes"]),
                num_nodes=int(row["number_nodes"]),
            )
            v = float(out["tokens_per_second"])
        except Exception as e:
            failures.append((_idx, repr(e)))
            v = np.nan
        predict_time_s += time.perf_counter() - t_pred
        predictions.append(v)

    if failures:
        raise RuntimeError(
            f"Kavier failed to simulate {len(failures)}/{len(predictions)} test rows; "
            f"refusing to report a partial MdAPE. First failure: row {failures[0][0]}: "
            f"{failures[0][1]}"
        )

    return {
        "y_true": y_true,
        "y_pred": np.array(predictions, dtype=np.float64),
        "meta": _build_meta(full_test),
        "n": len(y_true),
        "predict_time_s": predict_time_s,
    }


def _build_meta(full_test: pd.DataFrame) -> pd.DataFrame:
    """Build the meta DataFrame with total_gpus_used (shared by all evaluators)."""
    meta = full_test[["batch_size", "tokens_per_sample", "number_gpus", "number_nodes"]].copy()
    meta["total_gpus_used"] = (
        pd.to_numeric(meta["number_gpus"], errors="coerce").fillna(1)
        * pd.to_numeric(meta["number_nodes"], errors="coerce").fillna(1)
    ).astype(float)
    return meta


def evaluate_tabpfn_batch(data: dict) -> dict:
    """Evaluate TabPFN via batch prediction (per-row predict() is ~9s/row; batch is far faster)."""
    import pickle

    model_path = performance_trained_model_path("tabpfn")
    if not model_path.exists():
        raise FileNotFoundError(f"TabPFN model not found at {model_path}")

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    from coastline.sdk.predictors.performance.data_driven.tabpfn_predictor import (
        _tabpfn_regressor_compat,
    )

    model = model_data["model"] if isinstance(model_data, dict) and "model" in model_data else model_data

    cat_cols, num_cols = get_feature_lists()
    feat_columns = list(cat_cols) + list(num_cols)

    full_test = data["full_test"]
    y_true = data["y_throughput_test"]

    rows = [curated_series_to_ml_feature_row(full_test.loc[idx]) for idx in full_test.index]
    X_batch = pd.DataFrame(rows)[feat_columns]
    for col in cat_cols:
        X_batch[col] = X_batch[col].astype(str)
    X_array = X_batch.values.astype(object)
    for i, col in enumerate(X_batch.columns):
        if col in num_cols:
            X_array[:, i] = X_array[:, i].astype(np.float64)

    t_pred = time.perf_counter()
    if isinstance(model, dict) and "throughput" in model:
        _tabpfn_regressor_compat(model["throughput"])
        if "runtime" in model:
            _tabpfn_regressor_compat(model["runtime"])
        y_log_pred = model["throughput"].predict(X_array)
    else:
        _tabpfn_regressor_compat(model)
        y_log_pred = model.predict(X_array)
        if np.ndim(y_log_pred) > 1:
            y_log_pred = y_log_pred[:, 0]
    predict_time_s = time.perf_counter() - t_pred

    return {
        "y_true": y_true,
        "y_pred": np.maximum(np.expm1(y_log_pred), 0.0),
        "meta": _build_meta(full_test),
        "n": len(y_true),
        "predict_time_s": predict_time_s,
    }


def evaluate_ml_predictor(predictor, data: dict) -> dict:
    """Evaluate one ML predictor row-by-row on the test split."""
    full_test = data["full_test"]
    y_true_throughput = data["y_throughput_test"]
    y_true_runtime = data["y_runtime_test"]

    predictions_throughput = []
    predictions_runtime = []
    predict_time_s = 0.0

    for idx in full_test.index:
        row = full_test.loc[idx]
        td = row.get("torch_dtype")
        td_s = None if pd.isna(td) else str(td).strip()
        roce = row.get("enable_roce")
        roce_b = None
        if not pd.isna(roce):
            try:
                roce_b = bool(float(roce) != 0.0)
            except (TypeError, ValueError):
                roce_b = None
        workload = WorkloadSpec(
            fine_tuning_method=str(row["method"]),
            gpu_model=str(row["gpu_model"]),
            llm_model=str(row.get("model_name", "unknown")),
            batch_size=int(row["batch_size"]),
            tokens_per_sample=int(row["tokens_per_sample"]),
            gpus_per_node=int(row.get("number_gpus", 1)),
            number_of_nodes=int(row.get("number_nodes", 1)),
            torch_dtype=td_s,
            enable_roce=roce_b,
        )
        t_pred = time.perf_counter()
        try:
            prediction = predictor.predict(workload, CONTEXT)
            if prediction and prediction.predicted_throughput is not None:
                predictions_throughput.append(prediction.predicted_throughput)
            else:
                predictions_throughput.append(0.0)

            if prediction and getattr(prediction, "predicted_runtime_seconds", None) is not None:
                predictions_runtime.append(prediction.predicted_runtime_seconds)
            else:
                predictions_runtime.append(np.nan)
        except Exception:
            predictions_throughput.append(0.0)
            predictions_runtime.append(np.nan)
        finally:
            predict_time_s += time.perf_counter() - t_pred

    return {
        "y_true": y_true_throughput,
        "y_pred": np.array(predictions_throughput),
        "y_true_runtime": y_true_runtime,
        "y_pred_runtime": np.array(predictions_runtime),
        "meta": _build_meta(full_test),
        "n": len(y_true_throughput),
        "predict_time_s": predict_time_s,
    }


COL_W = 78

MODEL_IDS = {
    "Kavier": "PA1",
    "RandomForest": "PD1",
    "XGBoost": "PD2",
    "LightGBM": "PD3",
    "CatBoost": "PD4",
    "BayesianRidge": "PD5",
    "SVR": "PD6",
    "KNN": "PD7",
    "GaussianProcess": "PD8",
    "DeepLearning": "PD9",
    "TabPFN": "PD10",
    "CacheLookup": "PR1",
}

HEADER = f"  {'ID':<5s} {'Model':<20s} {'Type':<9s} {'n':>5s}  {'MdAPE':>7s}  {'ms/100':>10s}  {'W/in 20%':>8s}"


def _fmt_float(x, default: str, fmt: str) -> str:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(xf):
        return default
    return format(xf, fmt)


def _fmt_ms_per_100(ms: Optional[float]) -> str:
    try:
        x = float(ms)  # type: ignore[arg-type]  # None raises TypeError -> N/A
    except (TypeError, ValueError):
        return "       N/A"
    if not np.isfinite(x):
        return "       N/A"
    if abs(x - round(x)) < 0.05:
        return f"{int(round(x)):>10d}"
    return f"{x:>10.1f}"


def _fmt_row(name: str, mtype: str, m: dict, *, ms_per_100: Optional[float] = None) -> str:
    mid = MODEL_IDS.get(name, "?")
    if m.get("status") and m["status"] != "OK":
        return f"  {mid:<5s} {name:<20s} {mtype:<9s}   -- {m['status']}"
    n_s = _fmt_float(m.get("n"), "   --", ".0f").rjust(5)
    if n_s.strip() == "--":
        n_s = "   --"
    return (
        f"  {mid:<5s} {name:<20s} {mtype:<9s} {n_s}  {_fmt_float(m.get('mdape'), '   N/A', '>6.1f')}%"
        f"  {_fmt_ms_per_100(ms_per_100)}  {_fmt_float(m.get('within_20'), '    N/A', '>7.1f')}%"
    )


def print_results(results: dict):
    """Print throughput and latency tables."""
    sep = "  " + "-" * (COL_W - 2)
    print("\n" + "=" * COL_W)
    print("  THROUGHPUT BENCHMARK  (tokens/sec)")
    print("=" * COL_W)
    print(HEADER)
    print(sep)
    for name, r in results.items():
        print(_fmt_row(name, r["type"], r["throughput"], ms_per_100=r.get("ms_per_100")))
    print("=" * COL_W)

    print()
    print("=" * COL_W)
    print("  LATENCY BENCHMARK  (seconds per step)")
    print("=" * COL_W)
    print(HEADER)
    print(sep)
    for name, r in results.items():
        print(_fmt_row(name, r["type"], r["latency"], ms_per_100=r.get("ms_per_100")))
    print("=" * COL_W)
    print()


# Display name -> (module, predictor class). Loaded lazily per subprocess; TabPFN uses batch prediction.
_ML_MODELS = {
    "RandomForest": (
        "coastline.sdk.predictors.performance.data_driven.random_forest_predictor",
        "RandomForestPredictor",
    ),
    "SVR": ("coastline.sdk.predictors.performance.data_driven.svr_predictor", "SVRPredictor"),
    "KNN": ("coastline.sdk.predictors.performance.data_driven.knn_predictor", "KNNPredictor"),
    "CatBoost": ("coastline.sdk.predictors.performance.data_driven.catboost_predictor", "CatBoostPredictor"),
    "XGBoost": ("coastline.sdk.predictors.performance.data_driven.xgboost_predictor", "XGBoostPredictor"),
    "LightGBM": ("coastline.sdk.predictors.performance.data_driven.lightgbm_predictor", "LightGBMPredictor"),
    "GaussianProcess": (
        "coastline.sdk.predictors.performance.data_driven.gaussian_process_predictor",
        "GaussianProcessPredictor",
    ),
    "BayesianRidge": (
        "coastline.sdk.predictors.performance.data_driven.bayesian_ridge_predictor",
        "BayesianRidgePredictor",
    ),
    "DeepLearning": (
        "coastline.sdk.predictors.performance.data_driven.deep_learning_predictor",
        "DeepLearningPredictor",
    ),
    "CacheLookup": ("coastline.sdk.predictors.performance.retrieval.cache_predictor", "RetrievalPredictor"),
}

_RESULT_MARKER = "__BENCH_RESULT__"


def _json_default(o):
    """Keep numpy scalars numeric (not stringified) when serializing worker results."""
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    return str(o)


def _evaluate_one_entry(name: str, max_gpus: Optional[int]) -> dict:
    """Evaluate one model and return its results-dict entry (runs in an isolated subprocess)."""
    ml_data = prepare_ml_data(max_gpus=max_gpus)
    mtype = "retrieval" if name == "CacheLookup" else "ML"
    try:
        if name == "TabPFN":
            raw = evaluate_tabpfn_batch(ml_data)
        else:
            import importlib

            module, cls = _ML_MODELS[name]
            predictor = getattr(importlib.import_module(module), cls)()
            raw = evaluate_ml_predictor(predictor, ml_data)
        thr_m, lat_m = _throughput_and_latency_metrics(raw)
        return _result_entry(mtype, raw, thr_m, lat_m)
    except Exception as e:
        return _error_entry(mtype, e)


def _run_model_subprocess(name: str, max_gpus: Optional[int]) -> dict:
    """Spawn a child process to evaluate one model so its native backend loads alone."""
    cmd = [sys.executable, "-m", "benchmark.run_benchmark", "--one", name]
    if max_gpus is not None:
        cmd += ["--max-gpus", str(max_gpus)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    for ln in proc.stdout.splitlines():
        if ln.startswith(_RESULT_MARKER):
            return json.loads(ln[len(_RESULT_MARKER) :])
    return _error_entry(
        "retrieval" if name == "CacheLookup" else "ML",
        RuntimeError(f"worker failed (rc={proc.returncode})"),
    )


def _apply_metric_status(m: dict) -> None:
    """Set ``status`` to OK iff the row count n is a positive, finite number."""
    try:
        n = float(m.get("n"))  # type: ignore[arg-type]
        ok = n > 0 and np.isfinite(n)
    except (TypeError, ValueError):
        ok = False
    m["status"] = "OK" if ok else "Error: no valid predictions"


def _throughput_and_latency_metrics(raw: dict) -> tuple[dict, dict]:
    """Compute throughput and latency metrics from a raw evaluation result."""
    thr_m = compute_metrics(raw["y_true"], raw["y_pred"])
    _apply_metric_status(thr_m)

    meta = raw["meta"].reset_index(drop=True)
    bs = meta["batch_size"].values
    tps = meta["tokens_per_sample"].values
    gpus = meta["total_gpus_used"].values
    lat_m = compute_metrics(
        throughput_to_latency(raw["y_true"], bs, tps, gpus),
        throughput_to_latency(raw["y_pred"], bs, tps, gpus),
    )
    _apply_metric_status(lat_m)
    return thr_m, lat_m


def _result_entry(mtype: str, raw: dict, thr_m: dict, lat_m: dict) -> dict:
    """Build per-model results dict including inference timing."""
    ms100 = ms_per_100_predictions(raw.get("predict_time_s"), raw.get("n") or 0)
    return {
        "type": mtype,
        "throughput": thr_m,
        "latency": lat_m,
        "ms_per_100": ms100,
    }


def _error_entry(mtype: str, exc: Exception) -> dict:
    """Results dict entry for a model whose evaluation raised."""
    err = {"status": f"Error: {str(exc)[:50]}"}
    return {"type": mtype, "throughput": err, "latency": err, "ms_per_100": None}


def _eval_kavier_entry(ml_data: dict, label: str = "[1/1]") -> dict:
    """Evaluate Kavier and return a results dict entry with throughput + latency metrics."""
    print(f"  {label} Evaluating Kavier (physics-based)...", end=" ", flush=True)
    t0 = time.time()
    try:
        raw = evaluate_kavier(ml_data)
        thr_m, lat_m = _throughput_and_latency_metrics(raw)
        print(f"OK ({time.time() - t0:.1f}s)")
        return _result_entry("physics", raw, thr_m, lat_m)
    except Exception as e:
        print(f"FAILED: {e}")
        return _error_entry("physics", e)


def _load_test_data(title: str, max_gpus: Optional[int] = None) -> dict:
    """Print banner, load ML test split, return ml_data."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    gpu_note = f", ≤{max_gpus} GPUs" if max_gpus else ""
    print(f"\n  Loading ML test split (seed={SEED}, 15% holdout; shared with Kavier{gpu_note})...")
    ml_data = prepare_ml_data(max_gpus=max_gpus)
    print(f"  Test samples: {len(ml_data['y_throughput_test'])}\n")
    return ml_data


def _finalize(results: dict, results_csv: Optional[Path], output_json: Optional[str]) -> dict:
    """Print the tables, write the CSV, and optionally dump JSON."""
    print_results(results)
    save_results_csv(results, csv_path=results_csv)
    if output_json:
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Results saved to {output_json}\n")
    return results


def run_all(
    output_json: str = None,
    max_gpus: Optional[int] = None,
    results_csv: Optional[Path] = None,
):
    ml_data = _load_test_data(
        "UNIFIED BENCHMARKING SUITE\n  Models: 1 physics-based + 10 data-driven + 1 retrieval = 12 total",
        max_gpus,
    )

    results = {"Kavier": _eval_kavier_entry(ml_data, "[1/12]")}
    names = list(_ML_MODELS) + ["TabPFN"]  # each evaluated in its own subprocess

    for i, name in enumerate(names, start=2):
        print(f"  [{i}/12] Evaluating {name}...", end=" ", flush=True)
        t0 = time.time()
        entry = _run_model_subprocess(name, max_gpus)
        results[name] = entry
        status = entry.get("throughput", {}).get("status", "OK")
        print(f"OK ({time.time() - t0:.1f}s)" if status == "OK" else f"FAILED: {status}")

    return _finalize(results, results_csv, output_json)


def save_results_csv(results: dict, csv_path: Optional[Path] = None):
    """Save results to CSV in benchmarks/results/."""
    results_dir = BENCHMARKS_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    if csv_path is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        csv_path = results_dir / f"{timestamp}-results.csv"
    else:
        csv_path = Path(csv_path)
        if not csv_path.is_absolute():
            csv_path = results_dir / csv_path

    rows = []
    for name, r in results.items():
        for metric_type in ("throughput", "latency"):
            m = r[metric_type]
            row = {
                "id": MODEL_IDS.get(name, "?"),
                "model": name,
                "type": r["type"],
                "metric": metric_type,
            }
            if m.get("status") and m["status"] != "OK":
                row["status"] = m["status"]
            else:
                row.update(
                    status="OK",
                    n=m["n"],
                    mdape=m["mdape"],
                    mape=m.get("mape"),
                    ms_per_100=r.get("ms_per_100"),
                    within_20=m["within_20"],
                )
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"  Results saved to {csv_path}")


def run_kavier_only(
    output_json: Optional[str] = None,
    max_gpus: Optional[int] = None,
    results_csv: Optional[Path] = None,
) -> dict:
    """Evaluate only the Kavier physics simulator (same test split as full suite)."""
    ml_data = _load_test_data("KAVIER ONLY (physics simulator)", max_gpus)
    results = {"Kavier": _eval_kavier_entry(ml_data)}
    return _finalize(results, results_csv, output_json)


def main():
    parser = argparse.ArgumentParser(
        description=("Unified benchmarking suite: throughput + latency for Kavier and/or ML predictors.")
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to save results as JSON.",
    )
    parser.add_argument(
        "--kavier-only",
        action="store_true",
        help="Run only the Kavier physics simulator (same test split as the full suite).",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Alias: use 'kavier' for Kavier-only run (same as --kavier-only).",
    )
    parser.add_argument(
        "--exclude-128gpu",
        action="store_true",
        help="Exclude 128-GPU configurations from evaluation (keeps ≤32 GPUs).",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default=None,
        help=(
            "CSV filename under benchmarks/results/ (default: timestamped). "
            "Use thesis-benchmark-results.csv for the canonical thesis table."
        ),
    )

    parser.add_argument("--one", default=None, help="Internal: evaluate a single model in isolation and print JSON.")
    parser.add_argument(
        "--max-gpus", type=int, default=None, help="Internal: max total GPUs filter passed to a --one worker."
    )

    args = parser.parse_args()
    if args.one:
        # Worker mode: library chatter -> stderr, only the marked JSON entry -> stdout.
        import logging

        logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
        entry = _evaluate_one_entry(args.one, args.max_gpus)
        print(_RESULT_MARKER + json.dumps(entry, default=_json_default))
        return

    results_csv = Path(args.results_csv) if args.results_csv else None
    max_gpus = 32 if args.exclude_128gpu else None
    kavier_only = args.kavier_only or (args.model is not None and args.model.strip().lower() == "kavier")
    runner = run_kavier_only if kavier_only else run_all
    runner(output_json=args.output, max_gpus=max_gpus, results_csv=results_csv)


if __name__ == "__main__":
    main()
