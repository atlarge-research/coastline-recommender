#!/usr/bin/env python3
"""Evaluate all 10 trained models via predict() in isolated subprocesses (prevents macOS OpenMP co-load crash)."""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.workload import WorkloadSpec

from .common import (
    DATA_PATH,
    TARGET_COLUMNS,
    calculate_metrics,
    engineer_features,
    get_primary_target_name,
    load_and_preprocess_data,
    split_data,
)

# Display name -> (module basename, predictor class); imported lazily per subprocess.
_MODELS: dict[str, tuple[str, str]] = {
    "RandomForest": ("random_forest_predictor", "RandomForestPredictor"),
    "SVR": ("svr_predictor", "SVRPredictor"),
    "KNN": ("knn_predictor", "KNNPredictor"),
    "CatBoost": ("catboost_predictor", "CatBoostPredictor"),
    "XGBoost": ("xgboost_predictor", "XGBoostPredictor"),
    "LightGBM": ("lightgbm_predictor", "LightGBMPredictor"),
    "GaussianProcess": ("gaussian_process_predictor", "GaussianProcessPredictor"),
    "BayesianRidge": ("bayesian_ridge_predictor", "BayesianRidgePredictor"),
    "TabPFN": ("tabpfn_predictor", "TabPFNPredictor"),
    "DeepLearning": ("deep_learning_predictor", "DeepLearningPredictor"),
}

_RESULT_MARKER = "__EVAL_RESULT__"


def _load_test_curated_rows() -> "pd.DataFrame":
    """Curated test split (same filters/split as the training pipeline)."""
    df = pd.read_csv(DATA_PATH)
    if "is_valid" in df.columns:
        df = df.loc[df["is_valid"] == 1.0].copy()
    throughput_col = TARGET_COLUMNS["throughput"]
    runtime_col = TARGET_COLUMNS["runtime_seconds"]
    valid_mask = df[throughput_col].notna() & (df[throughput_col] > 0) & df[runtime_col].notna() & (df[runtime_col] > 0)
    df = df.loc[valid_mask].copy()
    df = engineer_features(df)
    indices = np.arange(len(df))
    (_,), (_,), (test_idx,) = split_data(indices)
    return df.iloc[test_idx].reset_index(drop=True)


def _stored_test_metrics_for_model(display_name: str) -> dict | None:
    """Use metrics saved at train time when live inference fails (e.g. TabPFN absent)."""
    import pickle

    stem = {"TabPFN": "tabpfn"}.get(display_name)
    if stem is None:
        return None
    from .common import performance_trained_model_path

    artifact_path = performance_trained_model_path(stem)
    if not artifact_path.exists():
        return None
    with open(artifact_path, "rb") as f:
        artifact = pickle.load(f)
    by_target = artifact.get("test_metrics_by_target") or {}
    throughput = by_target.get("throughput") or artifact.get("test_metrics") or {}
    orig = throughput.get("original_space") if isinstance(throughput, dict) else None
    return orig if isinstance(orig, dict) else None


def _row_to_workload(row: pd.Series) -> WorkloadSpec:
    """Map a curated CSV row to WorkloadSpec for predictor APIs."""
    enable_roce = row.get("enable_roce")
    if pd.isna(enable_roce):
        roce_val = None
    elif isinstance(enable_roce, (bool, np.bool_)):
        roce_val = bool(enable_roce)
    else:
        s = str(enable_roce).strip().lower()
        roce_val = True if s in ("1", "true", "yes") else False if s in ("0", "false", "no") else None

    td = row.get("torch_dtype")
    torch_dtype = None if pd.isna(td) else str(td)

    return WorkloadSpec(
        llm_model=str(row.get("model_name", "unknown")),
        fine_tuning_method=str(row["method"]),
        gpu_model=str(row["gpu_model"]),
        batch_size=int(row["batch_size"]),
        tokens_per_sample=int(row["tokens_per_sample"]),
        gpus_per_node=int(row.get("number_gpus", 1) or 1),
        number_of_nodes=int(row.get("number_nodes", 1) or 1),
        torch_dtype=torch_dtype,
        enable_roce=roce_val,
    )


def _evaluate_one_model(name: str) -> dict:
    """Evaluate ONE model over the test split. Runs in its own process."""
    import importlib

    X_cat, X_num, y, _cat, _num = load_and_preprocess_data()
    y_log = np.log1p(y)
    (_, _, _, _), (_, _, _, _), (_, _, y_test, y_log_test) = split_data(X_cat, X_num, y, y_log)
    df_test = _load_test_curated_rows()
    throughput_col = get_primary_target_name()
    y_true = y_test[throughput_col].to_numpy(dtype=float)
    context = SystemContext.for_gpus(["NVIDIA-A100-SXM4-80GB"], max_gpus=128, max_nodes=16)

    module, cls = _MODELS[name]
    try:
        Predictor = getattr(importlib.import_module(f"coastline.sdk.predictors.performance.data_driven.{module}"), cls)
        predictor = Predictor()
        preds = []
        for _, row in df_test.iterrows():
            p = predictor.predict(_row_to_workload(row), context)
            preds.append(p.predicted_throughput if (p and p.predicted_throughput is not None) else 0.0)
        y_pred = np.array(preds)
        m = calculate_metrics(y_true, y_pred, y_log_test[throughput_col].to_numpy(dtype=float), np.log1p(y_pred))[
            "original_space"
        ]
        return {
            "Model": name,
            "ok": True,
            "mdape": m["mdape"],
            "r2": m["r2"],
            "mae": m["mae"],
            "within20": m["within_20_pct"],
        }
    except FileNotFoundError:
        return {"Model": name, "ok": False, "status": "not trained"}
    except Exception as e:
        stored = _stored_test_metrics_for_model(name)
        if stored and stored.get("mdape") is not None:
            return {
                "Model": name,
                "ok": True,
                "artifact": True,
                "mdape": stored["mdape"],
                "r2": stored.get("r2", 0),
                "mae": stored.get("mae", 0),
                "within20": stored.get("within_20_pct", 0),
            }
        return {"Model": name, "ok": False, "status": str(e)[:50]}


def evaluate_all():
    """Evaluate all models, each in an isolated subprocess (no native co-load)."""
    print("\n" + "=" * 100)
    print("EVALUATING ALL 10 MODELS USING PUBLIC API (one isolated subprocess per model)")
    print("=" * 100)

    results = []
    for name in _MODELS:
        print(f"\n📊 Evaluating {name}...")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "trainer.evaluate_all",
                "--one",
                name,
            ],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        r = next(
            (json.loads(ln[len(_RESULT_MARKER) :]) for ln in proc.stdout.splitlines() if ln.startswith(_RESULT_MARKER)),
            {"Model": name, "ok": False, "status": f"worker failed (rc={proc.returncode})"},
        )
        results.append(r)
        if r.get("ok"):
            print(f"  ✅ {name}: MdAPE = {r['mdape']:.2f}%" + (" (train artifact)" if r.get("artifact") else ""))
        else:
            print(f"  ❌ {name}: {r.get('status')}")

    rows = []
    for r in results:
        if r.get("ok"):
            rows.append(
                {
                    "Model": r["Model"],
                    "MdAPE": f"{r['mdape']:.2f}%",
                    "R²": f"{r['r2']:.4f}",
                    "MAE": f"{r['mae']:.2f}",
                    "Within 20%": f"{r['within20']:.1f}%",
                    "Status": "✅ (train artifact)" if r.get("artifact") else "✅",
                }
            )
        else:
            rows.append(
                {
                    "Model": r["Model"],
                    "MdAPE": "N/A",
                    "R²": "N/A",
                    "MAE": "N/A",
                    "Within 20%": "N/A",
                    "Status": f"❌ {r.get('status')}",
                }
            )
    df_results = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("FINAL MODEL EVALUATION RESULTS")
    print("=" * 100)
    print(df_results.to_string(index=False))
    print("=" * 100)

    valid = [r for r in results if r.get("ok")]
    if valid:
        best = min(valid, key=lambda x: x["mdape"])
        print(f"\n🏆 Best Model: {best['Model']} (MdAPE: {best['mdape']:.2f}%)")
    print(f"\n✅ Working Models: {len(valid)}/{len(_MODELS)}")
    return df_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models (isolated per process).")
    parser.add_argument("--one", help="Evaluate a single model by name; print its result as JSON.")
    args = parser.parse_args()
    if args.one:
        # Worker mode: library chatter -> stderr, only the marked JSON result -> stdout.
        import logging

        logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
        print(_RESULT_MARKER + json.dumps(_evaluate_one_model(args.one)))
    else:
        evaluate_all()


if __name__ == "__main__":
    main()
