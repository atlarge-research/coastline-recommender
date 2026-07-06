#!/usr/bin/env python3
"""TabPFN foundation-model predictor (tabular), log1p targets, featv3 schema."""

import logging
import time
import warnings
from typing import Any, cast

import numpy as np
import pandas as pd

# Suppress warnings and set logging to ERROR only.
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
logging.getLogger("sklearn").setLevel(logging.ERROR)
logging.getLogger("tabpfn").setLevel(logging.ERROR)

TabPFNRegressor = None

try:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("ERROR: tabpfn not installed. Install with: pip install tabpfn")

from .common import (  # noqa: E402
    PORTFOLIO_DIR,
    TARGET_COLUMNS,
    as_dataframes,
    calculate_metrics,
    inverse_transform_targets,
    load_and_preprocess_data,
    performance_trained_model_path,
    print_metrics,
    save_pickled_artifact_if_better,
    split_data,
    transform_targets,
)


def train_tabpfn():
    if not TABPFN_AVAILABLE:
        print("\n❌ TabPFN not available. Install with: pip install tabpfn")
        return

    print("\n🚀 Training TabPFN Model...")

    # TabPFN handles mixed types natively, so feed concatenated raw features.
    X_cat, X_num, y, cat_features, num_features = load_and_preprocess_data()
    X_cat_df = pd.DataFrame(X_cat).astype(str)
    X_num_df = pd.DataFrame(X_num)
    X = pd.concat([X_cat_df, X_num_df], axis=1)

    y_log = transform_targets(y)

    (X_train, y_train, y_log_train), (X_val, y_val, y_log_val), (X_test, y_test, y_log_test) = split_data(X, y, y_log)
    X_train, y_train, y_log_train = as_dataframes(X_train, y_train, y_log_train)
    X_val, y_val, y_log_val = as_dataframes(X_val, y_val, y_log_val)
    X_test, y_test, y_log_test = as_dataframes(X_test, y_test, y_log_test)

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    # One model per target — TabPFN does not support multi-output.
    # Pinned to TabPFN v2 weights (Prior Labs License = Apache 2.0 + attribution):
    # v2.5/v2.6/v3 weights are research-only and cannot be redistributed, so the
    # saved pickle (which embeds the weights) must be built on v2.
    model_cls = cast(Any, TabPFNRegressor)

    def make_regressor():
        return model_cls.create_default_for_version(ModelVersion.V2, device=device, ignore_pretraining_limits=True)

    print(f"Training TabPFN (v2 weights) on {len(X_train)} samples...")
    start_time = time.time()

    model_throughput = make_regressor()
    y_throughput = y_log_train[TARGET_COLUMNS["throughput"]].to_numpy()
    model_throughput.fit(X_train.values, y_throughput)
    print("  ✓ Throughput model trained")

    model_runtime = make_regressor()
    y_runtime = y_log_train[TARGET_COLUMNS["runtime_seconds"]].to_numpy()
    model_runtime.fit(X_train.values, y_runtime)
    print("  ✓ Runtime model trained")

    model = {"throughput": model_throughput, "runtime": model_runtime}

    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.2f} seconds")

    y_log_val_pred_throughput = model["throughput"].predict(X_val.values)
    y_log_val_pred_runtime = model["runtime"].predict(X_val.values)
    y_log_val_pred = np.column_stack([y_log_val_pred_throughput, y_log_val_pred_runtime])
    y_val_pred = inverse_transform_targets(y_log_val_pred)
    val_metrics_throughput = calculate_metrics(
        y_val[TARGET_COLUMNS["throughput"]].to_numpy(),
        y_val_pred[:, 0],
        y_log_val[TARGET_COLUMNS["throughput"]].to_numpy(),
        y_log_val_pred_throughput,
    )
    val_metrics_runtime = calculate_metrics(
        y_val[TARGET_COLUMNS["runtime_seconds"]].to_numpy(),
        y_val_pred[:, 1],
        y_log_val[TARGET_COLUMNS["runtime_seconds"]].to_numpy(),
        y_log_val_pred_runtime,
    )

    y_log_test_pred_throughput = model["throughput"].predict(X_test.values)
    y_log_test_pred_runtime = model["runtime"].predict(X_test.values)
    y_log_test_pred = np.column_stack([y_log_test_pred_throughput, y_log_test_pred_runtime])
    y_test_pred = inverse_transform_targets(y_log_test_pred)
    test_metrics_throughput = calculate_metrics(
        y_test[TARGET_COLUMNS["throughput"]].to_numpy(),
        y_test_pred[:, 0],
        y_log_test[TARGET_COLUMNS["throughput"]].to_numpy(),
        y_log_test_pred_throughput,
    )
    test_metrics_runtime = calculate_metrics(
        y_test[TARGET_COLUMNS["runtime_seconds"]].to_numpy(),
        y_test_pred[:, 1],
        y_log_test[TARGET_COLUMNS["runtime_seconds"]].to_numpy(),
        y_log_test_pred_runtime,
    )

    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    model_path = performance_trained_model_path("tabpfn")
    artifacts = {
        "model": model,
        "cat_features": cat_features,
        "num_features": num_features,
        "val_metrics": val_metrics_throughput,
        "test_metrics": test_metrics_throughput,
        "val_metrics_by_target": {
            "throughput": val_metrics_throughput,
            "runtime_seconds": val_metrics_runtime,
        },
        "test_metrics_by_target": {
            "throughput": test_metrics_throughput,
            "runtime_seconds": test_metrics_runtime,
        },
    }
    new_mdape = float(test_metrics_throughput["original_space"]["mdape"])
    _, save_msg = save_pickled_artifact_if_better(model_path, artifacts, new_mdape)
    print(save_msg)

    print_metrics(test_metrics_throughput, "TabPFN Test (Throughput)")
    print_metrics(test_metrics_runtime, "TabPFN Test (Runtime)")

    return model, None


if __name__ == "__main__":
    train_tabpfn()
