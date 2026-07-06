#!/usr/bin/env python3
"""SVR predictor (model #2 of 10): RBF kernel + StandardScaler (scale-sensitive). Target MdAPE 10-18%."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .common import (
    PORTFOLIO_DIR,
    TARGET_COLUMNS,
    as_dataframes,
    calculate_metrics,
    encode_categorical_features,
    inverse_transform_targets,
    load_and_preprocess_data,
    performance_trained_model_path,
    print_metrics,
    save_pickled_artifact_if_better,
    split_data,
    transform_targets,
)


def train():
    """Train the SVR model."""
    print("=" * 70)
    print("SVR PREDICTOR TRAINING")
    print("=" * 70)

    print("\n📂 Loading data...")
    X_cat, X_num, y, cat_features, num_features = load_and_preprocess_data()
    y_log = transform_targets(y)

    print(f"  ✓ Loaded {len(y)} samples")
    print(
        f"  ✓ Throughput range: {y[TARGET_COLUMNS['throughput']].min():.0f} – {y[TARGET_COLUMNS['throughput']].max():.0f} tokens/sec"
    )
    print(
        f"  ✓ Runtime range: {y[TARGET_COLUMNS['runtime_seconds']].min():.2f} – {y[TARGET_COLUMNS['runtime_seconds']].max():.2f} sec"
    )
    print(f"  ✓ Categorical features: {cat_features}")
    print(f"  ✓ Numerical features: {num_features}")

    print("\n✂️  Splitting data...")
    (
        (X_cat_train, X_num_train, y_train, y_log_train),
        (X_cat_val, X_num_val, y_val, y_log_val),
        (X_cat_test, X_num_test, y_test, y_log_test),
    ) = split_data(X_cat, X_num, y, y_log)
    X_cat_train, X_num_train, y_train, y_log_train = as_dataframes(X_cat_train, X_num_train, y_train, y_log_train)
    X_cat_val, X_num_val, y_val, y_log_val = as_dataframes(X_cat_val, X_num_val, y_val, y_log_val)
    X_cat_test, X_num_test, y_test, y_log_test = as_dataframes(X_cat_test, X_num_test, y_test, y_log_test)

    print(f"  ✓ Train: {len(y_train)} samples")
    print(f"  ✓ Val:   {len(y_val)} samples")
    print(f"  ✓ Test:  {len(y_test)} samples")

    print("\n🔤 Encoding categorical features...")

    X_cat_train_enc, X_cat_val_enc, X_cat_test_enc, encoders, vocab_sizes = encode_categorical_features(
        X_cat_train, X_cat_val, X_cat_test
    )

    for col, size in vocab_sizes.items():
        print(f"  ✓ {col}: {size} classes")

    print("\n🔗 Combining features...")
    X_train = pd.concat([X_cat_train_enc.reset_index(drop=True), X_num_train.reset_index(drop=True)], axis=1)
    X_val = pd.concat([X_cat_val_enc.reset_index(drop=True), X_num_val.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_cat_test_enc.reset_index(drop=True), X_num_test.reset_index(drop=True)], axis=1)

    print(f"  ✓ Feature matrix shape: {X_train.shape}")

    print("\n🔍 Hyperparameter tuning with GridSearchCV...")
    print("  This may take several minutes...")

    param_grid = {
        "estimator__svr__C": [1.0, 10.0, 100.0],
        "estimator__svr__epsilon": [0.1, 0.25],
        "estimator__svr__gamma": ["scale", 0.01],
    }

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svr",
                SVR(
                    kernel="rbf",
                    cache_size=500,  # MB of cache for kernel computation
                    verbose=False,
                ),
            ),
        ]
    )

    base_model = MultiOutputRegressor(pipeline)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
    )

    grid_search.fit(X_train, y_log_train.to_numpy())

    print("\n✅ Hyperparameter tuning complete!")
    print("\n🏆 Best parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")

    print(f"\n📊 Best CV MAE (log space): {-grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    stripped_best_params = {k.replace("estimator__", "", 1): v for k, v in grid_search.best_params_.items()}

    print("\n" + "=" * 70)
    print("VALIDATION SET EVALUATION")
    print("=" * 70)

    y_val_pred_log = best_model.predict(X_val)
    y_val_pred = inverse_transform_targets(y_val_pred_log)

    val_metrics_throughput = calculate_metrics(
        y_val[TARGET_COLUMNS["throughput"]].to_numpy(),
        y_val_pred[:, 0],
        y_log_val[TARGET_COLUMNS["throughput"]].to_numpy(),
        np.asarray(y_val_pred_log)[:, 0],
    )
    val_metrics_runtime = calculate_metrics(
        y_val[TARGET_COLUMNS["runtime_seconds"]].to_numpy(),
        y_val_pred[:, 1],
        y_log_val[TARGET_COLUMNS["runtime_seconds"]].to_numpy(),
        np.asarray(y_val_pred_log)[:, 1],
    )
    print_metrics(val_metrics_throughput, "Validation Throughput")
    print_metrics(val_metrics_runtime, "Validation Runtime", unit="sec")

    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)

    y_test_pred_log = best_model.predict(X_test)
    y_test_pred = inverse_transform_targets(y_test_pred_log)

    test_metrics_throughput = calculate_metrics(
        y_test[TARGET_COLUMNS["throughput"]].to_numpy(),
        y_test_pred[:, 0],
        y_log_test[TARGET_COLUMNS["throughput"]].to_numpy(),
        np.asarray(y_test_pred_log)[:, 0],
    )
    test_metrics_runtime = calculate_metrics(
        y_test[TARGET_COLUMNS["runtime_seconds"]].to_numpy(),
        y_test_pred[:, 1],
        y_log_test[TARGET_COLUMNS["runtime_seconds"]].to_numpy(),
        np.asarray(y_test_pred_log)[:, 1],
    )
    print_metrics(test_metrics_throughput, "Test Throughput")
    print_metrics(test_metrics_runtime, "Test Runtime", unit="sec")

    print("\n🔍 Sample throughput predictions (first 10):")
    print(f"{'True':>12} {'Predicted':>12} {'Error %':>10}")
    print("-" * 36)
    y_test_throughput = y_test[TARGET_COLUMNS["throughput"]].to_numpy()
    for i in range(min(10, len(y_test_throughput))):
        error_pct = abs(y_test_throughput[i] - y_test_pred[i, 0]) / y_test_throughput[i] * 100
        print(f"{y_test_throughput[i]:>12,.1f} {y_test_pred[i, 0]:>12,.1f} {error_pct:>9.1f}%")

    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)

    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)

    model_path = performance_trained_model_path("svr")
    artifacts = {
        "model": best_model,  # Pipeline with scaler + SVR
        "encoders": encoders,
        "cat_features": cat_features,
        "num_features": num_features,
        "best_params": stripped_best_params,
        "test_metrics": test_metrics_throughput,
        "val_metrics": val_metrics_throughput,
        "test_metrics_by_target": {
            "throughput": test_metrics_throughput,
            "runtime_seconds": test_metrics_runtime,
        },
        "val_metrics_by_target": {
            "throughput": val_metrics_throughput,
            "runtime_seconds": val_metrics_runtime,
        },
    }

    new_mdape = float(test_metrics_throughput["original_space"]["mdape"])
    _, save_msg = save_pickled_artifact_if_better(model_path, artifacts, new_mdape)
    print(f"💾 {save_msg}")
    print(f"  ✓ Test throughput MdAPE: {test_metrics_throughput['original_space']['mdape']:.2f}%")
    print(f"  ✓ Test runtime MdAPE: {test_metrics_runtime['original_space']['mdape']:.2f}%")
    print(f"  ✓ Test throughput R²: {test_metrics_throughput['original_space']['r2']:.4f}")
    print(f"  ✓ Test runtime R²: {test_metrics_runtime['original_space']['r2']:.4f}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    mdape = test_metrics_throughput["original_space"]["mdape"]
    target_range = "10-18%"

    if mdape <= 18:
        status = "✅ SUCCESS"
        emoji = "🎉"
    else:
        status = "⚠️  NEEDS IMPROVEMENT"
        emoji = "🔧"

    print(f"\n{emoji} {status}")
    print(f"  Target MdAPE: {target_range}")
    print(f"  Achieved MdAPE: {mdape:.2f}%")
    print(f"  Test throughput R²: {test_metrics_throughput['original_space']['r2']:.4f}")
    print(f"  Test runtime R²: {test_metrics_runtime['original_space']['r2']:.4f}")
    print(f"  Throughput within 20%: {test_metrics_throughput['original_space']['within_20_pct']:.1f}%")
    print(f"  Runtime within 20%: {test_metrics_runtime['original_space']['within_20_pct']:.1f}%")
    print("\n🚀 SVR predictor ready for inference!")


if __name__ == "__main__":
    train()
