#!/usr/bin/env python3
"""LightGBM predictor (model #7 of 10), tuned with GridSearchCV. Target MdAPE 8-14%."""

from typing import cast

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder

from .common import (
    PORTFOLIO_DIR,
    SEED,
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


def train():
    """Train the LightGBM model."""
    print("=" * 70)
    print("LIGHTGBM PREDICTOR TRAINING")
    print("=" * 70)

    print("\n📂 Loading data...")
    X_cat, X_num, y, cat_features, num_features = load_and_preprocess_data()
    y_log = transform_targets(y)

    print(f"  ✓ Loaded {len(y)} samples")
    print(
        f"  ✓ Throughput range: {y[TARGET_COLUMNS['throughput']].min():.0f} – {y[TARGET_COLUMNS['throughput']].max():.0f} tokens/sec"
    )
    print(
        f"  ✓ Runtime range: {y[TARGET_COLUMNS['runtime_seconds']].min():.0f} – {y[TARGET_COLUMNS['runtime_seconds']].max():.0f} sec"
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
    encoders = {}
    X_cat_train_enc = pd.DataFrame()
    X_cat_val_enc = pd.DataFrame()
    X_cat_test_enc = pd.DataFrame()

    for col in cat_features:
        encoder = LabelEncoder()

        # Add explicit 'unknown' class
        train_vals = list(X_cat_train[col].unique()) + ["unknown"]
        encoder.fit(train_vals)

        unknown_idx = encoder.transform(["unknown"])[0]

        def safe_transform(values):
            return [encoder.transform([v])[0] if v in encoder.classes_ else unknown_idx for v in values]

        X_cat_train_enc[col] = encoder.transform(X_cat_train[col])
        X_cat_val_enc[col] = safe_transform(X_cat_val[col])
        X_cat_test_enc[col] = safe_transform(X_cat_test[col])

        encoders[col] = encoder
        classes = cast(np.ndarray, encoder.classes_)
        print(f"  ✓ {col}: {len(classes)} classes")

    print("\n🔗 Combining features...")
    X_train = pd.concat([X_cat_train_enc.reset_index(drop=True), X_num_train.reset_index(drop=True)], axis=1)
    X_val = pd.concat([X_cat_val_enc.reset_index(drop=True), X_num_val.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_cat_test_enc.reset_index(drop=True), X_num_test.reset_index(drop=True)], axis=1)

    print(f"  ✓ Feature matrix shape: {X_train.shape}")

    print("\n🔍 Hyperparameter tuning with GridSearchCV...")
    print("  This may take several minutes...")

    # Right-sized grid (256 candidates). The MultiOutputRegressor wrapper blocks
    # per-output eval_set, so there's no early stopping here — cost is bounded by
    # capping n_estimators instead of the 5,832-candidate × 2,500-tree runaway.
    param_grid = {
        "estimator__n_estimators": [500, 1000],
        "estimator__max_depth": [9, 11],
        "estimator__learning_rate": [0.025, 0.05],
        "estimator__num_leaves": [63, 127],
        "estimator__subsample": [0.8, 1.0],
        "estimator__colsample_bytree": [0.8, 1.0],
        "estimator__reg_alpha": [0, 0.1],
        "estimator__reg_lambda": [0, 0.1],
    }

    inner = LGBMRegressor(
        min_child_samples=15,
        random_state=SEED,
        n_jobs=1,  # single-thread inner; GridSearchCV(n_jobs=-1) does the parallelism (avoid nested-thread deadlock)
        verbose=-1,
    )

    base_model = MultiOutputRegressor(inner)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
    )

    # LightGBM >=4.0 removed early_stopping_rounds/verbose from fit(); with the
    # MultiOutputRegressor wrapper an eval_set can't be forwarded per-output, so
    # we fit the full grid without early stopping (n_estimators is grid-bounded).
    grid_search.fit(
        X_train,
        y_log_train.to_numpy(),
    )

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
    print_metrics(val_metrics_throughput, "Validation (Throughput)")
    print_metrics(val_metrics_runtime, "Validation (Runtime)")

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
    print_metrics(test_metrics_throughput, "Test (Throughput)")
    print_metrics(test_metrics_runtime, "Test (Runtime)")

    print("\n🔍 Sample throughput predictions (first 10):")
    print(f"{'True':>12} {'Predicted':>12} {'Error %':>10}")
    print("-" * 36)
    y_test_throughput = y_test[TARGET_COLUMNS["throughput"]].to_numpy()
    for i in range(min(10, len(y_test_throughput))):
        error_pct = abs(y_test_throughput[i] - y_test_pred[i, 0]) / y_test_throughput[i] * 100
        print(f"{y_test_throughput[i]:>12,.1f} {y_test_pred[i, 0]:>12,.1f} {error_pct:>9.1f}%")

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    feature_importance = np.mean(
        [est.feature_importances_ for est in best_model.estimators_],
        axis=0,
    )
    feature_names = X_train.columns.tolist()

    importance_df = pd.DataFrame({"feature": feature_names, "importance": feature_importance}).sort_values(
        "importance", ascending=False
    )

    print("\n📊 Top 10 most important features:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:20s}: {row['importance']:>8.0f}")

    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)

    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)

    model_path = performance_trained_model_path("lightgbm")
    artifacts = {
        "model": best_model,
        "encoders": encoders,
        "cat_features": cat_features,
        "num_features": num_features,
        "best_params": stripped_best_params,
        "feature_importance": importance_df.to_dict("records"),
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
    runtime_mdape = test_metrics_runtime["original_space"]["mdape"]
    target_range = "8-14%"

    if mdape <= 14:
        status = "✅ SUCCESS"
        emoji = "🎉"
    else:
        status = "⚠️  NEEDS IMPROVEMENT"
        emoji = "🔧"

    print(f"\n{emoji} {status}")
    print(f"  Throughput target MdAPE: {target_range}")
    print(f"  Achieved throughput MdAPE: {mdape:.2f}%")
    print(f"  Achieved runtime MdAPE: {runtime_mdape:.2f}%")
    print(f"  Throughput Test R²: {test_metrics_throughput['original_space']['r2']:.4f}")
    print(f"  Runtime Test R²: {test_metrics_runtime['original_space']['r2']:.4f}")
    print(f"  Throughput Within 20%: {test_metrics_throughput['original_space']['within_20_pct']:.1f}%")
    print(f"  Runtime Within 20%: {test_metrics_runtime['original_space']['within_20_pct']:.1f}%")
    print("\n🚀 LightGBM predictor ready for inference!")


if __name__ == "__main__":
    train()
