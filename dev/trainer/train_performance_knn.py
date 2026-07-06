#!/usr/bin/env python3
"""KNN predictor: QuantileTransformer + Minkowski distance. Target MdAPE <20%."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

from .common import (
    PORTFOLIO_DIR,
    SEED,
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
    """Train the KNN model."""
    print("=" * 70)
    print("KNN PREDICTOR TRAINING")
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
    print("  QuantileTransformer + Minkowski distances; median loss in log space")
    print("  (aligns better with MdAPE than mean MAE).")
    print("  This may take a few minutes...")

    # Minkowski only: p=1 → Manhattan, p=2 → Euclidean (avoids invalid metric/p combos).
    param_grid = {
        "knn__n_neighbors": [4, 6, 8, 10, 12, 16, 20, 28, 36],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["minkowski"],
        "knn__p": [1, 2],
    }

    nq = min(512, max(10, len(X_train) - 1))
    pipeline = Pipeline(
        [
            (
                "scaler",
                QuantileTransformer(
                    output_distribution="normal",
                    random_state=SEED,
                    n_quantiles=nq,
                ),
            ),
            ("knn", KNeighborsRegressor(n_jobs=-1)),
        ]
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="neg_median_absolute_error",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    grid_search.fit(X_train, y_log_train.to_numpy())

    # KNN is instance-based: more reference points → better neighbors. Refit the best
    # pipeline on train ∪ val (test stays untouched for final metrics only).
    print("\n📌 Refitting best model on train + validation (standard for k-NN)...")
    best_model = clone(grid_search.best_estimator_)
    X_trainval = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    y_trainval = pd.concat([y_log_train, y_log_val], axis=0, ignore_index=True)
    best_model.fit(X_trainval, y_trainval.to_numpy())

    print("\n✅ Enhanced hyperparameter tuning complete!")
    print("\n🏆 Best parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")

    print(f"\n📊 Best CV median absolute error (log space): {-grid_search.best_score_:.4f}")

    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values("rank_test_score")
    print("\n📊 Top 5 configurations:")
    for idx, row in results_df.head(5).iterrows():
        print(f"  Rank {int(row['rank_test_score'])}: medAE={-row['mean_test_score']:.4f}, params={row['params']}")

    print("\n" + "=" * 70)
    print("VALIDATION SET EVALUATION")
    print("=" * 70)

    # Validation metrics: use estimator fit on TRAIN only (GridSearchCV best_estimator_).
    # ``best_model`` is refit on train∪val and must not be used for val evaluation.
    est_train_only = grid_search.best_estimator_
    y_val_pred_log = est_train_only.predict(X_val)
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

    model_path = performance_trained_model_path("knn")
    artifacts = {
        "model": best_model,  # Pipeline with scaler + KNN
        "encoders": encoders,
        "cat_features": cat_features,
        "num_features": num_features,
        "best_params": grid_search.best_params_,
        "refit_on_trainval": True,
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
    print("  ✓ Using QuantileTransformer (critical for distance-based KNN)")
    print(f"  ✓ Test throughput MdAPE: {test_metrics_throughput['original_space']['mdape']:.2f}%")
    print(f"  ✓ Test runtime MdAPE: {test_metrics_runtime['original_space']['mdape']:.2f}%")
    print(f"  ✓ Test throughput R²: {test_metrics_throughput['original_space']['r2']:.4f}")
    print(f"  ✓ Test runtime R²: {test_metrics_runtime['original_space']['r2']:.4f}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    mdape = test_metrics_throughput["original_space"]["mdape"]
    target_threshold = 20.0

    if mdape < target_threshold:
        status = "✅ SUCCESS"
        emoji = "🎉"
    else:
        status = "⚠️  NEEDS IMPROVEMENT"
        emoji = "🔧"

    print(f"\n{emoji} {status}")
    print(f"  Target MdAPE: <{target_threshold}%")
    print(f"  Achieved MdAPE: {mdape:.2f}%")
    print(f"  Test throughput R²: {test_metrics_throughput['original_space']['r2']:.4f}")
    print(f"  Test runtime R²: {test_metrics_runtime['original_space']['r2']:.4f}")
    print(f"  Throughput within 20%: {test_metrics_throughput['original_space']['within_20_pct']:.1f}%")
    print(f"  Runtime within 20%: {test_metrics_runtime['original_space']['within_20_pct']:.1f}%")
    print("\n🚀 KNN predictor ready for inference!")


if __name__ == "__main__":
    train()
