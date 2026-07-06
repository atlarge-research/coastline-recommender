#!/usr/bin/env python3
"""Bayesian Ridge predictor: polynomial features + per-prediction uncertainty. Target MdAPE 15-25%."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from .common import (
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


def train():
    """Train the Bayesian Ridge model."""
    print("=" * 70)
    print("BAYESIAN RIDGE PREDICTOR TRAINING")
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

    print("\n🔗 Combining features for preprocessing...")

    cat_indices = list(range(len(cat_features)))
    num_indices = list(range(len(cat_features), len(cat_features) + len(num_features)))

    X_train = pd.concat([X_cat_train.reset_index(drop=True), X_num_train.reset_index(drop=True)], axis=1)
    X_val = pd.concat([X_cat_val.reset_index(drop=True), X_num_val.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_cat_test.reset_index(drop=True), X_num_test.reset_index(drop=True)], axis=1)

    print(f"  ✓ Feature matrix shape: {X_train.shape}")
    print(f"  ✓ Categorical indices: {cat_indices}")
    print(f"  ✓ Numerical indices: {num_indices}")

    print("\n🧠 Building Bayesian Ridge pipeline...")
    print("  Preprocessing: OneHotEncoder (categorical) + StandardScaler (numerical)")
    print("  Feature expansion: PolynomialFeatures (degree 1-2)")
    print("  Model: BayesianRidge with automatic relevance determination")

    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_indices),
            ("num", StandardScaler(), num_indices),
        ],
        remainder="drop",
    )

    def build_model() -> Pipeline:
        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("poly", PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
                (
                    "regressor",
                    BayesianRidge(
                        max_iter=300,
                        alpha_init=1.0,
                        lambda_init=1.0,
                        compute_score=True,
                    ),
                ),
            ]
        )

    model_throughput = build_model()
    model_runtime = build_model()

    print("\n🔍 Hyperparameter tuning with GridSearchCV...")
    print("  This may take several minutes...")

    param_grid = {
        "poly__degree": [1, 2],  # Linear vs polynomial
        "regressor__alpha_init": [0.1, 1.0, 10.0],  # Precision of noise
        "regressor__lambda_init": [0.1, 1.0, 10.0],  # Precision of weights
    }

    grid_search_throughput = GridSearchCV(
        estimator=model_throughput,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
    )
    grid_search_runtime = GridSearchCV(
        estimator=model_runtime,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
    )

    grid_search_throughput.fit(X_train, y_log_train[TARGET_COLUMNS["throughput"]].to_numpy())
    grid_search_runtime.fit(X_train, y_log_train[TARGET_COLUMNS["runtime_seconds"]].to_numpy())

    print("\n✅ Hyperparameter tuning complete!")
    print("\n🏆 Best throughput parameters:")
    for param, value in grid_search_throughput.best_params_.items():
        print(f"  {param}: {value}")
    print("\n🏆 Best runtime parameters:")
    for param, value in grid_search_runtime.best_params_.items():
        print(f"  {param}: {value}")

    print(f"\n📊 Best throughput CV MAE (log space): {-grid_search_throughput.best_score_:.4f}")
    print(f"📊 Best runtime CV MAE (log space): {-grid_search_runtime.best_score_:.4f}")

    best_model_throughput = grid_search_throughput.best_estimator_
    best_model_runtime = grid_search_runtime.best_estimator_

    print("\n" + "=" * 70)
    print("VALIDATION SET EVALUATION")
    print("=" * 70)

    y_val_pred_log_throughput, y_val_std_log_throughput = best_model_throughput.predict(X_val, return_std=True)
    y_val_pred_log_runtime, y_val_std_log_runtime = best_model_runtime.predict(X_val, return_std=True)
    y_val_pred_log = np.column_stack([y_val_pred_log_throughput, y_val_pred_log_runtime])
    y_val_pred = inverse_transform_targets(y_val_pred_log)

    val_metrics_throughput = calculate_metrics(
        y_val[TARGET_COLUMNS["throughput"]].to_numpy(),
        y_val_pred[:, 0],
        y_log_val[TARGET_COLUMNS["throughput"]].to_numpy(),
        y_val_pred_log[:, 0],
    )
    val_metrics_runtime = calculate_metrics(
        y_val[TARGET_COLUMNS["runtime_seconds"]].to_numpy(),
        y_val_pred[:, 1],
        y_log_val[TARGET_COLUMNS["runtime_seconds"]].to_numpy(),
        y_val_pred_log[:, 1],
    )
    print_metrics(val_metrics_throughput, "Validation (Throughput)")
    print_metrics(val_metrics_runtime, "Validation (Runtime)")

    print("\n📊 Throughput uncertainty statistics (log space):")
    print(f"  Mean std: {y_val_std_log_throughput.mean():.4f}")
    print(f"  Median std: {np.median(y_val_std_log_throughput):.4f}")
    print(f"  Min std: {y_val_std_log_throughput.min():.4f}")
    print(f"  Max std: {y_val_std_log_throughput.max():.4f}")
    print("\n📊 Runtime uncertainty statistics (log space):")
    print(f"  Mean std: {y_val_std_log_runtime.mean():.4f}")
    print(f"  Median std: {np.median(y_val_std_log_runtime):.4f}")
    print(f"  Min std: {y_val_std_log_runtime.min():.4f}")
    print(f"  Max std: {y_val_std_log_runtime.max():.4f}")

    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)

    y_test_pred_log_throughput, y_test_std_log_throughput = best_model_throughput.predict(X_test, return_std=True)
    y_test_pred_log_runtime, y_test_std_log_runtime = best_model_runtime.predict(X_test, return_std=True)
    y_test_pred_log = np.column_stack([y_test_pred_log_throughput, y_test_pred_log_runtime])
    y_test_pred = inverse_transform_targets(y_test_pred_log)

    test_metrics_throughput = calculate_metrics(
        y_test[TARGET_COLUMNS["throughput"]].to_numpy(),
        y_test_pred[:, 0],
        y_log_test[TARGET_COLUMNS["throughput"]].to_numpy(),
        y_test_pred_log[:, 0],
    )
    test_metrics_runtime = calculate_metrics(
        y_test[TARGET_COLUMNS["runtime_seconds"]].to_numpy(),
        y_test_pred[:, 1],
        y_log_test[TARGET_COLUMNS["runtime_seconds"]].to_numpy(),
        y_test_pred_log[:, 1],
    )
    print_metrics(test_metrics_throughput, "Test (Throughput)")
    print_metrics(test_metrics_runtime, "Test (Runtime)")

    print("\n📊 Throughput uncertainty statistics (log space):")
    print(f"  Mean std: {y_test_std_log_throughput.mean():.4f}")
    print(f"  Median std: {np.median(y_test_std_log_throughput):.4f}")
    print(f"  Min std: {y_test_std_log_throughput.min():.4f}")
    print(f"  Max std: {y_test_std_log_throughput.max():.4f}")
    print("\n📊 Runtime uncertainty statistics (log space):")
    print(f"  Mean std: {y_test_std_log_runtime.mean():.4f}")
    print(f"  Median std: {np.median(y_test_std_log_runtime):.4f}")
    print(f"  Min std: {y_test_std_log_runtime.min():.4f}")
    print(f"  Max std: {y_test_std_log_runtime.max():.4f}")

    print("\n🔍 Sample throughput predictions with uncertainty (first 10):")
    print(f"{'True':>12} {'Predicted':>12} {'Std (log)':>12} {'Error %':>10}")
    print("-" * 48)
    y_test_throughput = y_test[TARGET_COLUMNS["throughput"]].to_numpy()
    for i in range(min(10, len(y_test_throughput))):
        error_pct = abs(y_test_throughput[i] - y_test_pred[i, 0]) / y_test_throughput[i] * 100
        print(
            f"{y_test_throughput[i]:>12,.1f} {y_test_pred[i, 0]:>12,.1f} {y_test_std_log_throughput[i]:>12.4f} {error_pct:>9.1f}%"
        )

    print("\n" + "=" * 70)
    print("UNCERTAINTY ANALYSIS")
    print("=" * 70)

    throughput_errors = np.abs(y_test[TARGET_COLUMNS["throughput"]].to_numpy() - y_test_pred[:, 0])
    runtime_errors = np.abs(y_test[TARGET_COLUMNS["runtime_seconds"]].to_numpy() - y_test_pred[:, 1])
    throughput_correlation = np.corrcoef(y_test_std_log_throughput, throughput_errors)[0, 1]
    runtime_correlation = np.corrcoef(y_test_std_log_runtime, runtime_errors)[0, 1]

    correlation = float(throughput_correlation)

    print(f"\n📈 Throughput uncertainty-error correlation: {throughput_correlation:.4f}")
    print(f"📈 Runtime uncertainty-error correlation: {runtime_correlation:.4f}")
    if throughput_correlation > 0.3:
        print("  ✓ Good: Higher uncertainty correlates with larger errors")
    elif correlation > 0.1:
        print("  ⚠️  Moderate: Some correlation between uncertainty and errors")
    else:
        print("  ⚠️  Weak: Uncertainty may not be well-calibrated")

    print("\n" + "=" * 70)
    print("MODEL ANALYSIS")
    print("=" * 70)

    regressor_throughput = best_model_throughput.named_steps["regressor"]
    regressor_runtime = best_model_runtime.named_steps["regressor"]
    print("\n📊 Bayesian Ridge Throughput Parameters:")
    print(f"  Alpha (noise precision): {regressor_throughput.alpha_:.4f}")
    print(f"  Lambda (weight precision): {regressor_throughput.lambda_:.4f}")
    print(f"  Number of iterations: {regressor_throughput.n_iter_}")
    print("\n📊 Bayesian Ridge Runtime Parameters:")
    print(f"  Alpha (noise precision): {regressor_runtime.alpha_:.4f}")
    print(f"  Lambda (weight precision): {regressor_runtime.lambda_:.4f}")
    print(f"  Number of iterations: {regressor_runtime.n_iter_}")

    poly_degree = best_model_throughput.named_steps["poly"].degree
    print(f"\n🔧 Polynomial degree: {poly_degree}")

    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)

    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)

    model_path = performance_trained_model_path("bayesian_ridge")
    artifacts = {
        "model": {
            "throughput": best_model_throughput,
            "runtime_seconds": best_model_runtime,
        },
        "cat_features": cat_features,
        "num_features": num_features,
        "cat_indices": cat_indices,
        "num_indices": num_indices,
        "best_params": {
            "throughput": grid_search_throughput.best_params_,
            "runtime_seconds": grid_search_runtime.best_params_,
        },
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
        "uncertainty_correlation": float(throughput_correlation),
        "uncertainty_correlation_by_target": {
            "throughput": float(throughput_correlation),
            "runtime_seconds": float(runtime_correlation),
        },
        "alpha": float(regressor_throughput.alpha_),
        "lambda": float(regressor_throughput.lambda_),
        "alpha_by_target": {
            "throughput": float(regressor_throughput.alpha_),
            "runtime_seconds": float(regressor_runtime.alpha_),
        },
        "lambda_by_target": {
            "throughput": float(regressor_throughput.lambda_),
            "runtime_seconds": float(regressor_runtime.lambda_),
        },
    }

    new_mdape = float(test_metrics_throughput["original_space"]["mdape"])
    _, save_msg = save_pickled_artifact_if_better(model_path, artifacts, new_mdape)
    print(f"💾 {save_msg}")
    print(f"  ✓ Test throughput MdAPE: {test_metrics_throughput['original_space']['mdape']:.2f}%")
    print(f"  ✓ Test runtime MdAPE: {test_metrics_runtime['original_space']['mdape']:.2f}%")
    print(f"  ✓ Test throughput R²: {test_metrics_throughput['original_space']['r2']:.4f}")
    print(f"  ✓ Test runtime R²: {test_metrics_runtime['original_space']['r2']:.4f}")
    print(f"  ✓ Throughput uncertainty-error correlation: {throughput_correlation:.4f}")
    print(f"  ✓ Runtime uncertainty-error correlation: {runtime_correlation:.4f}")
    print(f"  ✓ Polynomial degree: {poly_degree}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    mdape = test_metrics_throughput["original_space"]["mdape"]
    runtime_mdape = test_metrics_runtime["original_space"]["mdape"]
    target_range = "15-25%"

    if mdape <= 25:
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
    print(f"  Throughput Uncertainty Correlation: {throughput_correlation:.4f}")
    print(f"  Runtime Uncertainty Correlation: {runtime_correlation:.4f}")
    print("\n🚀 Bayesian Ridge predictor ready for inference with uncertainty!")


if __name__ == "__main__":
    train()
