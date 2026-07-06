#!/usr/bin/env python3
"""CatBoost throughput/runtime predictor with native categorical support. Target MdAPE 6-12%."""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

# The picklable wrapper is the shipped package's — one definition, shared with inference.
from coastline.sdk.predictors.performance.data_driven._catboost_model import _DualOutputCatBoost

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


def _tune_catboost_head(
    target_idx,
    target_name,
    param_combinations,
    cat_feature_indices,
    X_train,
    X_val,
    y_log_train,
    y_log_val,
):
    """Grid-search a single-target CatBoost head; select by validation MAE (log space)."""
    print(f"\n🔍 Tuning CatBoost head for {target_name} (target column {target_idx})...")
    best_score = float("inf")
    best_params = None
    best_model = None

    y_tr = y_log_train.iloc[:, target_idx].to_numpy()
    y_va = y_log_val.iloc[:, target_idx].to_numpy()

    for i, params in enumerate(param_combinations, 1):
        model = CatBoostRegressor(
            **params,
            cat_features=cat_feature_indices,
            random_seed=SEED,
            early_stopping_rounds=500,
            eval_metric="MAE",
            task_type="CPU",
            verbose=False,
            allow_writing_files=False,  # don't write catboost_info/ into read-only workdir
        )
        model.fit(X_train, y_tr, eval_set=(X_val, y_va), verbose=False)
        val_pred = model.predict(X_val)
        val_mae = float(np.mean(np.abs(val_pred - y_va)))
        print(f"  [{target_name}] combination {i}/{len(param_combinations)} {params} -> val MAE {val_mae:.4f}")
        if val_mae < best_score:
            best_score = val_mae
            best_params = params
            best_model = model
            print(f"    ✓ New best {target_name} model!")

    print(
        f"  🏆 Best {target_name} params: {best_params} (val MAE {best_score:.4f}, "
        f"best iteration {best_model.get_best_iteration()})"
    )
    return best_model, best_params, best_score


def train():
    """Train the CatBoost model."""
    print("=" * 70)
    print("CATBOOST PREDICTOR TRAINING")
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

    print("\n🔗 Combining features...")
    X_train = pd.concat([X_cat_train.reset_index(drop=True), X_num_train.reset_index(drop=True)], axis=1)
    X_val = pd.concat([X_cat_val.reset_index(drop=True), X_num_val.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_cat_test.reset_index(drop=True), X_num_test.reset_index(drop=True)], axis=1)

    print(f"  ✓ Feature matrix shape: {X_train.shape}")

    cat_feature_indices = [i for i, col in enumerate(X_train.columns) if col in cat_features]
    print(f"  ✓ Categorical feature indices: {cat_feature_indices}")

    print("\n🔍 Hyperparameter tuning with manual grid search...")
    print("  This may take several minutes...")

    param_combinations = [
        {"iterations": 1000, "depth": 6, "learning_rate": 0.01, "l2_leaf_reg": 1},
        {"iterations": 1000, "depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 3},
        {"iterations": 1500, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 3},
        {"iterations": 1500, "depth": 8, "learning_rate": 0.01, "l2_leaf_reg": 5},
        {"iterations": 1500, "depth": 10, "learning_rate": 0.03, "l2_leaf_reg": 3},
        {"iterations": 2000, "depth": 8, "learning_rate": 0.01, "l2_leaf_reg": 3},
        {"iterations": 2000, "depth": 10, "learning_rate": 0.03, "l2_leaf_reg": 5},
    ]

    # Tune ONE CatBoost head per target. CatBoost is single-output, so the prior
    # implementation only fit the throughput column and filled runtime with zeros
    # (expm1(0)=0) -> a degenerate runtime head. We now fit a dedicated runtime
    # head and expose both via a multi-output wrapper, matching every other model.
    throughput_idx = list(y_log_train.columns).index(TARGET_COLUMNS["throughput"])
    runtime_idx = list(y_log_train.columns).index(TARGET_COLUMNS["runtime_seconds"])

    throughput_model, best_params, best_score = _tune_catboost_head(
        throughput_idx,
        "throughput",
        param_combinations,
        cat_feature_indices,
        X_train,
        X_val,
        y_log_train,
        y_log_val,
    )
    runtime_model, runtime_best_params, runtime_best_score = _tune_catboost_head(
        runtime_idx,
        "runtime",
        param_combinations,
        cat_feature_indices,
        X_train,
        X_val,
        y_log_train,
        y_log_val,
    )

    best_model = _DualOutputCatBoost(throughput_model, runtime_model)

    print("\n✅ Hyperparameter tuning complete!")
    print("\n🏆 Best throughput parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print("\n🏆 Best runtime parameters:")
    for param, value in runtime_best_params.items():
        print(f"  {param}: {value}")

    print(f"\n📊 Best Validation MAE (log space): throughput {best_score:.4f}, runtime {runtime_best_score:.4f}")
    print(
        f"  Best iterations: throughput {throughput_model.get_best_iteration()}, "
        f"runtime {runtime_model.get_best_iteration()}"
    )

    print("\n" + "=" * 70)
    print("VALIDATION SET EVALUATION")
    print("=" * 70)

    y_val_pred_log = best_model.predict(X_val)  # 2D: [throughput, runtime] in log space
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

    y_test_pred_log = best_model.predict(X_test)  # 2D: [throughput, runtime] in log space
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

    feature_importance = best_model.get_feature_importance()
    feature_names = X_train.columns.tolist()

    importance_df = pd.DataFrame({"feature": feature_names, "importance": feature_importance}).sort_values(
        "importance", ascending=False
    )

    print("\n📊 Top 10 most important features:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:20s}: {row['importance']:>8.2f}")

    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)

    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)

    model_path = performance_trained_model_path("catboost")
    artifacts = {
        "model": best_model,
        "cat_features": cat_features,
        "num_features": num_features,
        "cat_feature_indices": cat_feature_indices,
        "best_params": best_params,
        "best_params_runtime": runtime_best_params,
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
    print(f"  ✓ Iterations: {best_model.get_best_iteration()}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    mdape = test_metrics_throughput["original_space"]["mdape"]
    runtime_mdape = test_metrics_runtime["original_space"]["mdape"]
    target_range = "6-12%"

    if mdape <= 12:
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
    print("\n🚀 CatBoost predictor ready for inference!")
    print("\n💡 CatBoost is expected to be the best single model in the portfolio.")


if __name__ == "__main__":
    train()
