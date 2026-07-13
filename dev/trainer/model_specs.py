"""Per-model configuration table for the generic sklearn-family trainer.

Each entry is a :class:`ModelSpec`: metadata (title, artifact stem, target band) plus
one ``fit`` function holding the model's genuinely-unique part — its estimator,
hyperparameter grid, and the artifact keys it contributes. Everything shared
(load, split, encode, score, save) lives in ``generic_trainer``.

Heavy backends (xgboost / lightgbm / catboost) are imported inside their ``fit`` so
importing this table never co-loads native ML runtimes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import SEED
from .generic_trainer import (
    RUNTIME,
    THROUGHPUT,
    Encoding,
    FinalizeCtx,
    Fitted,
    ModelSpec,
    TrainData,
    importance_records,
    report_grid_search,
)

# Artifact key sets shared by several models (the four metric keys every model stores).
_METRIC_KEYS = frozenset(
    {"test_metrics", "val_metrics", "test_metrics_by_target", "val_metrics_by_target"}
)
_BASE_LABEL = _METRIC_KEYS | {"model", "encoders", "cat_features", "num_features"}
_BASE_RAW = _METRIC_KEYS | {"model", "cat_features", "num_features"}


def _grid_search(estimator, param_grid, **kwargs):
    from sklearn.model_selection import GridSearchCV

    defaults = dict(cv=3, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=2, return_train_score=True)
    defaults.update(kwargs)
    return GridSearchCV(estimator=estimator, param_grid=param_grid, **defaults)


def _uncertainty_finalize(ctx: FinalizeCtx) -> dict:
    """GP / BayesianRidge: correlate log-space std with original-space |error| on test."""
    yt = ctx.y_test[THROUGHPUT].to_numpy()
    yr = ctx.y_test[RUNTIME].to_numpy()
    t_corr = float(np.corrcoef(ctx.test_std[:, 0], np.abs(yt - ctx.test_pred[:, 0]))[0, 1])
    r_corr = float(np.corrcoef(ctx.test_std[:, 1], np.abs(yr - ctx.test_pred[:, 1]))[0, 1])
    print(f"\n📈 Throughput uncertainty-error correlation: {t_corr:.4f}")
    print(f"📈 Runtime uncertainty-error correlation: {r_corr:.4f}")
    return {
        "uncertainty_correlation": t_corr,
        "uncertainty_correlation_by_target": {"throughput": t_corr, "runtime_seconds": r_corr},
    }


# --------------------------------------------------------------------------- #
# fit functions — one per model, each near-verbatim from its old train script
# --------------------------------------------------------------------------- #


def _fit_xgboost(d: TrainData) -> Fitted:
    from xgboost import XGBRegressor

    print("\n🔍 Hyperparameter tuning with GridSearchCV...")
    param_grid = {
        "n_estimators": [1000, 1500, 2000, 2500],
        "max_depth": [8, 10, 12],
        "learning_rate": [0.01, 0.025, 0.05],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2],
        "min_child_weight": [1, 3, 5],
    }
    base = XGBRegressor(
        reg_alpha=0.1, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=0, early_stopping_rounds=500
    )
    grid = _grid_search(base, param_grid)
    grid.fit(d.X_train, d.y_log_train.to_numpy(), eval_set=[(d.X_val, d.y_log_val.to_numpy())], verbose=False)
    report_grid_search(grid)

    best = grid.best_estimator_
    fi = importance_records(best.feature_importances_, d.X_train.columns)
    return Fitted(
        model=best,
        predict=lambda X: (best.predict(X), None),
        metadata={"best_params": grid.best_params_, "feature_importance": fi},
    )


def _fit_lightgbm(d: TrainData) -> Fitted:
    from lightgbm import LGBMRegressor

    print("\n🔍 Hyperparameter tuning with GridSearchCV...")
    # MultiOutputRegressor blocks per-output eval_set, so bound cost by capping
    # n_estimators rather than early stopping.
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
    inner = LGBMRegressor(min_child_samples=15, random_state=SEED, n_jobs=1, verbose=-1)
    grid = _grid_search(MultiOutputRegressor(inner), param_grid)
    grid.fit(d.X_train, d.y_log_train.to_numpy())
    report_grid_search(grid)

    best = grid.best_estimator_
    stripped = {k.replace("estimator__", "", 1): v for k, v in grid.best_params_.items()}
    fi_values = np.mean([est.feature_importances_ for est in best.estimators_], axis=0)
    fi = importance_records(fi_values, d.X_train.columns)
    return Fitted(
        model=best,
        predict=lambda X: (best.predict(X), None),
        metadata={"best_params": stripped, "feature_importance": fi},
    )


def _fit_random_forest(d: TrainData) -> Fitted:
    from sklearn.ensemble import RandomForestRegressor

    print("\n🔍 Hyperparameter tuning with GridSearchCV...")
    param_grid = {
        "n_estimators": [1000, 1200, 1500],
        "max_depth": [20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5],
    }
    base = RandomForestRegressor(bootstrap=True, oob_score=True, random_state=SEED, n_jobs=-1, verbose=0)
    grid = _grid_search(base, param_grid)
    grid.fit(d.X_train, d.y_log_train.to_numpy())
    report_grid_search(grid)

    best = grid.best_estimator_
    return Fitted(
        model=best,
        predict=lambda X: (best.predict(X), None),
        metadata={"best_params": grid.best_params_, "oob_score": best.oob_score_},
    )


def _fit_svr(d: TrainData) -> Fitted:
    from sklearn.svm import SVR

    print("\n🔍 Hyperparameter tuning with GridSearchCV...")
    param_grid = {
        "estimator__svr__C": [1.0, 10.0, 100.0],
        "estimator__svr__epsilon": [0.1, 0.25],
        "estimator__svr__gamma": ["scale", 0.01],
    }
    pipeline = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf", cache_size=500, verbose=False))])
    grid = _grid_search(MultiOutputRegressor(pipeline), param_grid)
    grid.fit(d.X_train, d.y_log_train.to_numpy())
    report_grid_search(grid)

    best = grid.best_estimator_
    stripped = {k.replace("estimator__", "", 1): v for k, v in grid.best_params_.items()}
    return Fitted(model=best, predict=lambda X: (best.predict(X), None), metadata={"best_params": stripped})


def _fit_knn(d: TrainData) -> Fitted:
    from sklearn.base import clone
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import QuantileTransformer

    print("\n🔍 Hyperparameter tuning with GridSearchCV...")
    # Minkowski only: p=1 → Manhattan, p=2 → Euclidean.
    param_grid = {
        "knn__n_neighbors": [4, 6, 8, 10, 12, 16, 20, 28, 36],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["minkowski"],
        "knn__p": [1, 2],
    }
    nq = min(512, max(10, len(d.X_train) - 1))
    pipeline = Pipeline(
        [
            ("scaler", QuantileTransformer(output_distribution="normal", random_state=SEED, n_quantiles=nq)),
            ("knn", KNeighborsRegressor(n_jobs=-1)),
        ]
    )
    grid = _grid_search(pipeline, param_grid, scoring="neg_median_absolute_error", verbose=1)
    grid.fit(d.X_train, d.y_log_train.to_numpy())

    # k-NN is instance-based: refit the best pipeline on train ∪ val (test stays held out).
    best = clone(grid.best_estimator_)
    X_trainval = pd.concat([d.X_train, d.X_val], axis=0, ignore_index=True)
    y_trainval = pd.concat([d.y_log_train, d.y_log_val], axis=0, ignore_index=True)
    best.fit(X_trainval, y_trainval.to_numpy())
    report_grid_search(grid)

    train_only = grid.best_estimator_  # val is scored on the train-only estimator
    return Fitted(
        model=best,
        predict=lambda X: (best.predict(X), None),
        predict_val=lambda X: (train_only.predict(X), None),
        metadata={"best_params": grid.best_params_, "refit_on_trainval": True},
    )


def _tune_catboost_head(target_idx, target_name, param_combinations, cat_feature_indices, X_train, X_val, yl_train, yl_val):
    """Grid-search a single-target CatBoost head; select by validation MAE (log space)."""
    from catboost import CatBoostRegressor

    print(f"\n🔍 Tuning CatBoost head for {target_name} (target column {target_idx})...")
    best_score = float("inf")
    best_params = None
    best_model = None
    y_tr = yl_train.iloc[:, target_idx].to_numpy()
    y_va = yl_val.iloc[:, target_idx].to_numpy()

    for i, params in enumerate(param_combinations, 1):
        model = CatBoostRegressor(
            **params,
            cat_features=cat_feature_indices,
            random_seed=SEED,
            early_stopping_rounds=500,
            eval_metric="MAE",
            task_type="CPU",
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(X_train, y_tr, eval_set=(X_val, y_va), verbose=False)
        val_mae = float(np.mean(np.abs(model.predict(X_val) - y_va)))
        print(f"  [{target_name}] combination {i}/{len(param_combinations)} {params} -> val MAE {val_mae:.4f}")
        if val_mae < best_score:
            best_score, best_params, best_model = val_mae, params, model
            print(f"    ✓ New best {target_name} model!")

    print(f"  🏆 Best {target_name} params: {best_params} (val MAE {best_score:.4f})")
    return best_model, best_params


def _fit_catboost(d: TrainData) -> Fitted:
    # The picklable wrapper is the shipped package's — one definition, shared with inference.
    from coastline.sdk.predictors.performance.data_driven._catboost_model import _DualOutputCatBoost

    cat_feature_indices = [i for i, col in enumerate(d.X_train.columns) if col in d.cat_features]
    print(f"  ✓ Categorical feature indices: {cat_feature_indices}")

    print("\n🔍 Hyperparameter tuning with manual grid search...")
    param_combinations = [
        {"iterations": 1000, "depth": 6, "learning_rate": 0.01, "l2_leaf_reg": 1},
        {"iterations": 1000, "depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 3},
        {"iterations": 1500, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 3},
        {"iterations": 1500, "depth": 8, "learning_rate": 0.01, "l2_leaf_reg": 5},
        {"iterations": 1500, "depth": 10, "learning_rate": 0.03, "l2_leaf_reg": 3},
        {"iterations": 2000, "depth": 8, "learning_rate": 0.01, "l2_leaf_reg": 3},
        {"iterations": 2000, "depth": 10, "learning_rate": 0.03, "l2_leaf_reg": 5},
    ]
    throughput_idx = list(d.y_log_train.columns).index(THROUGHPUT)
    runtime_idx = list(d.y_log_train.columns).index(RUNTIME)

    args = (param_combinations, cat_feature_indices, d.X_train, d.X_val, d.y_log_train, d.y_log_val)
    t_model, best_params = _tune_catboost_head(throughput_idx, "throughput", *args)
    r_model, r_best_params = _tune_catboost_head(runtime_idx, "runtime", *args)
    best = _DualOutputCatBoost(t_model, r_model)

    fi = importance_records(best.get_feature_importance(), d.X_train.columns)
    return Fitted(
        model=best,
        predict=lambda X: (best.predict(X), None),
        metadata={
            "cat_feature_indices": cat_feature_indices,
            "best_params": best_params,
            "best_params_runtime": r_best_params,
            "feature_importance": fi,
        },
    )


def _fit_gaussian_process(d: TrainData) -> Fitted:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    print("\n🧠 Building Gaussian Process model (kernel: ConstantKernel * RBF + WhiteKernel)...")
    kernel = ConstantKernel(1.0, constant_value_bounds=(0.1, 10.0)) * RBF(
        length_scale=1.0, length_scale_bounds=(0.1, 10.0)
    ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))

    def build_model() -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "gp",
                    GaussianProcessRegressor(
                        kernel=kernel, n_restarts_optimizer=5, normalize_y=True, alpha=1e-6, random_state=SEED
                    ),
                ),
            ]
        )

    model_t, model_r = build_model(), build_model()
    print("🚀 Training Gaussian Process (scales O(n³))...")
    model_t.fit(d.X_train, d.y_log_train[THROUGHPUT].to_numpy())
    model_r.fit(d.X_train, d.y_log_train[RUNTIME].to_numpy())

    gp_t, scaler_t = model_t.named_steps["gp"], model_t.named_steps["scaler"]
    gp_r, scaler_r = model_r.named_steps["gp"], model_r.named_steps["scaler"]
    print(f"🔧 Optimized throughput kernel: {gp_t.kernel_}")
    print(f"🔧 Optimized runtime kernel:    {gp_r.kernel_}")

    def predict(X):
        t_pred, t_std = gp_t.predict(scaler_t.transform(X), return_std=True)
        r_pred, r_std = gp_r.predict(scaler_r.transform(X), return_std=True)
        return np.column_stack([t_pred, r_pred]), np.column_stack([t_std, r_std])

    return Fitted(
        model={"throughput": model_t, "runtime_seconds": model_r},
        predict=predict,
        metadata={"kernel": {"throughput": str(gp_t.kernel_), "runtime_seconds": str(gp_r.kernel_)}},
        finalize=_uncertainty_finalize,
    )


def _fit_bayesian_ridge(d: TrainData) -> Fitted:
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import BayesianRidge
    from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

    cat_indices = list(range(len(d.cat_features)))
    num_indices = list(range(len(d.cat_features), len(d.cat_features) + len(d.num_features)))
    print(f"  ✓ Categorical indices: {cat_indices}")
    print(f"  ✓ Numerical indices: {num_indices}")

    print("\n🧠 Building Bayesian Ridge pipeline (OneHot + StandardScaler + PolynomialFeatures)...")
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
                ("regressor", BayesianRidge(max_iter=300, alpha_init=1.0, lambda_init=1.0, compute_score=True)),
            ]
        )

    print("\n🔍 Hyperparameter tuning with GridSearchCV...")
    param_grid = {
        "poly__degree": [1, 2],
        "regressor__alpha_init": [0.1, 1.0, 10.0],
        "regressor__lambda_init": [0.1, 1.0, 10.0],
    }
    gs_t = _grid_search(build_model(), param_grid)
    gs_r = _grid_search(build_model(), param_grid)
    gs_t.fit(d.X_train, d.y_log_train[THROUGHPUT].to_numpy())
    gs_r.fit(d.X_train, d.y_log_train[RUNTIME].to_numpy())
    report_grid_search(gs_t)

    best_t, best_r = gs_t.best_estimator_, gs_r.best_estimator_
    reg_t = best_t.named_steps["regressor"]
    reg_r = best_r.named_steps["regressor"]

    def predict(X):
        t_pred, t_std = best_t.predict(X, return_std=True)
        r_pred, r_std = best_r.predict(X, return_std=True)
        return np.column_stack([t_pred, r_pred]), np.column_stack([t_std, r_std])

    return Fitted(
        model={"throughput": best_t, "runtime_seconds": best_r},
        predict=predict,
        metadata={
            "cat_indices": cat_indices,
            "num_indices": num_indices,
            "best_params": {"throughput": gs_t.best_params_, "runtime_seconds": gs_r.best_params_},
            "alpha": float(reg_t.alpha_),
            "lambda": float(reg_t.lambda_),
            "alpha_by_target": {"throughput": float(reg_t.alpha_), "runtime_seconds": float(reg_r.alpha_)},
            "lambda_by_target": {"throughput": float(reg_t.lambda_), "runtime_seconds": float(reg_r.lambda_)},
        },
        finalize=_uncertainty_finalize,
    )


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #

PERFORMANCE_MODELS: dict[str, ModelSpec] = {
    "xgboost": ModelSpec(
        stem="xgboost",
        title="XGBOOST PREDICTOR TRAINING",
        ready_message="🚀 XGBoost predictor ready for inference!",
        target_range="8-14%",
        target_threshold=14.0,
        encoding=Encoding.LABEL,
        fit=_fit_xgboost,
        artifact_keys=_BASE_LABEL | {"best_params", "feature_importance"},
    ),
    "lightgbm": ModelSpec(
        stem="lightgbm",
        title="LIGHTGBM PREDICTOR TRAINING",
        ready_message="🚀 LightGBM predictor ready for inference!",
        target_range="8-14%",
        target_threshold=14.0,
        encoding=Encoding.LABEL,
        fit=_fit_lightgbm,
        artifact_keys=_BASE_LABEL | {"best_params", "feature_importance"},
    ),
    "catboost": ModelSpec(
        stem="catboost",
        title="CATBOOST PREDICTOR TRAINING",
        ready_message="🚀 CatBoost predictor ready for inference!",
        target_range="6-12%",
        target_threshold=12.0,
        encoding=Encoding.RAW,
        fit=_fit_catboost,
        artifact_keys=_BASE_RAW | {"cat_feature_indices", "best_params", "best_params_runtime", "feature_importance"},
    ),
    "random_forest": ModelSpec(
        stem="random_forest",
        title="RANDOMFOREST PREDICTOR TRAINING",
        ready_message="🚀 RandomForest predictor ready for inference!",
        target_range="8-15%",
        target_threshold=15.0,
        encoding=Encoding.LABEL,
        fit=_fit_random_forest,
        runtime_unit="sec",
        artifact_keys=_BASE_LABEL | {"best_params", "oob_score"},
    ),
    "svr": ModelSpec(
        stem="svr",
        title="SVR PREDICTOR TRAINING",
        ready_message="🚀 SVR predictor ready for inference!",
        target_range="10-18%",
        target_threshold=18.0,
        encoding=Encoding.LABEL,
        fit=_fit_svr,
        runtime_unit="sec",
        artifact_keys=_BASE_LABEL | {"best_params"},
    ),
    "knn": ModelSpec(
        stem="knn",
        title="KNN PREDICTOR TRAINING",
        ready_message="🚀 KNN predictor ready for inference!",
        target_range="<20%",
        target_threshold=20.0,
        target_strict=True,
        encoding=Encoding.LABEL,
        fit=_fit_knn,
        runtime_unit="sec",
        artifact_keys=_BASE_LABEL | {"best_params", "refit_on_trainval"},
    ),
    "gaussian_process": ModelSpec(
        stem="gaussian_process",
        title="GAUSSIAN PROCESS PREDICTOR TRAINING",
        ready_message="🚀 Gaussian Process predictor ready for inference with uncertainty!",
        target_range="12-20%",
        target_threshold=20.0,
        encoding=Encoding.LABEL,
        fit=_fit_gaussian_process,
        artifact_keys=_BASE_LABEL | {"kernel", "uncertainty_correlation", "uncertainty_correlation_by_target"},
    ),
    "bayesian_ridge": ModelSpec(
        stem="bayesian_ridge",
        title="BAYESIAN RIDGE PREDICTOR TRAINING",
        ready_message="🚀 Bayesian Ridge predictor ready for inference with uncertainty!",
        target_range="15-25%",
        target_threshold=25.0,
        encoding=Encoding.RAW,
        fit=_fit_bayesian_ridge,
        artifact_keys=_BASE_RAW
        | {
            "cat_indices",
            "num_indices",
            "best_params",
            "alpha",
            "lambda",
            "alpha_by_target",
            "lambda_by_target",
            "uncertainty_correlation",
            "uncertainty_correlation_by_target",
        },
    ),
}
