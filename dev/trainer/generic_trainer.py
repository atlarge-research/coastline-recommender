"""One generic trainer for the sklearn-family performance models.

Every ``train_performance_*`` script used to be its own ~250-line copy of the same
skeleton — load, split, encode, tune, score, save. That skeleton lives here once;
the per-model differences (estimator, hyperparameters, categorical handling,
artifact keys) are data in ``model_specs.PERFORMANCE_MODELS``. The two genuinely
distinct runtimes — TabPFN (in-context) and Deep Learning (torch MLP) — keep their
own scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from .common import (
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

THROUGHPUT = TARGET_COLUMNS["throughput"]
RUNTIME = TARGET_COLUMNS["runtime_seconds"]

# A predictor returns log-space predictions and, for the Bayesian models, a
# matching (n, 2) std array; the others return None for the std.
Prediction = tuple[np.ndarray, Optional[np.ndarray]]
Predict = Callable[[pd.DataFrame], Prediction]


class Encoding(Enum):
    """How categoricals reach the estimator."""

    LABEL = "label"  # LabelEncoder each column, concat with numerics
    RAW = "raw"  # concat raw categoricals (native / one-hot models handle them)


@dataclass
class TrainData:
    """Everything a per-model ``fit`` needs: encoded matrices + dual-target frames."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_val: pd.DataFrame
    y_test: pd.DataFrame
    y_log_train: pd.DataFrame
    y_log_val: pd.DataFrame
    y_log_test: pd.DataFrame
    cat_features: list[str]
    num_features: list[str]
    encoders: Optional[dict] = None  # LabelEncoders (LABEL encoding only), for the artifact


@dataclass
class FinalizeCtx:
    """Test-set context handed to the optional post-scoring ``finalize`` hook."""

    y_test: pd.DataFrame
    test_pred: np.ndarray  # original space, (n, 2)
    test_std: Optional[np.ndarray]  # log space, (n, 2) or None
    feature_names: list[str]


@dataclass
class Fitted:
    """What a per-model ``fit`` returns: the object to pickle, a predictor, and the
    model-specific artifact keys (everything except model / encoders / features /
    metrics, which the generic owns)."""

    model: Any
    predict: Predict
    metadata: dict = field(default_factory=dict)
    predict_val: Optional[Predict] = None  # KNN scores val on the train-only estimator
    finalize: Optional[Callable[[FinalizeCtx], dict]] = None  # GP / BayesianRidge uncertainty


@dataclass(frozen=True)
class ModelSpec:
    stem: str
    title: str
    ready_message: str
    target_range: str
    target_threshold: float
    encoding: Encoding
    fit: Callable[[TrainData], Fitted]
    artifact_keys: frozenset[str]
    runtime_unit: str = "tokens/sec"
    target_strict: bool = False  # KNN's goal is '<' the threshold, not '<='


# --------------------------------------------------------------------------- #
# Helpers the per-model fit functions reuse
# --------------------------------------------------------------------------- #


def importance_records(values: Any, feature_names: Any) -> list[dict]:
    """Feature importances as descending ``[{feature, importance}, ...]`` records."""
    return (
        pd.DataFrame({"feature": list(feature_names), "importance": values})
        .sort_values("importance", ascending=False)
        .to_dict("records")
    )


def report_grid_search(grid: Any, display_params: Optional[dict] = None) -> None:
    """Print the best GridSearchCV parameters and CV score."""
    print("\n✅ Hyperparameter tuning complete!")
    print("\n🏆 Best parameters:")
    for param, value in (display_params or grid.best_params_).items():
        print(f"  {param}: {value}")
    print(f"\n📊 Best CV MAE (log space): {-grid.best_score_:.4f}")


# --------------------------------------------------------------------------- #
# The generic pipeline
# --------------------------------------------------------------------------- #


def _bar(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def _load_split(spec: ModelSpec) -> TrainData:
    print("=" * 70)
    print(spec.title)
    print("=" * 70)

    print("\n📂 Loading data...")
    X_cat, X_num, y, cat_features, num_features = load_and_preprocess_data()
    y_log = transform_targets(y)

    print(f"  ✓ Loaded {len(y)} samples")
    print(f"  ✓ Throughput range: {y[THROUGHPUT].min():.0f} – {y[THROUGHPUT].max():.0f} tokens/sec")
    print(f"  ✓ Runtime range: {y[RUNTIME].min():.0f} – {y[RUNTIME].max():.0f} sec")
    print(f"  ✓ Categorical features: {cat_features}")
    print(f"  ✓ Numerical features: {num_features}")

    print("\n✂️  Splitting data...")
    (
        (Xc_tr, Xn_tr, y_tr, yl_tr),
        (Xc_va, Xn_va, y_va, yl_va),
        (Xc_te, Xn_te, y_te, yl_te),
    ) = split_data(X_cat, X_num, y, y_log)
    Xc_tr, Xn_tr, y_tr, yl_tr = as_dataframes(Xc_tr, Xn_tr, y_tr, yl_tr)
    Xc_va, Xn_va, y_va, yl_va = as_dataframes(Xc_va, Xn_va, y_va, yl_va)
    Xc_te, Xn_te, y_te, yl_te = as_dataframes(Xc_te, Xn_te, y_te, yl_te)
    print(f"  ✓ Train: {len(y_tr)} samples")
    print(f"  ✓ Val:   {len(y_va)} samples")
    print(f"  ✓ Test:  {len(y_te)} samples")

    X_train, X_val, X_test, encoders = _encode(spec.encoding, (Xc_tr, Xc_va, Xc_te), (Xn_tr, Xn_va, Xn_te))

    return TrainData(
        X_train, X_val, X_test, y_tr, y_va, y_te, yl_tr, yl_va, yl_te, cat_features, num_features, encoders
    )


def _encode(
    encoding: Encoding,
    cat: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    num: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[dict]]:
    Xc_tr, Xc_va, Xc_te = cat
    encoders = None
    if encoding is Encoding.LABEL:
        print("\n🔤 Encoding categorical features...")
        Xc_tr, Xc_va, Xc_te, encoders, vocab = encode_categorical_features(Xc_tr, Xc_va, Xc_te)
        for col, size in vocab.items():
            print(f"  ✓ {col}: {size} classes")

    print("\n🔗 Combining features...")
    Xn_tr, Xn_va, Xn_te = num
    X_train = pd.concat([Xc_tr.reset_index(drop=True), Xn_tr.reset_index(drop=True)], axis=1)
    X_val = pd.concat([Xc_va.reset_index(drop=True), Xn_va.reset_index(drop=True)], axis=1)
    X_test = pd.concat([Xc_te.reset_index(drop=True), Xn_te.reset_index(drop=True)], axis=1)
    print(f"  ✓ Feature matrix shape: {X_train.shape}")
    return X_train, X_val, X_test, encoders


def _score(name: str, predict: Predict, X: pd.DataFrame, y: pd.DataFrame, y_log: pd.DataFrame, runtime_unit: str):
    y_pred_log, std = predict(X)
    y_pred_log = np.asarray(y_pred_log)
    y_pred = inverse_transform_targets(y_pred_log)

    m_tput = calculate_metrics(
        y[THROUGHPUT].to_numpy(), y_pred[:, 0], y_log[THROUGHPUT].to_numpy(), y_pred_log[:, 0]
    )
    m_rt = calculate_metrics(
        y[RUNTIME].to_numpy(), y_pred[:, 1], y_log[RUNTIME].to_numpy(), y_pred_log[:, 1]
    )
    _bar(f"{name} SET EVALUATION")
    print_metrics(m_tput, f"{name.title()} (Throughput)")
    print_metrics(m_rt, f"{name.title()} (Runtime)", unit=runtime_unit)
    return m_tput, m_rt, y_pred, std


def _print_samples(y_test: pd.DataFrame, y_pred: np.ndarray, std: Optional[np.ndarray]) -> None:
    y_true = y_test[THROUGHPUT].to_numpy()
    has_std = std is not None
    print("\n🔍 Sample throughput predictions (first 10):")
    header = f"{'True':>12} {'Predicted':>12}" + (f" {'Std (log)':>12}" if has_std else "") + f" {'Error %':>10}"
    print(header)
    print("-" * (48 if has_std else 36))
    for i in range(min(10, len(y_true))):
        err = abs(y_true[i] - y_pred[i, 0]) / y_true[i] * 100
        std_col = f" {std[i, 0]:>12.4f}" if has_std else ""
        print(f"{y_true[i]:>12,.1f} {y_pred[i, 0]:>12,.1f}{std_col} {err:>9.1f}%")


def _print_feature_importance(records: list[dict]) -> None:
    _bar("FEATURE IMPORTANCE")
    print("\n📊 Top 10 most important features:")
    for row in records[:10]:
        print(f"  {row['feature']:20s}: {row['importance']:>8.4f}")


def _assemble(
    spec: ModelSpec,
    fitted: Fitted,
    encoders: Any,
    cat_features: list[str],
    num_features: list[str],
    val: tuple[dict, dict],
    test: tuple[dict, dict],
    extra: dict,
) -> dict:
    val_tput, val_rt = val
    test_tput, test_rt = test
    artifacts: dict = {"model": fitted.model}
    if encoders is not None:
        artifacts["encoders"] = encoders
    artifacts["cat_features"] = cat_features
    artifacts["num_features"] = num_features
    artifacts.update(fitted.metadata)
    artifacts.update(extra)
    artifacts["test_metrics"] = test_tput
    artifacts["val_metrics"] = val_tput
    artifacts["test_metrics_by_target"] = {"throughput": test_tput, "runtime_seconds": test_rt}
    artifacts["val_metrics_by_target"] = {"throughput": val_tput, "runtime_seconds": val_rt}

    if set(artifacts) != spec.artifact_keys:
        raise ValueError(
            f"{spec.stem}: artifact keys {sorted(artifacts)} != declared {sorted(spec.artifact_keys)}"
        )
    return artifacts


def _final_status(spec: ModelSpec, test_tput: dict, test_rt: dict, extra: dict) -> None:
    tput, rt = test_tput["original_space"], test_rt["original_space"]
    mdape = tput["mdape"]
    ok = mdape < spec.target_threshold if spec.target_strict else mdape <= spec.target_threshold

    _bar("TRAINING COMPLETE")
    status, emoji = ("✅ SUCCESS", "🎉") if ok else ("⚠️  NEEDS IMPROVEMENT", "🔧")
    print(f"\n{emoji} {status}")
    print(f"  Target MdAPE: {spec.target_range}")
    print(f"  Achieved throughput MdAPE: {mdape:.2f}%")
    print(f"  Achieved runtime MdAPE: {rt['mdape']:.2f}%")
    print(f"  Throughput Test R²: {tput['r2']:.4f}")
    print(f"  Runtime Test R²: {rt['r2']:.4f}")
    print(f"  Throughput Within 20%: {tput['within_20_pct']:.1f}%")
    print(f"  Runtime Within 20%: {rt['within_20_pct']:.1f}%")
    by_target = extra.get("uncertainty_correlation_by_target")
    if by_target:
        print(f"  Throughput Uncertainty Correlation: {by_target['throughput']:.4f}")
        print(f"  Runtime Uncertainty Correlation: {by_target['runtime_seconds']:.4f}")
    print(f"\n{spec.ready_message}")


def run_training(spec: ModelSpec) -> None:
    """Load → split → encode → fit → score → save, driven entirely by ``spec``."""
    data = _load_split(spec)

    fitted = spec.fit(data)

    val = _score(
        "VALIDATION", fitted.predict_val or fitted.predict, data.X_val, data.y_val, data.y_log_val, spec.runtime_unit
    )
    test = _score("TEST", fitted.predict, data.X_test, data.y_test, data.y_log_test, spec.runtime_unit)
    val_tput, val_rt, _, _ = val
    test_tput, test_rt, test_pred, test_std = test

    _print_samples(data.y_test, test_pred, test_std)

    if "feature_importance" in fitted.metadata:
        _print_feature_importance(fitted.metadata["feature_importance"])

    extra: dict = {}
    if fitted.finalize is not None:
        _bar("UNCERTAINTY ANALYSIS")
        extra = fitted.finalize(FinalizeCtx(data.y_test, test_pred, test_std, list(data.X_train.columns)))

    artifacts = _assemble(
        spec,
        fitted,
        data.encoders,
        data.cat_features,
        data.num_features,
        (val_tput, val_rt),
        (test_tput, test_rt),
        extra,
    )

    _bar("SAVING MODEL")
    model_path = performance_trained_model_path(spec.stem)
    new_mdape = float(test_tput["original_space"]["mdape"])
    _, save_msg = save_pickled_artifact_if_better(model_path, artifacts, new_mdape)
    print(f"💾 {save_msg}")
    print(f"  ✓ Test throughput MdAPE: {test_tput['original_space']['mdape']:.2f}%")
    print(f"  ✓ Test runtime MdAPE: {test_rt['original_space']['mdape']:.2f}%")
    print(f"  ✓ Test throughput R²: {test_tput['original_space']['r2']:.4f}")
    print(f"  ✓ Test runtime R²: {test_rt['original_space']['r2']:.4f}")

    _final_status(spec, test_tput, test_rt, extra)
