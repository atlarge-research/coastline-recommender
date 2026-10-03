"""Tests for the per-model artifact keys and the generic-trainer table in ``model_specs``.

No test trains a model. The tests check:

* the keys each model writes into its pickle, which the SDK predictors read;
* that the table covers the models the dispatch registry sends to the generic trainer;
* that xgboost, lightgbm and catboost are not imported at ``model_specs`` top level
  (this avoids the macOS OpenMP co-load crash);
* that the shared label encoder gives the same ids as a reference encoder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from .. import common as C
from ..generic_trainer import Encoding, ModelSpec
from ..model_specs import PERFORMANCE_MODELS

_METRICS = {"test_metrics", "val_metrics", "test_metrics_by_target", "val_metrics_by_target"}

# Expected artifact keys per model, written out by hand so the test does not read model_specs'
# _BASE_LABEL/_BASE_RAW.
GOLDEN_ARTIFACT_KEYS: dict[str, set[str]] = {
    "xgboost": _METRICS | {"model", "encoders", "cat_features", "num_features", "best_params", "feature_importance"},
    "lightgbm": _METRICS | {"model", "encoders", "cat_features", "num_features", "best_params", "feature_importance"},
    "catboost": _METRICS
    | {
        "model",
        "cat_features",
        "num_features",
        "cat_feature_indices",
        "best_params",
        "best_params_runtime",
        "feature_importance",
    },
    "random_forest": _METRICS | {"model", "encoders", "cat_features", "num_features", "best_params", "oob_score"},
    "svr": _METRICS | {"model", "encoders", "cat_features", "num_features", "best_params"},
    "knn": _METRICS | {"model", "encoders", "cat_features", "num_features", "best_params", "refit_on_trainval"},
    "gaussian_process": _METRICS
    | {
        "model",
        "encoders",
        "cat_features",
        "num_features",
        "kernel",
        "uncertainty_correlation",
        "uncertainty_correlation_by_target",
    },
    "bayesian_ridge": _METRICS
    | {
        "model",
        "cat_features",
        "num_features",
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
}

# The two RAW-encoding models store no "encoders" key (native / one-hot handling).
_RAW_MODELS = {"catboost", "bayesian_ridge"}


def test_table_covers_exactly_the_generic_backed_models():
    # The 10 dispatch models minus tabpfn and deep_learning, which have their own scripts.
    assert set(PERFORMANCE_MODELS) == set(GOLDEN_ARTIFACT_KEYS)
    assert len(PERFORMANCE_MODELS) == 8


@pytest.mark.parametrize("stem", sorted(GOLDEN_ARTIFACT_KEYS))
def test_artifact_keys_match_the_expected_keys(stem):
    """Each ModelSpec declares the artifact keys its model writes."""
    assert set(PERFORMANCE_MODELS[stem].artifact_keys) == GOLDEN_ARTIFACT_KEYS[stem]


@pytest.mark.parametrize("stem", sorted(GOLDEN_ARTIFACT_KEYS))
def test_encoders_key_present_iff_label_encoding(stem):
    """LABEL models store an ``encoders`` key and RAW models do not."""
    spec = PERFORMANCE_MODELS[stem]
    has_encoders = "encoders" in spec.artifact_keys
    assert has_encoders == (spec.encoding is Encoding.LABEL)
    assert has_encoders == (stem not in _RAW_MODELS)


@pytest.mark.parametrize("stem,spec", sorted(PERFORMANCE_MODELS.items()))
def test_spec_is_well_formed(stem, spec):
    assert isinstance(spec, ModelSpec)
    assert spec.stem == stem  # the artifact path is built from the stem
    assert callable(spec.fit)
    assert isinstance(spec.encoding, Encoding)
    assert spec.target_threshold > 0
    # Every model stores the four metric keys.
    assert _METRICS <= set(spec.artifact_keys)


def test_model_specs_does_not_import_heavy_backends_at_top_level():
    """Importing model_specs binds no XGBoost, LightGBM or CatBoost class at module level.

    These backends are imported inside each fit function to avoid the macOS OpenMP co-load crash.
    """
    from .. import model_specs

    for name in ("XGBRegressor", "LGBMRegressor", "CatBoostRegressor"):
        assert not hasattr(model_specs, name), f"{name} is imported at model_specs top level; keep it lazy"


# The shared label encoder and a reference encoder both fit on the train values plus "unknown"
# and map unseen val/test values to the 'unknown' id, so they give the same integer matrices.


def _reference_label_encode(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """A LabelEncoder per column that gives values unseen in train the 'unknown' id."""
    train_enc, val_enc, test_enc = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for col in train.columns:
        encoder = LabelEncoder()
        train_vals = list(train[col].unique()) + ["unknown"]
        encoder.fit(train_vals)
        unknown_idx = encoder.transform(["unknown"])[0]

        def safe_transform(values, _enc=encoder, _unk=unknown_idx):
            return [_enc.transform([v])[0] if v in _enc.classes_ else _unk for v in values]

        train_enc[col] = encoder.transform(train[col])
        val_enc[col] = safe_transform(val[col])
        test_enc[col] = safe_transform(test[col])
    return test_enc


def test_shared_label_encoder_matches_a_reference_encoder():
    train = pd.DataFrame(
        {
            "method": ["full", "lora", "full", "unknown"],  # 'unknown' already a train value
            "gpu_model": ["A100", "L40S", "A100", "H100"],
        }
    )
    val = pd.DataFrame({"method": ["full", "lora"], "gpu_model": ["A100", "L40S"]})
    test = pd.DataFrame({"method": ["lora", "zzz_unseen"], "gpu_model": ["H100", "brand_new_gpu"]})

    _, _, helper_test, _, _ = C.encode_categorical_features(train, val, test)
    reference_test = _reference_label_encode(train, val, test)

    for col in train.columns:
        np.testing.assert_array_equal(
            helper_test[col].to_numpy(), np.asarray(reference_test[col], dtype=int), err_msg=f"column {col} diverged"
        )
