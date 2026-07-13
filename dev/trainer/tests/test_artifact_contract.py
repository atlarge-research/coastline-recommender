"""Characterization of the per-model artifact contract + generic-trainer table.

The eight sklearn-family trainers collapsed into one generic driver
(``generic_trainer.run_training``) plus a config table (``model_specs``). No test
trains a model — training needs the large curated CSV, the native backends, and
minutes of compute — so this file pins the parts that CAN drift silently:

* the exact set of keys each model writes into its pickle (the metadata contract
  the SDK predictors load at inference), captured here INDEPENDENTLY from the old
  per-model scripts, then checked against each ``ModelSpec.artifact_keys``;
* the table covers exactly the models the dispatch registry routes to the generic;
* the lazy-import discipline the macOS OpenMP co-load crash depends on — the heavy
  backends must NOT be imported at ``model_specs`` top level;
* the label-encoding switch (xgboost / lightgbm moved off their inline
  LabelEncoder loop onto the shared ``encode_categorical_features`` helper) is
  byte-for-byte behavior-preserving.
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

# Golden artifact key sets, transcribed by hand from the pre-collapse
# ``train_performance_*.py`` scripts (their ``artifacts = {...}`` dicts). This is the
# oracle: an INDEPENDENT source from model_specs' own _BASE_LABEL/_BASE_RAW unions.
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
    # The generic table = the 10 dispatch models minus the two distinct runtimes
    # (tabpfn in-context, deep_learning torch MLP) that keep their own scripts.
    assert set(PERFORMANCE_MODELS) == set(GOLDEN_ARTIFACT_KEYS)
    assert len(PERFORMANCE_MODELS) == 8


@pytest.mark.parametrize("stem", sorted(GOLDEN_ARTIFACT_KEYS))
def test_artifact_keys_match_the_pre_collapse_scripts(stem):
    """Each ModelSpec must declare exactly the artifact keys its old script wrote."""
    assert set(PERFORMANCE_MODELS[stem].artifact_keys) == GOLDEN_ARTIFACT_KEYS[stem]


@pytest.mark.parametrize("stem", sorted(GOLDEN_ARTIFACT_KEYS))
def test_encoders_key_present_iff_label_encoding(stem):
    """LABEL models persist their LabelEncoders; RAW models never carry an
    ``encoders`` key (the generic only adds it when encoders exist)."""
    spec = PERFORMANCE_MODELS[stem]
    has_encoders = "encoders" in spec.artifact_keys
    assert has_encoders == (spec.encoding is Encoding.LABEL)
    assert has_encoders == (stem not in _RAW_MODELS)


@pytest.mark.parametrize("stem,spec", sorted(PERFORMANCE_MODELS.items()))
def test_spec_is_well_formed(stem, spec):
    assert isinstance(spec, ModelSpec)
    assert spec.stem == stem  # key and stem agree -> the artifact lands at the right path
    assert callable(spec.fit)
    assert isinstance(spec.encoding, Encoding)
    assert spec.target_threshold > 0
    # Every model always stores the four metric keys.
    assert _METRICS <= set(spec.artifact_keys)


def test_model_specs_does_not_import_heavy_backends_at_top_level():
    """The macOS OpenMP co-load crash is avoided by importing xgboost / lightgbm /
    catboost INSIDE each fit function, never at module scope. Guard that discipline:
    importing the table must not bind these classes as module attributes."""
    from .. import model_specs

    for name in ("XGBRegressor", "LGBMRegressor", "CatBoostRegressor"):
        assert not hasattr(model_specs, name), f"{name} is imported at model_specs top level; keep it lazy"


# ---------------------------------------------------------------------------
# The label-encoding switch is behavior-preserving.
#
# xgboost / lightgbm used to encode categoricals with an inline LabelEncoder loop;
# they now share ``common.encode_categorical_features``. Both fit on
# {train uniques} ∪ {"unknown"} and map unseen val/test values to the 'unknown'
# id, so the encoded integer matrices must be identical. Reproduce the OLD inline
# logic here and assert equality — a regression in either path would diverge.
# ---------------------------------------------------------------------------


def _old_inline_label_encode(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """The pre-collapse inline encoder from train_performance_xgboost, verbatim."""
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


def test_shared_helper_matches_old_inline_label_encoder():
    train = pd.DataFrame(
        {
            "method": ["full", "lora", "full", "unknown"],  # 'unknown' already a train value
            "gpu_model": ["A100", "L40S", "A100", "H100"],
        }
    )
    val = pd.DataFrame({"method": ["full", "lora"], "gpu_model": ["A100", "L40S"]})
    test = pd.DataFrame({"method": ["lora", "zzz_unseen"], "gpu_model": ["H100", "brand_new_gpu"]})

    _, _, helper_test, _, _ = C.encode_categorical_features(train, val, test)
    old_test = _old_inline_label_encode(train, val, test)

    for col in train.columns:
        np.testing.assert_array_equal(
            helper_test[col].to_numpy(), np.asarray(old_test[col], dtype=int), err_msg=f"column {col} diverged"
        )
