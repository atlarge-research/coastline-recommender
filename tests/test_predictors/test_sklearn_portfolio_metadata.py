"""Characterization tests for the sklearn-style portfolio predictors' metadata contract.

The six featv3 portfolio models (catboost, xgboost, lightgbm, random_forest, svr, knn)
share one inference path; their ONLY per-model difference is the ``metadata`` dict
``predict()`` stamps. These tests pin that dict EXACTLY (keys, values, and order) for
each model, driving the real ``_load`` + ``predict`` through a FAKE pickle (no native
runtime, no real artifact) so they run in the default suite — the real-pickle contract
lives in ``test_ml_predictors.py`` (``-m ml_isolated``, own process).

They are written to survive the collapse of the six per-model classes into one generic
``SklearnPortfolioPredictor``: construction goes through the production resolver
(``_build_named_ml_predictor``), and the metadata oracle is the documented dict, not an
implementation detail — so the same file is green before and after the refactor.
"""

import pickle
import sys
import types

import numpy as np
import pytest

from coastline.sdk.policies import _build_named_ml_predictor
from coastline.sdk.predictors.performance.data_driven.ml_common import get_feature_lists

_MISSING = object()


class _FakeModel:
    """Returns a fixed log-space prediction, ignoring X. Two targets -> (throughput,
    runtime) dual output; one target -> throughput only (runtime None)."""

    def __init__(self, values):
        self._values = values

    def predict(self, X):
        if len(self._values) == 2:
            t, r = self._values
            return np.array([[np.log1p(t), np.log1p(r)]])
        (t,) = self._values
        return np.array([np.log1p(t)])


class _All:
    def __contains__(self, _):
        return True


class _FakeEncoder:
    """LabelEncoder stand-in: every value is 'known' and maps to 0."""

    classes_ = _All()

    def transform(self, values):
        return [0 for _ in values]


def _fake_artifacts(best_params, *, dual=True, oob_score=_MISSING):
    cat_features, num_features = get_feature_lists()
    values = (1000.0, 50.0) if dual else (1000.0,)
    artifacts = {
        "model": _FakeModel(values),
        "encoders": {c: _FakeEncoder() for c in cat_features},
        "cat_features": cat_features,
        "num_features": num_features,
        "best_params": best_params,
        "test_metrics": {},
    }
    if oob_score is not _MISSING:
        artifacts["oob_score"] = oob_score
    return artifacts


def _predict(monkeypatch, tmp_path, name, artifacts, workload, context):
    """Build the named predictor via the production resolver and drive its real
    _load + predict against a faked pickle."""
    # catboost's _load aliases a legacy dev-trainer module for the pickle; pre-register
    # both keys through monkeypatch so the alias is a no-op AND is restored (no sys.modules leak).
    monkeypatch.setitem(sys.modules, "trainer", types.ModuleType("trainer"))
    shim = types.ModuleType("trainer.train_performance_catboost")
    shim._DualOutputCatBoost = object
    monkeypatch.setitem(sys.modules, "trainer.train_performance_catboost", shim)
    monkeypatch.setattr(pickle, "load", lambda f: artifacts)

    predictor = _build_named_ml_predictor(name)
    path = tmp_path / "model.pkl"
    path.write_bytes(b"x")
    predictor._model_path = path
    return predictor


_FULL_PARAMS = {
    "n_estimators": 111,
    "max_depth": 7,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "iterations": 500,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "svr__C": 10.0,
    "svr__epsilon": 0.1,
    "svr__gamma": "scale",
    "knn__n_neighbors": 5,
    "knn__weights": "distance",
    "knn__p": 2,
}

# name -> (expected metadata dict for a dual-output prediction, oob_score to inject).
_EXPECTED = {
    "xgboost": (
        {
            "predictor": "xgboost",
            "n_estimators": 111,
            "max_depth": 7,
            "learning_rate": 0.05,
            "algorithm": "gradient_boosting",
            "cache_hit": False,
            "dual_output": True,
        },
        _MISSING,
    ),
    "lightgbm": (
        {
            "predictor": "lightgbm",
            "n_estimators": 111,
            "max_depth": 7,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "algorithm": "gradient_boosting",
            "cache_hit": False,
            "dual_output": True,
        },
        _MISSING,
    ),
    "catboost": (
        {
            "predictor": "catboost",
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "algorithm": "gradient_boosting",
            "cache_hit": False,
            "dual_output": True,
        },
        _MISSING,
    ),
    "svr": (
        {
            "predictor": "svr",
            "kernel": "rbf",
            "C": 10.0,
            "epsilon": 0.1,
            "gamma": "scale",
            "cache_hit": False,
            "dual_output": True,
        },
        _MISSING,
    ),
    "knn": (
        {
            "predictor": "knn",
            "n_neighbors": 5,
            "weights": "distance",
            "p": 2,
            "metric": "minkowski",
            "cache_hit": False,
            "dual_output": True,
        },
        _MISSING,
    ),
    "random_forest": (
        {
            "predictor": "random_forest",
            "oob_score": 0.87,
            "cache_hit": False,
            "dual_output": True,
        },
        0.87,
    ),
}


@pytest.mark.parametrize("name", list(_EXPECTED))
def test_metadata_dict_is_exact_including_order(monkeypatch, tmp_path, name, known_workload, a100_context):
    expected_metadata, oob = _EXPECTED[name]
    artifacts = _fake_artifacts(_FULL_PARAMS, oob_score=oob)
    predictor = _predict(monkeypatch, tmp_path, name, artifacts, known_workload, a100_context)

    assert predictor.get_name() == name
    p = predictor.predict(known_workload, a100_context)
    assert p is not None
    # Exact dict AND key order (downstream JSON serialization depends on order).
    assert list(p.metadata.items()) == list(expected_metadata.items())
    # finalize_ml_prediction copies the GPU layout off the WorkloadSpec: 8 * 2 = 16.
    assert (p.gpus_per_node, p.number_of_nodes, p.total_gpus) == (8, 2, 16)
    assert p.predicted_throughput == pytest.approx(1000.0)
    assert p.predicted_runtime_seconds == pytest.approx(50.0)


def test_missing_best_params_default_to_na(monkeypatch, tmp_path, known_workload, a100_context):
    predictor = _predict(monkeypatch, tmp_path, "xgboost", _fake_artifacts({}), known_workload, a100_context)
    p = predictor.predict(known_workload, a100_context)
    assert p.metadata == {
        "predictor": "xgboost",
        "n_estimators": "N/A",
        "max_depth": "N/A",
        "learning_rate": "N/A",
        "algorithm": "gradient_boosting",
        "cache_hit": False,
        "dual_output": True,
    }


def test_random_forest_missing_oob_score_is_none(monkeypatch, tmp_path, known_workload, a100_context):
    predictor = _predict(monkeypatch, tmp_path, "random_forest", _fake_artifacts({}), known_workload, a100_context)
    p = predictor.predict(known_workload, a100_context)
    assert p.metadata == {
        "predictor": "random_forest",
        "oob_score": None,
        "cache_hit": False,
        "dual_output": True,
    }


def test_single_output_model_sets_dual_output_false(monkeypatch, tmp_path, known_workload, a100_context):
    artifacts = _fake_artifacts(_FULL_PARAMS, dual=False)
    predictor = _predict(monkeypatch, tmp_path, "xgboost", artifacts, known_workload, a100_context)
    p = predictor.predict(known_workload, a100_context)
    assert p.metadata["dual_output"] is False
    assert p.predicted_runtime_seconds is None
