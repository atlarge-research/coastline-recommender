"""TabPFN predictor: loads trained TabPFN model (featv3 pickle) for inference."""

import logging
import pickle
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.recommendation import Prediction  # noqa: F401  (return type annotation)
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.base import BasePredictor
from coastline.sdk.predictors.performance.data_driven.ml_common import (
    ModelNotShippedError,
    feature_row_has_unknown_specs,
    finalize_ml_prediction,
    get_feature_lists,
    model_error_prediction,
    model_not_shipped_error,
    performance_trained_model_path,
    workload_to_ml_feature_row,
)

logger = logging.getLogger(__name__)


def _patch_tabpfn_sklearn_imputers(obj, _seen: set[int] | None = None) -> None:
    """Add attributes that sklearn 1.8+ expects to the imputers inside older TabPFN pickles."""
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return
    _seen.add(oid)

    if type(obj).__name__ == "_NoInverseImputer" and not hasattr(obj, "_fill_dtype"):
        obj._fill_dtype = np.float64

    if isinstance(obj, dict):
        for value in obj.values():
            _patch_tabpfn_sklearn_imputers(value, _seen)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _patch_tabpfn_sklearn_imputers(value, _seen)
    elif hasattr(obj, "__dict__"):
        for value in vars(obj).values():
            _patch_tabpfn_sklearn_imputers(value, _seen)


class _TabPFNEnsemblePreprocessorStub:
    """Minimal stand-in when executor_.ensemble_preprocessor was not pickled (TabPFN 8.x)."""

    @staticmethod
    def any_estimator_uses_gpu_svd() -> bool:
        return False


def _tabpfn_regressor_compat(model) -> None:
    """Add attributes that are missing when a TabPFN pickle is loaded by another library version."""
    if isinstance(model, dict) and "throughput" in model:
        for key in ("throughput", "runtime"):
            if key in model and model[key] is not None:
                _tabpfn_regressor_compat(model[key])
        return

    if not hasattr(model, "n_estimators_"):
        model.n_estimators_ = getattr(model, "n_estimators", 8)
    if not hasattr(model, "show_progress_bar"):
        model.show_progress_bar = False

    _patch_tabpfn_sklearn_imputers(model)

    executor = getattr(model, "executor_", None)
    if executor is not None and not hasattr(executor, "ensemble_preprocessor"):
        executor.ensemble_preprocessor = _TabPFNEnsemblePreprocessorStub()


def _load_pickle(path: Path) -> Any:
    """Unpickle ``path``, with one log line for a scikit-learn version mismatch.

    scikit-learn warns once per estimator class when the pickle was written by another version
    (the bundled tabpfn.pkl by 1.8.0). Those warnings become one INFO line with both versions;
    any other warning is passed on.
    """
    try:
        from sklearn.exceptions import InconsistentVersionWarning
    except ImportError:  # no scikit-learn, so no version check to quiet
        with open(path, "rb") as f:
            return pickle.load(f)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        with open(path, "rb") as f:
            model = pickle.load(f)
    skew = [w.message for w in caught if isinstance(w.message, InconsistentVersionWarning)]
    for w in caught:
        if not isinstance(w.message, InconsistentVersionWarning):
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno, source=w.source)
    if skew:
        logger.info(
            "%s was written by scikit-learn %s; this install has scikit-learn %s",
            path,
            ", ".join(sorted({w.original_sklearn_version for w in skew})),
            skew[0].current_sklearn_version,
        )
    return model


DEFAULT_MODEL_PATH = performance_trained_model_path("tabpfn")


class TabPFNPredictor(BasePredictor):
    """TabPFN predictor; takes the raw string and numeric features without encoding.

    A class-level cache keys predictions by (model_path, feature row). The grid is re-scored per
    job and policy arm, so most rows repeat, and the TabPFN forward pass dominates the cost.
    """

    #: False so this model is never forked, although it is the most expensive one in the portfolio.
    #: A repeated configuration comes from the per-process ``_prediction_cache``, so forking would
    #: lose every hit and make each worker warm the cache again. Keep it in one long-lived process.
    EXPENSIVE = False

    _prediction_cache: dict = {}

    def __init__(self, model_path: Optional[Path] = None):
        import os

        _env = os.environ.get("TABPFN_MODEL_PATH")
        self.model_path = model_path or (Path(_env) if _env else DEFAULT_MODEL_PATH)
        self._model = None

    def _load(self):
        """Lazy-load the model from disk."""
        if self._model is not None:
            return

        if not self.model_path.exists():
            raise model_not_shipped_error("tabpfn", self.model_path)

        self._model = _load_pickle(self.model_path)

        logger.info(f"TabPFN model loaded from {self.model_path}")

    def predict(self, workload: WorkloadSpec, context: SystemContext) -> Optional[Prediction]:
        """Predict throughput for a workload, or None for a model or GPU missing from Kavier's library.

        A model file that cannot be loaded, or holds no model, gives a Prediction with no numbers
        and the reason in ``metadata['error_detail']``. A missing model file raises
        ModelNotShippedError.
        """
        try:
            self._load()
        except ModelNotShippedError:
            raise
        except Exception as e:  # corrupt pickle, version skew, compat shims, etc.
            logger.warning(str(e))
            return model_error_prediction(
                workload, model_name="tabpfn", detail=f"model artifact could not be loaded: {self.model_path} ({e})"
            )

        # The pickle holds either a dict with a "model" key or the model itself.
        if isinstance(self._model, dict) and "model" in self._model:
            model = self._model["model"]
        else:
            model = self._model

        if model is None:
            logger.warning("TabPFN predictor artifacts are incomplete")
            return model_error_prediction(
                workload, model_name="tabpfn", detail=f"model artifact is incomplete: {self.model_path}"
            )

        cat_cols, num_feats = get_feature_lists()
        row = workload_to_ml_feature_row(workload)
        if feature_row_has_unknown_specs(row):
            logger.info("tabpfn: unknown model/GPU specs, cannot predict")
            return None
        cols = list(cat_cols) + list(num_feats)
        X = pd.DataFrame([row])[cols]
        for c in cat_cols:
            X[c] = X[c].astype(str)
        X_array = X.values.astype(object)
        for i, col in enumerate(X.columns):
            if col in num_feats:
                X_array[:, i] = X_array[:, i].astype(np.float64)

        # Cache by (model path, feature row): the candidate grid is re-scored for every job and
        # every policy arm, so most rows are exact repeats.
        ckey = (str(self.model_path), tuple(X_array[0].tolist()))
        _cached = type(self)._prediction_cache.get(ckey)
        if _cached is not None:
            throughput, runtime_seconds = _cached
        # The model is a dict of throughput and runtime models, or a single model.
        elif isinstance(model, dict) and "throughput" in model:
            # Separate models for throughput and runtime
            _tabpfn_regressor_compat(model["throughput"])
            _tabpfn_regressor_compat(model["runtime"])
            y_log_throughput = model["throughput"].predict(X_array)
            y_log_runtime = model["runtime"].predict(X_array)
            throughput = float(np.expm1(y_log_throughput[0]))
            runtime_seconds = float(np.expm1(y_log_runtime[0]))
            type(self)._prediction_cache[ckey] = (throughput, runtime_seconds)
        else:
            # A single model, possibly multi-output
            _tabpfn_regressor_compat(model)
            y_log_pred = model.predict(X_array)
            if np.ndim(y_log_pred) > 1:
                throughput = float(np.expm1(y_log_pred[0][0]))
                runtime_seconds = float(np.expm1(y_log_pred[0][1]))
            else:
                throughput = float(np.expm1(y_log_pred[0]))
                runtime_seconds = None
            type(self)._prediction_cache[ckey] = (throughput, runtime_seconds)

        return finalize_ml_prediction(
            workload,
            throughput=throughput,
            runtime_seconds=runtime_seconds,
            metadata={
                "predictor": "tabpfn",
                "cache_hit": False,
                "model_path": str(self.model_path),
                "dual_output": runtime_seconds is not None,
            },
        )

    def get_name(self) -> str:
        return "TabPFNPredictor"
