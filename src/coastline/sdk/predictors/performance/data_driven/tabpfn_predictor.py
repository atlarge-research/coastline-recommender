"""TabPFN predictor: loads trained TabPFN model (featv3 pickle) for inference."""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.recommendation import Prediction  # noqa: F401  (return type annotation)
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.base import BasePredictor
from coastline.sdk.predictors.performance.data_driven.ml_common import (
    feature_row_has_unknown_specs,
    finalize_ml_prediction,
    get_feature_lists,
    performance_trained_model_path,
    workload_to_ml_feature_row,
)

logger = logging.getLogger(__name__)


def _patch_tabpfn_sklearn_imputers(obj, _seen: set[int] | None = None) -> None:
    """Backfill attrs on pickled TabPFN/sklearn preprocessors (sklearn 1.8+ vs older pickles)."""
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
    """Backfill attrs missing when unpickling TabPFN across library versions."""
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


DEFAULT_MODEL_PATH = performance_trained_model_path("tabpfn")


class TabPFNPredictor(BasePredictor):
    """TabPFN predictor; passes raw string+numeric features (no encoding needed).

    Class-level cache memoizes predictions by (model_path, feature-row): the grid is
    re-scored per job and policy arm so most rows repeat. TabPFN forward pass dominates cost.
    """

    _prediction_cache: dict = {}

    def __init__(self, model_path: Optional[Path] = None):
        import os

        _env = os.environ.get("TABPFN_MODEL_PATH")
        self.model_path = model_path or (Path(_env) if _env else DEFAULT_MODEL_PATH)
        self._model = None

    def _load(self):
        """Lazy-load model from disk."""
        if self._model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"TabPFN model not found at {self.model_path}. Train it first: python -m trainer.main --model tabpfn"
            )

        with open(self.model_path, "rb") as f:
            self._model = pickle.load(f)

        logger.info(f"TabPFN model loaded from {self.model_path}")

    def predict(self, workload: WorkloadSpec, context: SystemContext) -> Optional[Prediction]:
        """Predict throughput for a workload, or None if the model can't load."""
        try:
            self._load()
        except Exception as e:  # missing/corrupt pickle, version skew, compat shims, etc.
            logger.warning(str(e))
            return None

        # Extract model from pickle dict (handles both old and new formats)
        if isinstance(self._model, dict) and "model" in self._model:
            model = self._model["model"]
        else:
            model = self._model

        if model is None:
            logger.warning("TabPFN predictor artifacts are incomplete")
            return None

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

        # Memoize by (model, feature-row): the candidate grid is re-scored for
        # every job and every policy arm, so most of these are exact repeats.
        ckey = (str(self.model_path), tuple(X_array[0].tolist()))
        _cached = type(self)._prediction_cache.get(ckey)
        if _cached is not None:
            throughput, runtime_seconds = _cached
        # Handle both old (single model) and new (dict with dual models) formats
        elif isinstance(model, dict) and "throughput" in model:
            # New format: separate models for throughput and runtime
            _tabpfn_regressor_compat(model["throughput"])
            _tabpfn_regressor_compat(model["runtime"])
            y_log_throughput = model["throughput"].predict(X_array)
            y_log_runtime = model["runtime"].predict(X_array)
            throughput = float(np.expm1(y_log_throughput[0]))
            runtime_seconds = float(np.expm1(y_log_runtime[0]))
            type(self)._prediction_cache[ckey] = (throughput, runtime_seconds)
        else:
            # Old format: single model (may have multi-output)
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
