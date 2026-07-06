"""CatBoost predictor (#4 in portfolio): featv3 pickle, log1p target, native categoricals."""

import logging
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd

from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.recommendation import Prediction  # noqa: F401  (return type annotation)
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.base import BasePredictor
from coastline.sdk.predictors.performance.data_driven.ml_common import (
    feature_row_has_unknown_specs,
    finalize_ml_prediction,
    invert_log_targets,
    performance_trained_model_path,
    workload_to_ml_feature_row,
)

logger = logging.getLogger(__name__)

_MODEL_PATH = performance_trained_model_path("catboost")


class CatBoostPredictor(BasePredictor):
    """CatBoost predictor for dataset_tokens_per_second."""

    def __init__(self, model_path: Optional[Path] = None):
        self._model_path = model_path or _MODEL_PATH
        self._model = None
        self._cat_features = None
        self._num_features = None
        self._test_metrics = None
        self._best_params = None
        self._feature_importance = None
        self._loaded = False

    def _load(self):
        """Lazy-load model and preprocessing artifacts."""
        if self._loaded:
            return

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"CatBoost model not found at {self._model_path}. "
                "Train it first: python -m trainer.main --model catboost  (dev/ on PYTHONPATH)"
            )

        try:
            # The committed artifact was pickled under the pre-refactor top-level module path
            # "trainer.train_performance_catboost". Alias that legacy path to the class's current
            # home so the pickle resolves with no retrain — synthetic, so it works in a wheel
            # install with no dev/ trainer on the path. setdefault leaves a real dev trainer intact.
            import sys
            import types

            from coastline.sdk.predictors.performance.data_driven import _catboost_model

            sys.modules.setdefault("trainer", types.ModuleType("trainer"))
            if "trainer.train_performance_catboost" not in sys.modules:
                _shim = types.ModuleType("trainer.train_performance_catboost")
                _shim._DualOutputCatBoost = _catboost_model._DualOutputCatBoost
                sys.modules["trainer.train_performance_catboost"] = _shim

            with open(self._model_path, "rb") as f:
                artifacts = pickle.load(f)

            self._model = artifacts["model"]
            self._cat_features = artifacts["cat_features"]
            self._num_features = artifacts["num_features"]
            self._test_metrics = artifacts.get("test_metrics", {})
            self._best_params = artifacts.get("best_params", {})
            self._feature_importance = artifacts.get("feature_importance", [])

            self._loaded = True

            logger.info(f"CatBoost model loaded from {self._model_path}")
            if self._test_metrics:
                mdape = self._test_metrics.get("original_space", {}).get("mdape", "N/A")
                r2 = self._test_metrics.get("original_space", {}).get("r2", "N/A")
                logger.info(f"  Test MdAPE: {mdape}%, R²: {r2}")
            if self._best_params:
                logger.info(
                    f"  Best params: iterations={self._best_params.get('iterations')}, "
                    f"depth={self._best_params.get('depth')}, "
                    f"learning_rate={self._best_params.get('learning_rate')}"
                )

        except Exception as e:
            logger.error(f"Failed to load CatBoost model: {e}")
            raise

    def predict(self, workload: WorkloadSpec, context: SystemContext) -> Optional[Prediction]:
        """Predict throughput for a workload, or None if the model can't load."""
        try:
            self._load()
        except Exception as e:
            logger.warning(f"CatBoost predictor unavailable: {e}")
            return None

        model = self._model
        cat_features = self._cat_features
        num_features = self._num_features
        best_params = self._best_params or {}
        if model is None or cat_features is None or num_features is None:
            logger.warning("CatBoost predictor artifacts are incomplete")
            return None

        row_dict = workload_to_ml_feature_row(workload)
        if feature_row_has_unknown_specs(row_dict):
            logger.info("catboost: unknown model/GPU specs, cannot predict")
            return None

        # CatBoost uses native categorical support, so pass raw values (no encoders).
        X_cat = pd.DataFrame({col: [row_dict[col]] for col in cat_features})
        X_num = pd.DataFrame({col: [row_dict[col]] for col in num_features})
        X = pd.concat([X_cat, X_num], axis=1)

        throughput, runtime_seconds = invert_log_targets(model.predict(X))

        return finalize_ml_prediction(
            workload,
            throughput=throughput,
            runtime_seconds=runtime_seconds,
            metadata={
                "predictor": "catboost",
                "iterations": best_params.get("iterations", "N/A"),
                "depth": best_params.get("depth", "N/A"),
                "learning_rate": best_params.get("learning_rate", "N/A"),
                "l2_leaf_reg": best_params.get("l2_leaf_reg", "N/A"),
                "algorithm": "gradient_boosting",
                "cache_hit": False,
                "dual_output": runtime_seconds is not None,
            },
        )

    def get_name(self) -> str:
        return "catboost"
