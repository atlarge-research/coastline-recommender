"""SVR predictor (#2 in portfolio): featv3 pickle, log1p target, LabelEncoder categoricals."""

import logging
import pickle
from pathlib import Path
from typing import Optional

from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.recommendation import Prediction  # noqa: F401  (return type annotation)
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.base import BasePredictor
from coastline.sdk.predictors.performance.data_driven.ml_common import (
    build_encoded_features,
    feature_row_has_unknown_specs,
    finalize_ml_prediction,
    invert_log_targets,
    performance_trained_model_path,
    workload_to_ml_feature_row,
)

logger = logging.getLogger(__name__)

_MODEL_PATH = performance_trained_model_path("svr")


class SVRPredictor(BasePredictor):
    """SVR predictor for dataset_tokens_per_second."""

    def __init__(self, model_path: Optional[Path] = None):
        self._model_path = model_path or _MODEL_PATH
        self._model = None  # Pipeline with scaler + SVR
        self._encoders = None
        self._cat_features = None
        self._num_features = None
        self._test_metrics = None
        self._best_params = None
        self._loaded = False

    def _load(self):
        """Lazy-load model and preprocessing artifacts."""
        if self._loaded:
            return

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"SVR model not found at {self._model_path}. Train it first: python -m trainer.main --model svr"
            )

        try:
            with open(self._model_path, "rb") as f:
                artifacts = pickle.load(f)

            self._model = artifacts["model"]  # Pipeline
            self._encoders = artifacts["encoders"]
            self._cat_features = artifacts["cat_features"]
            self._num_features = artifacts["num_features"]
            self._test_metrics = artifacts.get("test_metrics", {})
            self._best_params = artifacts.get("best_params", {})

            self._loaded = True

            logger.info(f"SVR model loaded from {self._model_path}")
            if self._test_metrics:
                mdape = self._test_metrics.get("original_space", {}).get("mdape", "N/A")
                r2 = self._test_metrics.get("original_space", {}).get("r2", "N/A")
                logger.info(f"  Test MdAPE: {mdape}%, R²: {r2}")
            if self._best_params:
                logger.info(
                    f"  Best params: C={self._best_params.get('svr__C')}, "
                    f"epsilon={self._best_params.get('svr__epsilon')}, "
                    f"gamma={self._best_params.get('svr__gamma')}"
                )

        except Exception as e:
            logger.error(f"Failed to load SVR model: {e}")
            raise

    def predict(self, workload: WorkloadSpec, context: SystemContext) -> Optional[Prediction]:
        """Predict throughput for a workload, or None if the model can't load."""
        try:
            self._load()
        except Exception as e:
            logger.warning(f"SVR predictor unavailable: {e}")
            return None

        model = self._model
        encoders = self._encoders
        cat_features = self._cat_features
        num_features = self._num_features
        best_params = self._best_params or {}
        if model is None or encoders is None or cat_features is None or num_features is None:
            logger.warning("SVR predictor artifacts are incomplete")
            return None

        row = workload_to_ml_feature_row(workload)
        if feature_row_has_unknown_specs(row):
            logger.info("SVR: unknown model/GPU specs, cannot predict")
            return None

        X = build_encoded_features(row, encoders, cat_features, num_features)
        throughput, runtime_seconds = invert_log_targets(model.predict(X))

        return finalize_ml_prediction(
            workload,
            throughput=throughput,
            runtime_seconds=runtime_seconds,
            metadata={
                "predictor": "svr",
                "kernel": "rbf",
                "C": best_params.get("svr__C", "N/A"),
                "epsilon": best_params.get("svr__epsilon", "N/A"),
                "gamma": best_params.get("svr__gamma", "N/A"),
                "cache_hit": False,
                "dual_output": runtime_seconds is not None,
            },
        )

    def get_name(self) -> str:
        return "svr"
