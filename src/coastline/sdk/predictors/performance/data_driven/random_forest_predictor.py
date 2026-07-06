"""RandomForest predictor (#1 in portfolio): featv3 pickle, log1p target, LabelEncoder categoricals."""

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

_MODEL_PATH = performance_trained_model_path("random_forest")


class RandomForestPredictor(BasePredictor):
    """RandomForest predictor for dataset_tokens_per_second."""

    def __init__(self, model_path: Optional[Path] = None):
        self._model_path = model_path or _MODEL_PATH
        self._model = None
        self._encoders = None
        self._cat_features = None
        self._num_features = None
        self._oob_score = None
        self._test_metrics = None
        self._loaded = False

    def _load(self):
        """Lazy-load model and preprocessing artifacts."""
        if self._loaded:
            return

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"RandomForest model not found at {self._model_path}. "
                "Train it first: python -m trainer.main --model random_forest"
            )

        try:
            with open(self._model_path, "rb") as f:
                artifacts = pickle.load(f)

            self._model = artifacts["model"]
            self._encoders = artifacts["encoders"]
            self._cat_features = artifacts["cat_features"]
            self._num_features = artifacts["num_features"]
            self._oob_score = artifacts.get("oob_score")  # honest None when absent, no fabricated 0.85
            self._test_metrics = artifacts.get("test_metrics", {})

            self._loaded = True

            logger.info(f"RandomForest model loaded from {self._model_path}")
            if self._test_metrics:
                mdape = self._test_metrics.get("original_space", {}).get("mdape", "N/A")
                r2 = self._test_metrics.get("original_space", {}).get("r2", "N/A")
                logger.info(f"  Test MdAPE: {mdape}%, R²: {r2}")

        except Exception as e:
            logger.error(f"Failed to load RandomForest model: {e}")
            raise

    def predict(self, workload: WorkloadSpec, context: SystemContext) -> Optional[Prediction]:
        """Predict throughput or None on load failure (pipeline skips this candidate)."""
        try:
            self._load()
        except Exception as e:
            logger.warning(f"RandomForest unavailable, skipping prediction: {e}")
            return None

        model = self._model
        encoders = self._encoders
        cat_features = self._cat_features
        num_features = self._num_features
        oob_score = self._oob_score
        if model is None or encoders is None or cat_features is None or num_features is None:
            logger.warning("RandomForest predictor artifacts are incomplete")
            return None

        row = workload_to_ml_feature_row(workload)
        if feature_row_has_unknown_specs(row):
            logger.info("random_forest: unknown model/GPU specs, cannot predict")
            return None

        X = build_encoded_features(row, encoders, cat_features, num_features)
        throughput, runtime_seconds = invert_log_targets(model.predict(X))

        return finalize_ml_prediction(
            workload,
            throughput=throughput,
            runtime_seconds=runtime_seconds,
            metadata={
                "predictor": "random_forest",
                "oob_score": oob_score,
                "cache_hit": False,
                "dual_output": runtime_seconds is not None,
            },
        )

    def get_name(self) -> str:
        return "random_forest"
