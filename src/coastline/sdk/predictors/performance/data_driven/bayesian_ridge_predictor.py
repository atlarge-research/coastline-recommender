"""Bayesian Ridge predictor (#9 in portfolio): featv3 pickle, log1p target, optional uncertainty."""

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
    performance_trained_model_path,
    workload_to_ml_feature_row,
)

logger = logging.getLogger(__name__)

_MODEL_PATH = performance_trained_model_path("bayesian_ridge")


class BayesianRidgePredictor(BasePredictor):
    """Bayesian Ridge predictor for dataset_tokens_per_second (with uncertainty)."""

    def __init__(self, model_path: Optional[Path] = None):
        self._model_path = model_path or _MODEL_PATH
        self._model = None
        self._cat_features = None
        self._num_features = None
        self._cat_indices = None
        self._num_indices = None
        self._test_metrics = None
        self._best_params = None
        self._uncertainty_correlation = None
        self._alpha = None
        self._lambda = None
        self._loaded = False

    def _load(self):
        """Lazy-load model and preprocessing artifacts."""
        if self._loaded:
            return

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Bayesian Ridge model not found at {self._model_path}. "
                "Train it first: python -m trainer.main --model bayesian_ridge"
            )

        try:
            with open(self._model_path, "rb") as f:
                artifacts = pickle.load(f)

            # Two pickle formats coexist: a dict-of-models (throughput + runtime) or a
            # bare single-output throughput model. Normalise both to the dict form.
            model_artifact = artifacts["model"]
            if isinstance(model_artifact, dict):
                self._model = model_artifact
                self._is_dual_output = True
            else:
                self._model = {"throughput": model_artifact}
                self._is_dual_output = False

            self._cat_features = artifacts["cat_features"]
            self._num_features = artifacts["num_features"]
            self._cat_indices = artifacts["cat_indices"]
            self._num_indices = artifacts["num_indices"]
            self._test_metrics = artifacts.get("test_metrics", {})
            self._best_params = artifacts.get("best_params", {})
            self._uncertainty_correlation = artifacts.get("uncertainty_correlation", 0.0)
            self._alpha = artifacts.get("alpha", "N/A")
            self._lambda = artifacts.get("lambda", "N/A")

            self._loaded = True

            logger.info(f"Bayesian Ridge model loaded from {self._model_path}")
            logger.info(f"  Format: {'dual-output' if self._is_dual_output else 'single-output (legacy)'}")
            if self._test_metrics:
                # Handle both old single dict and new dict-of-dicts for test_metrics
                if self._is_dual_output and "throughput" in self._test_metrics:
                    mdape = self._test_metrics.get("throughput", {}).get("original_space", {}).get("mdape", "N/A")
                    r2 = self._test_metrics.get("throughput", {}).get("original_space", {}).get("r2", "N/A")
                else:
                    mdape = self._test_metrics.get("original_space", {}).get("mdape", "N/A")
                    r2 = self._test_metrics.get("original_space", {}).get("r2", "N/A")
                logger.info(f"  Test MdAPE: {mdape}%, R²: {r2}")
                logger.info(f"  Uncertainty-Error Correlation: {self._uncertainty_correlation:.4f}")
            if self._best_params:
                # Handle both old single dict and new dict-of-dicts for best_params
                if self._is_dual_output and "throughput" in self._best_params:
                    poly_degree = self._best_params.get("throughput", {}).get("poly__degree", "N/A")
                else:
                    poly_degree = self._best_params.get("poly__degree", "N/A")
                logger.info(f"  Polynomial degree: {poly_degree}")
                logger.info(f"  Alpha (noise precision): {self._alpha}")
                logger.info(f"  Lambda (weight precision): {self._lambda}")

        except Exception as e:
            logger.error(f"Failed to load Bayesian Ridge model: {e}")
            raise

    def predict(self, workload: WorkloadSpec, context: SystemContext, return_std: bool = False) -> Optional[Prediction]:
        """Predict throughput; return_std=True adds uncertainty to metadata. Returns None on load failure."""
        try:
            self._load()
        except Exception as e:
            logger.warning(f"Bayesian Ridge predictor unavailable: {e}")
            return None

        model = self._model
        best_params = self._best_params or {}
        alpha = self._alpha if self._alpha is not None else "N/A"
        lambda_value = self._lambda if self._lambda is not None else "N/A"
        uncertainty_correlation = self._uncertainty_correlation if self._uncertainty_correlation is not None else 0.0
        if model is None:
            logger.warning("Bayesian Ridge predictor artifacts are incomplete")
            return None

        X = pd.DataFrame([workload_to_ml_feature_row(workload)])

        if feature_row_has_unknown_specs(X):
            logger.info("Bayesian Ridge: unknown model/GPU specs, cannot predict")
            return None

        if isinstance(model, dict):
            throughput_model = model["throughput"]

            if return_std:
                y_log_pred_throughput, y_log_std_throughput = throughput_model.predict(X, return_std=True)
                y_log_std_value = float(y_log_std_throughput[0])
            else:
                y_log_pred_throughput = throughput_model.predict(X)
                y_log_std_value = None

            throughput = float(np.expm1(y_log_pred_throughput[0]))

            runtime_key = "runtime_seconds" if "runtime_seconds" in model else "runtime"
            if runtime_key in model:
                runtime_model = model[runtime_key]
                y_log_pred_runtime = runtime_model.predict(X)
                runtime_seconds = float(np.expm1(y_log_pred_runtime[0]))
            else:
                runtime_seconds = None
        else:
            if return_std:
                y_log_pred, y_log_std = model.predict(X, return_std=True)
                y_log_std_value = float(y_log_std[0])
            else:
                y_log_pred = model.predict(X)
                y_log_std_value = None

            throughput = float(np.expm1(y_log_pred[0]))
            runtime_seconds = None

        # Handle both old single dict and new dict-of-dicts for best_params
        if isinstance(model, dict) and "throughput" in best_params:
            poly_degree = best_params.get("throughput", {}).get("poly__degree", "N/A")
        else:
            poly_degree = best_params.get("poly__degree", "N/A")

        metadata = {
            "predictor": "bayesian_ridge",
            "polynomial_degree": poly_degree,
            "alpha": alpha,
            "lambda": lambda_value,
            "algorithm": "bayesian_linear_regression",
            "cache_hit": False,
            "dual_output": runtime_seconds is not None,
        }
        if y_log_std_value is not None:
            metadata["std"] = y_log_std_value
            metadata["uncertainty_correlation"] = uncertainty_correlation

        return finalize_ml_prediction(
            workload,
            throughput=throughput,
            runtime_seconds=runtime_seconds,
            metadata=metadata,
        )

    def get_name(self) -> str:
        return "bayesian_ridge"
