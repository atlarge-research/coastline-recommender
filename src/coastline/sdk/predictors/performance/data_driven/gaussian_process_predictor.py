"""Gaussian Process predictor (#8 in portfolio): featv3 pickle, log1p target, optional uncertainty."""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.recommendation import Prediction  # noqa: F401  (return type annotation)
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.base import BasePredictor
from coastline.sdk.predictors.performance.data_driven.ml_common import (
    build_encoded_features,
    feature_row_has_unknown_specs,
    finalize_ml_prediction,
    performance_trained_model_path,
    workload_to_ml_feature_row,
)

logger = logging.getLogger(__name__)

_MODEL_PATH = performance_trained_model_path("gaussian_process")


class GaussianProcessPredictor(BasePredictor):
    """Gaussian Process predictor for dataset_tokens_per_second (with uncertainty)."""

    def __init__(self, model_path: Optional[Path] = None):
        self._model_path = model_path or _MODEL_PATH
        self._model = None
        self._encoders = None
        self._cat_features = None
        self._num_features = None
        self._test_metrics = None
        self._kernel = None
        self._uncertainty_correlation = None
        self._loaded = False

    def _load(self):
        """Lazy-load model and preprocessing artifacts."""
        if self._loaded:
            return

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Gaussian Process model not found at {self._model_path}. "
                "Train it first: python -m trainer.main --model gaussian_process"
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

            self._encoders = artifacts["encoders"]
            self._cat_features = artifacts["cat_features"]
            self._num_features = artifacts["num_features"]
            self._test_metrics = artifacts.get("test_metrics", {})
            self._kernel = artifacts.get("kernel", "N/A")
            self._uncertainty_correlation = artifacts.get("uncertainty_correlation", 0.0)

            self._loaded = True

            logger.info(f"Gaussian Process model loaded from {self._model_path}")
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
                logger.info(f"  Uncertainty-correlation: {self._uncertainty_correlation:.4f}")
            if self._kernel:
                logger.info(f"  Kernel: {self._kernel}")

        except Exception as e:
            logger.error(f"Failed to load Gaussian Process model: {e}")
            raise

    def predict(self, workload: WorkloadSpec, context: SystemContext, return_std: bool = False) -> Optional[Prediction]:
        """Predict throughput; return_std=True adds uncertainty to metadata. Returns None on load failure."""
        try:
            self._load()
        except Exception as e:
            logger.warning(f"Gaussian Process predictor unavailable: {e}")
            return None

        model = self._model
        encoders = self._encoders
        cat_features = self._cat_features
        num_features = self._num_features
        kernel = self._kernel if self._kernel is not None else "N/A"
        uncertainty_correlation = self._uncertainty_correlation if self._uncertainty_correlation is not None else 0.0
        if model is None or encoders is None or cat_features is None or num_features is None:
            logger.warning("Gaussian Process predictor artifacts are incomplete")
            return None

        row = workload_to_ml_feature_row(workload)
        if feature_row_has_unknown_specs(row):
            logger.info("Gaussian Process: unknown model/GPU specs, cannot predict")
            return None

        X = build_encoded_features(row, encoders, cat_features, num_features)

        if isinstance(model, dict):
            throughput_model = model["throughput"]
            gp_throughput = throughput_model.named_steps["gp"]
            scaler_throughput = throughput_model.named_steps["scaler"]
            X_scaled_throughput = scaler_throughput.transform(X)

            if return_std:
                y_log_pred_throughput, y_log_std_throughput = gp_throughput.predict(
                    X_scaled_throughput, return_std=True
                )
                y_log_std_value = float(y_log_std_throughput[0])
            else:
                y_log_pred_throughput = gp_throughput.predict(X_scaled_throughput)
                y_log_std_value = None

            throughput = float(np.expm1(y_log_pred_throughput[0]))

            runtime_key = "runtime_seconds" if "runtime_seconds" in model else "runtime"
            if runtime_key in model:
                runtime_model = model[runtime_key]
                gp_runtime = runtime_model.named_steps["gp"]
                scaler_runtime = runtime_model.named_steps["scaler"]
                X_scaled_runtime = scaler_runtime.transform(X)
                y_log_pred_runtime = gp_runtime.predict(X_scaled_runtime)
                runtime_seconds = float(np.expm1(y_log_pred_runtime[0]))
            else:
                runtime_seconds = None
        else:
            gp_model = model.named_steps["gp"]
            scaler = model.named_steps["scaler"]
            X_scaled = scaler.transform(X)

            if return_std:
                y_log_pred, y_log_std = gp_model.predict(X_scaled, return_std=True)
                y_log_std_value = float(y_log_std[0])
            else:
                y_log_pred = gp_model.predict(X_scaled)
                y_log_std_value = None

            throughput = float(np.expm1(y_log_pred[0]))
            runtime_seconds = None

        metadata = {
            "predictor": "gaussian_process",
            "kernel": str(kernel),
            "algorithm": "gaussian_process",
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
        return "gaussian_process"
