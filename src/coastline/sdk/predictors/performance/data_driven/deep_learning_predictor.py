"""Deep Learning predictor: loads trained EmbeddingNN (featv3 pickle) for inference."""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.recommendation import Prediction  # noqa: F401  (return type annotation)
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.base import BasePredictor
from coastline.sdk.predictors.performance.data_driven.ml_common import (
    ModelNotShippedError,
    feature_row_has_unknown_specs,
    finalize_ml_prediction,
    model_error_prediction,
    model_not_shipped_error,
    performance_deep_learning_model_dir,
    workload_to_ml_feature_row,
)

from ._nn import EmbeddingNN

logger = logging.getLogger(__name__)

DL_MODEL_DIR = performance_deep_learning_model_dir()


class DeepLearningPredictor(BasePredictor):
    """Performance predictor backed by EmbeddingNN; runs on CUDA, MPS or CPU, whichever is available."""

    #: A torch forward pass: ~4 ms per prediction.
    EXPENSIVE = True

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or DL_MODEL_DIR
        self._model: Optional[EmbeddingNN] = None
        self._artifacts: Optional[dict] = None

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

    @property
    def model_path(self) -> Path:
        """Path to the model weights file (used by orchestrator to check existence)."""
        return self.model_dir / "performance_deep_learning.pth"

    def _load(self):
        """Lazy-load model and artifacts from disk."""
        if self._model is not None:
            return

        weights_path = self.model_dir / "performance_deep_learning.pth"
        artifacts_path = self.model_dir / "performance_deep_learning_artifacts.pkl"

        for path in (weights_path, artifacts_path):
            if not path.exists():
                raise model_not_shipped_error("deep_learning", path)

        with open(artifacts_path, "rb") as f:
            self._artifacts = pickle.load(f)

        checkpoint = torch.load(weights_path, map_location=self._device, weights_only=False)

        # No Gaussian input noise at inference.
        self._model = EmbeddingNN(
            embedding_dims=checkpoint["embedding_dims"],
            num_numerical_features=checkpoint["num_numerical_features"],
            hidden_dims=checkpoint["hidden_dims"],
            dropout_rate=checkpoint["dropout_rate"],
            noise_std=0.0,
        )
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(self._device)
        self._model.eval()

        logger.info(f"Deep Learning model loaded from {self.model_dir} on {self._device}")

    def predict(self, workload: WorkloadSpec, context: SystemContext) -> Optional[Prediction]:
        """Predict throughput for a workload, or None for a model or GPU missing from Kavier's library.

        Model files that cannot be loaded give a Prediction with no numbers and the reason in
        ``metadata['error_detail']``. A missing model file raises ModelNotShippedError.
        """
        try:
            self._load()
        except ModelNotShippedError:
            raise
        except Exception as e:  # corrupt artifacts, torch load errors, etc.
            logger.warning(str(e))
            return model_error_prediction(
                workload,
                model_name="deep_learning",
                detail=f"model artifact could not be loaded: {self.model_dir} ({e})",
            )

        model = self._model
        artifacts = self._artifacts
        if model is None or artifacts is None:
            logger.warning("Deep learning predictor artifacts are incomplete")
            return model_error_prediction(
                workload, model_name="deep_learning", detail=f"model artifact is incomplete: {self.model_dir}"
            )

        row = workload_to_ml_feature_row(workload)
        if feature_row_has_unknown_specs(row):
            logger.info("deep_learning: unknown model/GPU specs, cannot predict")
            return None

        cat_features = artifacts["cat_features"]
        num_features = artifacts["num_features"]
        encoders = artifacts["encoders"]
        scaler = artifacts["scaler"]

        cat_codes = []
        for col in cat_features:
            encoder = encoders[col]
            val = row.get(col, "unknown")
            if val in encoder.classes_:
                cat_codes.append(encoder.transform([val])[0])
            else:
                cat_codes.append(0)

        num_vals = [[row[f] for f in num_features]]
        num_scaled = scaler.transform(num_vals)

        X_cat = torch.LongTensor([cat_codes]).to(self._device)
        X_num = torch.FloatTensor(num_scaled).to(self._device)

        with torch.no_grad():
            y_log_pred = model(X_cat, X_num)
            y_pred = np.expm1(y_log_pred.cpu().numpy())
            if np.ndim(y_pred) > 1 and np.shape(y_pred)[1] > 1:
                throughput = float(y_pred[0][0])
                runtime_seconds = float(y_pred[0][1])
            else:
                throughput = float(np.asarray(y_pred).reshape(-1)[0])
                runtime_seconds = None

        return finalize_ml_prediction(
            workload,
            throughput=throughput,
            runtime_seconds=runtime_seconds,
            metadata={
                "predictor": "deep_learning",
                "cache_hit": False,
                "device": str(self._device),
                "model_dir": str(self.model_dir),
                "dual_output": runtime_seconds is not None,
            },
        )

    def get_name(self) -> str:
        return "DeepLearningPredictor"
