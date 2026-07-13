"""One predictor for the six sklearn-style portfolio models.

catboost, xgboost, lightgbm, random_forest, svr and knn are all thin wrappers over a
featv3 pickle with a log1p target. Their only differences are (a) the hyperparameters
they surface in ``metadata`` and (b) whether categoricals are native (catboost) or
LabelEncoded (the rest). Both are data, so one class + the per-model table in
``policies`` replaces the six near-identical modules.

Models with a distinct runtime (tabpfn, deep_learning) or a ``return_std`` path
(gaussian_process, bayesian_ridge) keep their own classes — they are not portfolio-shaped.
"""

import logging
import pickle
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import pandas as pd

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


@dataclass(frozen=True)
class Param:
    """A hyperparameter surfaced from the pickle's ``best_params`` (absent -> 'N/A')."""

    key: str
    source: Optional[str] = None

    def resolve(self, best_params: Dict[str, Any], artifacts: Dict[str, Any]) -> Any:
        return best_params.get(self.source or self.key, "N/A")


@dataclass(frozen=True)
class Const:
    """A fixed metadata value (e.g. ``algorithm='gradient_boosting'``)."""

    key: str
    value: Any

    def resolve(self, best_params: Dict[str, Any], artifacts: Dict[str, Any]) -> Any:
        return self.value


@dataclass(frozen=True)
class Artifact:
    """A value read from the pickle's top-level artifacts dict (absent -> None)."""

    key: str
    source: Optional[str] = None

    def resolve(self, best_params: Dict[str, Any], artifacts: Dict[str, Any]) -> Any:
        return artifacts.get(self.source or self.key)


MetaField = Union[Param, Const, Artifact]


def _alias_legacy_catboost_module() -> None:
    """The committed catboost pickle was serialized under the pre-refactor top-level
    module ``trainer.train_performance_catboost``. Alias that path to the shipped class so
    the pickle resolves with no retrain in a wheel install with no dev trainer on the path.
    ``setdefault`` leaves a real dev trainer intact."""
    from coastline.sdk.predictors.performance.data_driven import _catboost_model

    sys.modules.setdefault("trainer", types.ModuleType("trainer"))
    if "trainer.train_performance_catboost" not in sys.modules:
        shim = types.ModuleType("trainer.train_performance_catboost")
        shim._DualOutputCatBoost = _catboost_model._DualOutputCatBoost  # type: ignore[attr-defined]
        sys.modules["trainer.train_performance_catboost"] = shim


class SklearnPortfolioPredictor(BasePredictor):
    """Featv3 sklearn-style throughput predictor, configured by name + metadata fields."""

    def __init__(
        self,
        name: str,
        metadata_param_keys: Sequence[MetaField],
        native_categorical: bool = False,
        model_path: Optional[Path] = None,
    ):
        self._name = name
        self._metadata_fields = tuple(metadata_param_keys)
        self._native_categorical = native_categorical
        self._model_path = model_path or performance_trained_model_path(name)
        self._model = None
        self._encoders = None
        self._cat_features = None
        self._num_features = None
        self._static_metadata: Dict[str, Any] = {}
        self._loaded = False

    def _load(self) -> None:
        """Lazy-load model + preprocessing artifacts from the featv3 pickle."""
        if self._loaded:
            return

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"{self._name} model not found at {self._model_path}. "
                f"Train it first: python -m trainer.main --model {self._name}"
            )

        try:
            if self._native_categorical:
                _alias_legacy_catboost_module()

            with open(self._model_path, "rb") as f:
                artifacts = pickle.load(f)

            self._model = artifacts["model"]
            self._cat_features = artifacts["cat_features"]
            self._num_features = artifacts["num_features"]
            # catboost uses native categoricals, so it ships no encoders.
            self._encoders = None if self._native_categorical else artifacts["encoders"]

            best_params = artifacts.get("best_params", {})
            self._static_metadata = {f.key: f.resolve(best_params, artifacts) for f in self._metadata_fields}
            self._loaded = True

            logger.info("%s model loaded from %s", self._name, self._model_path)
            metrics = artifacts.get("test_metrics", {}).get("original_space", {})
            if metrics:
                logger.info("  Test MdAPE: %s%%, R2: %s", metrics.get("mdape", "N/A"), metrics.get("r2", "N/A"))

        except Exception as e:
            logger.error("Failed to load %s model: %s", self._name, e)
            raise

    def predict(self, workload: WorkloadSpec, context: SystemContext) -> Optional[Prediction]:
        """Predict throughput, or None on load failure / out-of-library workload
        (the pipeline skips that candidate)."""
        try:
            self._load()
        except Exception as e:
            logger.warning("%s unavailable, skipping prediction: %s", self._name, e)
            return None

        if self._model is None or self._cat_features is None or self._num_features is None:
            logger.warning("%s predictor artifacts are incomplete", self._name)
            return None

        row = workload_to_ml_feature_row(workload)
        if feature_row_has_unknown_specs(row):
            logger.info("%s: unknown model/GPU specs, cannot predict", self._name)
            return None

        if self._native_categorical:
            X_cat = pd.DataFrame({col: [row[col]] for col in self._cat_features})
            X_num = pd.DataFrame({col: [row[col]] for col in self._num_features})
            X = pd.concat([X_cat, X_num], axis=1)
        else:
            X = build_encoded_features(row, self._encoders, self._cat_features, self._num_features)

        throughput, runtime_seconds = invert_log_targets(self._model.predict(X))

        metadata = {
            "predictor": self._name,
            **self._static_metadata,
            "cache_hit": False,
            "dual_output": runtime_seconds is not None,
        }
        return finalize_ml_prediction(
            workload,
            throughput=throughput,
            runtime_seconds=runtime_seconds,
            metadata=metadata,
        )

    def get_name(self) -> str:
        return self._name


@dataclass(frozen=True)
class PortfolioModel:
    """One portfolio model's config — its metadata fields and categorical mode. The
    only per-model difference; the inference path is shared by ``SklearnPortfolioPredictor``."""

    metadata: tuple[MetaField, ...]
    native_categorical: bool = False

    def build(self, name: str) -> SklearnPortfolioPredictor:
        return SklearnPortfolioPredictor(name, self.metadata, native_categorical=self.native_categorical)
