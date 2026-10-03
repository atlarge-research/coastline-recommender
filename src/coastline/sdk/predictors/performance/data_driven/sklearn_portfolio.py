"""One predictor class for the six sklearn-style portfolio models.

catboost, xgboost, lightgbm, random_forest, svr and knn each load a featv3 pickle with a
log1p target. They differ only in the hyperparameters they report in ``metadata`` and in
whether categoricals are native (catboost) or LabelEncoded (the rest); the per-model table
is in ``policies``.

tabpfn and deep_learning (separate runtimes) and gaussian_process and bayesian_ridge (a
``return_std`` path) have their own classes.
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
    ModelNotShippedError,
    build_encoded_features,
    feature_row_has_unknown_specs,
    finalize_ml_prediction,
    invert_log_targets,
    model_error_prediction,
    model_not_shipped_error,
    performance_trained_model_path,
    workload_to_ml_feature_row,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Param:
    """A hyperparameter read from the pickle's ``best_params`` ('N/A' if absent)."""

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
    """A value read from the pickle's top-level artifacts dict (None if absent)."""

    key: str
    source: Optional[str] = None

    def resolve(self, best_params: Dict[str, Any], artifacts: Dict[str, Any]) -> Any:
        return artifacts.get(self.source or self.key)


MetaField = Union[Param, Const, Artifact]


def _alias_legacy_catboost_module() -> None:
    """Map ``trainer.train_performance_catboost``, the module the bundled catboost pickle was
    saved under, to the installed class, so the pickle loads without the dev trainer.
    An already imported dev trainer is left in place."""
    from coastline.sdk.predictors.performance.data_driven import _catboost_model

    sys.modules.setdefault("trainer", types.ModuleType("trainer"))
    if "trainer.train_performance_catboost" not in sys.modules:
        shim = types.ModuleType("trainer.train_performance_catboost")
        shim._DualOutputCatBoost = _catboost_model._DualOutputCatBoost  # type: ignore[attr-defined]
        sys.modules["trainer.train_performance_catboost"] = shim


def _pin_to_one_thread(model: Any, name: str) -> Any:
    """Set ``n_jobs=1`` so a pickled ensemble returns the same numbers on every call.

    ``random_forest.pkl`` has ``n_jobs=-1`` from training, so sklearn sums its tree outputs from
    a joblib thread pool. Float addition is not associative, so the summation order changes the
    last bits of a prediction from call to call. With one thread the result is bit-stable, and
    the forest is faster, since dispatching single-row tree calls to a thread pool costs more
    than running them.
    """
    n_jobs = getattr(model, "n_jobs", None)
    if n_jobs is not None and n_jobs != 1:
        model.n_jobs = 1
        logger.info("%s: pinned n_jobs %s -> 1 for reproducible predictions", name, n_jobs)
    return model


class SklearnPortfolioPredictor(BasePredictor):
    """Featv3 sklearn-style throughput predictor, configured by model name and metadata fields."""

    #: A pickled sklearn/boosted-tree model costs more per prediction than a worker dispatch.
    EXPENSIVE = True

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
        """Lazy-load the model and preprocessing artifacts from the featv3 pickle."""
        if self._loaded:
            return

        if not self._model_path.exists():
            raise model_not_shipped_error(self._name, self._model_path)

        try:
            if self._native_categorical:
                _alias_legacy_catboost_module()

            with open(self._model_path, "rb") as f:
                artifacts = pickle.load(f)

            self._model = _pin_to_one_thread(artifacts["model"], self._name)
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
        """Predict throughput, or None for a model or GPU missing from Kavier's library (the
        pipeline skips that candidate).

        A model file that cannot be loaded, or holds no model, gives a Prediction with no numbers
        and the reason in ``metadata['error_detail']``. A missing model file raises
        ModelNotShippedError, since no candidate could be scored.
        """
        try:
            self._load()
        except ModelNotShippedError:
            raise
        except Exception as e:
            logger.warning("%s unavailable, skipping prediction: %s", self._name, e)
            return model_error_prediction(
                workload, model_name=self._name, detail=f"model artifact could not be loaded: {self._model_path} ({e})"
            )

        if self._model is None or self._cat_features is None or self._num_features is None:
            logger.warning("%s predictor artifacts are incomplete", self._name)
            return model_error_prediction(
                workload, model_name=self._name, detail=f"model artifact is incomplete: {self._model_path}"
            )

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
    """Per-model config: metadata fields and categorical mode. Inference is shared in
    ``SklearnPortfolioPredictor``."""

    metadata: tuple[MetaField, ...]
    native_categorical: bool = False

    def build(self, name: str) -> SklearnPortfolioPredictor:
        return SklearnPortfolioPredictor(name, self.metadata, native_categorical=self.native_categorical)
