"""Data-driven performance predictors (ML models).

Inference needs two classes from this package: ``EmbeddingNN`` (in ``_nn``) and
``_DualOutputCatBoost`` (in ``_catboost_model``). The training scripts in ``dev/trainer``
import the same classes, so training and inference share one definition.
"""

# Lazy attribute access (PEP 562): a predictor module is imported when its name is first used.
# Loading torch, catboost, xgboost and lightgbm into one process puts their OpenMP runtimes side
# by side, which segfaults on macOS, so the playground subprocess for "xgboost" loads only xgboost.
_PREDICTOR_MODULES = {
    "SklearnPortfolioPredictor": "sklearn_portfolio",
    "GaussianProcessPredictor": "gaussian_process_predictor",
    "BayesianRidgePredictor": "bayesian_ridge_predictor",
    "TabPFNPredictor": "tabpfn_predictor",
    "DeepLearningPredictor": "deep_learning_predictor",
}

__all__ = list(_PREDICTOR_MODULES)


def __getattr__(name: str):
    module = _PREDICTOR_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    mod = importlib.import_module(f".{module}", __name__)
    return getattr(mod, name)


def __dir__():
    return sorted(__all__)
