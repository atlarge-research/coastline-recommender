"""Data-driven performance predictors (ML models).

Inference needs only two shipped classes — ``EmbeddingNN`` (in ``_nn``) and
``_DualOutputCatBoost`` (in ``_catboost_model``); the training scripts that produce the
pickles live in the dev-only ``dev/trainer`` package and import those same classes back,
so there is one definition and no drift.
"""

# Lazy attribute access (PEP 562). Importing one predictor must NOT drag in every
# other ML runtime: torch (deep_learning) + catboost + xgboost + lightgbm loading
# into a single process makes their native OpenMP runtimes coexist, which segfaults
# on macOS. Each predictor module is imported only when its name is first accessed,
# so e.g. the playground subprocess for "xgboost" loads xgboost alone.
_PREDICTOR_MODULES = {
    "RandomForestPredictor": "random_forest_predictor",
    "SVRPredictor": "svr_predictor",
    "KNNPredictor": "knn_predictor",
    "CatBoostPredictor": "catboost_predictor",
    "XGBoostPredictor": "xgboost_predictor",
    "LightGBMPredictor": "lightgbm_predictor",
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
