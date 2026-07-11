"""ML model trainer entry point.

Usage: python -m trainer.main [--all | --model xgboost | --evaluate]
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
from pathlib import Path

from coastline.sdk.logging import setup_logging

logger = logging.getLogger(__name__)

# --model names -> (module basename, callable attribute)
_MODEL_TRAINERS: dict[str, tuple[str, str]] = {
    "xgboost": ("train_performance_xgboost", "train"),
    "lightgbm": ("train_performance_lightgbm", "train"),
    "catboost": ("train_performance_catboost", "train"),
    "random_forest": ("train_performance_random_forest", "train"),
    "svr": ("train_performance_svr", "train"),
    "knn": ("train_performance_knn", "train"),
    "gaussian_process": ("train_performance_gaussian_process", "train"),
    "bayesian_ridge": ("train_performance_bayesian_ridge", "train"),
    "tabpfn": ("train_performance_tabpfn", "train_tabpfn"),
    "deep_learning": ("train_performance_deep_learning", "train_deep_learning_model"),
}


def _run_single_model(model: str) -> None:
    spec = _MODEL_TRAINERS.get(model)
    if spec is None:
        valid = ", ".join(sorted(_MODEL_TRAINERS))
        raise SystemExit(f"Unknown model {model!r}. Valid: {valid}")
    module_name, attr = spec
    mod = importlib.import_module(f".{module_name}", package=__package__)
    train_fn = getattr(mod, attr)
    train_fn()


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="ML Model Trainer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Train all models")
    group.add_argument("--model", type=str, help="Train specific model (e.g. xgboost)")
    group.add_argument("--evaluate", action="store_true", help="Evaluate all trained models")
    parser.add_argument("--config", default=os.environ.get("CONFIG_FILE"))
    args = parser.parse_args()

    data_dir = Path(os.environ.get("DATA_DIR", "./trace-archive"))
    os.environ["DATA_DIR"] = str(data_dir)

    if args.all:
        from .train_all import train_all

        train_all()
    elif args.evaluate:
        from .evaluate_all import evaluate_all

        evaluate_all()
    elif args.model:
        _run_single_model(args.model)

    logger.info("Training complete")


if __name__ == "__main__":
    main()
