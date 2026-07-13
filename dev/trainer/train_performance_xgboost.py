#!/usr/bin/env python3
"""XGBoost predictor (model #6 of 10), tuned with GridSearchCV. Target MdAPE 8-14%."""

from .generic_trainer import run_training
from .model_specs import PERFORMANCE_MODELS


def train():
    """Train the XGBoost model."""
    run_training(PERFORMANCE_MODELS["xgboost"])


if __name__ == "__main__":
    train()
