#!/usr/bin/env python3
"""RandomForest predictor (model #1 of 10), tuned with GridSearchCV. Target MdAPE 8-15%."""

from .generic_trainer import run_training
from .model_specs import PERFORMANCE_MODELS


def train():
    """Train the RandomForest model."""
    run_training(PERFORMANCE_MODELS["random_forest"])


if __name__ == "__main__":
    train()
