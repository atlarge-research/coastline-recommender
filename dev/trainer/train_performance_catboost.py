#!/usr/bin/env python3
"""CatBoost throughput/runtime predictor with native categorical support. Target MdAPE 6-12%."""

from .generic_trainer import run_training
from .model_specs import PERFORMANCE_MODELS


def train():
    """Train the CatBoost model."""
    run_training(PERFORMANCE_MODELS["catboost"])


if __name__ == "__main__":
    train()
