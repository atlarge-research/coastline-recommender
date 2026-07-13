#!/usr/bin/env python3
"""KNN predictor: QuantileTransformer + Minkowski distance. Target MdAPE <20%."""

from .generic_trainer import run_training
from .model_specs import PERFORMANCE_MODELS


def train():
    """Train the KNN model."""
    run_training(PERFORMANCE_MODELS["knn"])


if __name__ == "__main__":
    train()
