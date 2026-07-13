#!/usr/bin/env python3
"""SVR predictor (model #2 of 10): RBF kernel + StandardScaler (scale-sensitive). Target MdAPE 10-18%."""

from .generic_trainer import run_training
from .model_specs import PERFORMANCE_MODELS


def train():
    """Train the SVR model."""
    run_training(PERFORMANCE_MODELS["svr"])


if __name__ == "__main__":
    train()
