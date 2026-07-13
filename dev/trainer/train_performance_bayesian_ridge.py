#!/usr/bin/env python3
"""Bayesian Ridge predictor: polynomial features + per-prediction uncertainty. Target MdAPE 15-25%."""

from .generic_trainer import run_training
from .model_specs import PERFORMANCE_MODELS


def train():
    """Train the Bayesian Ridge model."""
    run_training(PERFORMANCE_MODELS["bayesian_ridge"])


if __name__ == "__main__":
    train()
