#!/usr/bin/env python3
"""Gaussian Process predictor: non-parametric Bayesian with per-prediction uncertainty. Target MdAPE 12-20%."""

from .generic_trainer import run_training
from .model_specs import PERFORMANCE_MODELS


def train():
    """Train the Gaussian Process model."""
    run_training(PERFORMANCE_MODELS["gaussian_process"])


if __name__ == "__main__":
    train()
