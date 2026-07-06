#!/usr/bin/env python3
"""Train all 10 models sequentially and print a summary table."""

from .train_performance_bayesian_ridge import train as train_bayesian
from .train_performance_catboost import train as train_catboost
from .train_performance_deep_learning import train_deep_learning_model as train_dl
from .train_performance_gaussian_process import train as train_gp
from .train_performance_knn import train as train_knn
from .train_performance_lightgbm import train as train_lightgbm
from .train_performance_random_forest import train as train_rf
from .train_performance_svr import train as train_svr
from .train_performance_tabpfn import train_tabpfn
from .train_performance_xgboost import train as train_xgboost


def train_all():
    """Train all 10 models and print comparison table."""
    models = [
        ("RandomForest", train_rf),
        ("SVR", train_svr),
        ("KNN", train_knn),
        ("CatBoost", train_catboost),
        ("XGBoost", train_xgboost),
        ("LightGBM", train_lightgbm),
        ("GaussianProcess", train_gp),
        ("BayesianRidge", train_bayesian),
        ("DeepLearning", train_dl),
        ("TabPFN", train_tabpfn),
    ]

    print("=" * 80)
    print("Training All 10 Models")
    print("=" * 80)

    results = []
    total = len(models)
    for i, (name, train_fn) in enumerate(models, 1):
        print(f"\n[{i}/{total}] Training {name}...")
        try:
            train_fn()
            results.append((name, "✅ Success"))
        except Exception as e:
            results.append((name, f"❌ Failed: {str(e)[:50]}"))

    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    for name, status in results:
        print(f"{name:20s} {status}")
    print("=" * 80)


if __name__ == "__main__":
    train_all()
