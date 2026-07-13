#!/usr/bin/env python3
"""Train every model in the registry sequentially and print a summary table."""

from .main import _MODEL_TRAINERS, _run_single_model


def train_all():
    """Train all registered models and print a comparison table."""
    total = len(_MODEL_TRAINERS)
    print("=" * 80)
    print(f"Training All {total} Models")
    print("=" * 80)

    results = []
    for i, name in enumerate(_MODEL_TRAINERS, 1):
        print(f"\n[{i}/{total}] Training {name}...")
        try:
            _run_single_model(name)
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
