"""Single source of truth for the available configuration options.

Loads the selectable models, GPUs, methods, sequence lengths and batch sizes
from the curated dataset (falling back to a hardcoded set if it is missing).
"""

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

from coastline.sdk.constants import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_TOKENS_PER_SAMPLE,
    Method,
)


def _default_options_path() -> Path:
    """Curated options CSV under the repo's trace-archive, overridable via DATA_DIR."""
    data_dir = os.environ.get("DATA_DIR")
    base = Path(data_dir) if data_dir else Path(__file__).resolve().parents[5] / "trace-archive"
    return base / "profiling-dataset" / "curated_trace.csv"


DEFAULT_OPTIONS_PATH = _default_options_path()


@lru_cache(maxsize=8)
def _load_options(csv_path: str) -> dict[str, list]:
    try:
        df = pd.read_csv(csv_path)
        return {
            "models": sorted(df["model_name"].dropna().unique().tolist()),
            "methods": sorted(df["method"].dropna().unique().tolist()),
            "gpus": sorted(df["gpu_model"].dropna().unique().tolist()),
            "tokens_per_sample": sorted(df["tokens_per_sample"].dropna().unique().astype(int).tolist()),
            "batch_sizes": sorted(df["batch_size"].dropna().unique().astype(int).tolist()),
        }
    except FileNotFoundError:
        return get_fallback_options()
    except Exception as e:
        print(f"Warning: Failed to load options from {csv_path}: {e}")
        return get_fallback_options()


def load_available_options(csv_path: Path | None = None) -> dict[str, list]:
    """Load options from the curated dataset; falls back to hardcoded defaults if missing."""
    return _load_options(str(csv_path or _default_options_path()))


# Expose the underlying per-path cache control on the public function.
load_available_options.cache_clear = _load_options.cache_clear  # type: ignore[attr-defined]


def get_fallback_options() -> dict[str, list]:
    """A minimal hardcoded option set, used when the curated CSV is unavailable."""
    return {
        "models": [
            "granite-3.1-3b-a800m-instruct",
            "granite-3.3-8b",
            "mistral-7b-v0.1",
            "mixtral-8x7b-instruct-v0.1",
        ],
        "methods": sorted(m.value for m in Method),
        "gpus": ["L40S", "NVIDIA-A100-80GB-PCIe", "NVIDIA-A100-SXM4-80GB"],
        "tokens_per_sample": list(DEFAULT_TOKENS_PER_SAMPLE),
        "batch_sizes": list(DEFAULT_BATCH_SIZES),
    }


def get_models() -> list[str]:
    """Available LLM models."""
    return load_available_options()["models"]


def get_methods() -> list[str]:
    """Available training methods."""
    return load_available_options()["methods"]


def get_gpus() -> list[str]:
    """Available GPU models."""
    return load_available_options()["gpus"]


def get_tokens_per_sample() -> list[int]:
    """Available sequence lengths."""
    return load_available_options()["tokens_per_sample"]


def get_batch_sizes() -> list[int]:
    """Available batch sizes."""
    return load_available_options()["batch_sizes"]


__all__ = [
    "load_available_options",
    "get_fallback_options",
    "get_models",
    "get_methods",
    "get_gpus",
    "get_tokens_per_sample",
    "get_batch_sizes",
]
