"""Shared fixtures / helpers for the trainer unit tests.

These tests exercise the *pure* logic in the trainer's ``common`` module (feature
engineering, splitting, target transforms, conditional artifact save) using
small synthetic frames. They deliberately do NOT load the large real model
pickles — unpickling the XGBoost artifacts can segfault on host — nor do they
train any model, so the suite stays fast and deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from .. import common as C

# A couple of keys that are known to exist in Kavier's spec libraries
# (verified against kavier/src/library/{llm,gpu}.py). Used to assert the
# feature-parity augmentation actually pulls real specs rather than NaN.
KNOWN_LLM = "mistral-7b-v0.1"
KNOWN_GPU = "NVIDIA-A100-SXM4-80GB"


def make_synthetic_options(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic stand-in for the curated training CSV.

    Columns mirror the curated CSV schema that ``load_and_preprocess_data``
    reads: the two targets, the base categorical/numeric inputs, plus a few
    raw columns the engineering step consumes (``model_name``, ``torch_dtype``,
    ``enable_roce``, ``is_valid``).
    """
    rng = np.random.default_rng(seed)
    models = [KNOWN_LLM, "granite-3.1-3b-a800m-instruct", "mixtral-8x7b-instruct-v0.1"]
    gpus = [KNOWN_GPU, "NVIDIA-A100-80GB-PCIe", "L40S"]
    methods = ["full", "lora"]
    dtypes = ["bfloat16", "float16", None]

    df = pd.DataFrame(
        {
            "model_name": rng.choice(models, n),
            "gpu_model": rng.choice(gpus, n),
            "method": rng.choice(methods, n),
            "torch_dtype": rng.choice(np.array(dtypes, dtype=object), n),
            "enable_roce": rng.choice([0, 1], n),
            "number_nodes": rng.integers(1, 4, n),
            "number_gpus": rng.integers(1, 9, n),
            "tokens_per_sample": rng.choice([512, 1024, 2048, 4096], n),
            "batch_size": rng.choice([1, 2, 4, 8], n),
            "is_valid": 1.0,
            # Targets, strictly positive so the validity mask keeps every row.
            "dataset_tokens_per_second": rng.uniform(50.0, 5000.0, n),
            "train_runtime": rng.uniform(10.0, 10000.0, n),
        }
    )
    return df


@pytest.fixture()
def synthetic_options() -> pd.DataFrame:
    return make_synthetic_options()


@pytest.fixture()
def patched_data_path(tmp_path, monkeypatch, synthetic_options):
    """Point ``common.DATA_PATH`` at a synthetic CSV so the real (large) curated
    file is never required for the load/preprocess tests."""
    csv = tmp_path / "synthetic_options.csv"
    synthetic_options.to_csv(csv, index=False)
    monkeypatch.setattr(C, "DATA_PATH", csv)
    return csv


class StubWorkload:
    """Duck-typed stand-in for ``coastline_common`` WorkloadSpec.

    ``workload_to_ml_feature_row`` only uses attribute access, so we avoid
    importing/constructing the real Pydantic model to keep the unit isolated.
    """

    def __init__(
        self,
        llm_model=KNOWN_LLM,
        gpu_model=KNOWN_GPU,
        fine_tuning_method="full",
        number_of_nodes=2,
        gpus_per_node=4,
        tokens_per_sample=2048,
        batch_size=4,
        torch_dtype="bfloat16",
        enable_roce=True,
        set_optional=True,
    ):
        self.llm_model = llm_model
        self.gpu_model = gpu_model
        self.fine_tuning_method = fine_tuning_method
        self.number_of_nodes = number_of_nodes
        self.gpus_per_node = gpus_per_node
        self.tokens_per_sample = tokens_per_sample
        self.batch_size = batch_size
        # Optional attributes are accessed via getattr(..., None); only set
        # them when requested so we can test the "missing attribute" branch.
        if set_optional:
            self.torch_dtype = torch_dtype
            self.enable_roce = enable_roce
