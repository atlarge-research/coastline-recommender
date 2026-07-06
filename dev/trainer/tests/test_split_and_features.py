"""split_data ratios/determinism + feature assembly (incl. Kavier spec parity).

These cover the data-plumbing logic that every one of the 10 trainers shares:
the 70/15/15 split (seed 42), feature engineering / spec augmentation, the
end-to-end ``load_and_preprocess_data`` contract, and the categorical /
numerical encoders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from .. import common as C
from .conftest import KNOWN_GPU, KNOWN_LLM, StubWorkload, make_synthetic_options

# ---------------------------------------------------------------------------
# split_data: ratios, alignment, determinism
# ---------------------------------------------------------------------------


def test_split_ratios_70_15_15():
    n = 1000
    X = pd.DataFrame({"f": np.arange(n)})
    y = pd.DataFrame({"t": np.arange(n) * 2.0})
    (Xtr, ytr), (Xva, yva), (Xte, yte) = C.split_data(X, y)
    # Hand-derived from sklearn's split rule (n_test = ceil(frac * n)):
    #   test  = ceil(0.15  * 1000)        = 150   -> remainder 850
    #   val   = ceil(0.176 *  850)=ceil(149.6)=150 -> train 700
    #   train = 850 - 150                 = 700
    assert (len(Xtr), len(Xva), len(Xte)) == (700, 150, 150)
    # Partition is exhaustive: parts tile the whole.
    assert len(Xtr) + len(Xva) + len(Xte) == n


def test_split_keeps_rows_aligned_across_arrays():
    """Each input array must be split with the SAME row assignment; here y = 2*X
    so the relationship must hold within every split."""
    n = 300
    X = pd.DataFrame({"f": np.arange(n)})
    y = pd.DataFrame({"t": np.arange(n) * 2.0})
    (Xtr, ytr), (Xva, yva), (Xte, yte) = C.split_data(X, y)
    for Xs, ys in [(Xtr, ytr), (Xva, yva), (Xte, yte)]:
        np.testing.assert_array_equal(Xs["f"].to_numpy() * 2.0, ys["t"].to_numpy())


def test_split_partitions_without_overlap_and_covers_all():
    n = 250
    X = pd.DataFrame({"f": np.arange(n)})
    (Xtr,), (Xva,), (Xte,) = C.split_data(X)
    tr, va, te = set(Xtr["f"]), set(Xva["f"]), set(Xte["f"])
    assert tr & va == set() and tr & te == set() and va & te == set()
    assert tr | va | te == set(range(n))


def test_split_is_deterministic_with_seed_42():
    n = 400
    X = pd.DataFrame({"f": np.arange(n)})
    r1 = C.split_data(X)
    r2 = C.split_data(X)
    for grp1, grp2 in zip(r1, r2):
        np.testing.assert_array_equal(grp1[0]["f"].to_numpy(), grp2[0]["f"].to_numpy())
    # Default seed is the project-wide SEED=42.
    assert C.SEED == 42


def test_split_different_seed_changes_partition():
    n = 400
    X = pd.DataFrame({"f": np.arange(n)})
    (a,), _, _ = C.split_data(X)
    (b,), _, _ = C.split_data(X, seed=7)
    assert not np.array_equal(a["f"].to_numpy(), b["f"].to_numpy())


def test_split_handles_four_arrays_like_the_trainers():
    """Trainers call split_data(X_cat, X_num, y, y_log); verify the 4-array
    unpacking shape used everywhere in train_performance_*.py."""
    n = 120
    arrays = [pd.DataFrame({c: np.arange(n) + j for c in ["a"]}) for j in range(4)]
    train, val, test = C.split_data(*arrays)
    # 4 arrays in -> 4 arrays out in each of the 3 groups.
    assert len(train) == len(val) == len(test) == 4
    # Hand-derived sizes (ceil rule): test=ceil(0.15*120)=18 -> rem 102;
    #   val=ceil(0.176*102)=ceil(17.95)=18 -> train=84.
    assert (len(train[0]), len(val[0]), len(test[0])) == (84, 18, 18)
    # Every one of the 4 arrays must be split identically (row-count parity).
    for g in (train, val, test):
        assert len({len(a) for a in g}) == 1


# ---------------------------------------------------------------------------
# Model-family / size-bucket extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,family",
    [
        ("meta-llama/Llama-3.1-8B", "llama"),  # vendor-prefixed
        ("granite-3.1-3b-a800m-instruct", "granite"),
        ("mixtral-8x7b-instruct-v0.1", "mixtral"),  # MoE name vs mistral
    ],
)
def test_extract_model_family(name, family):
    assert C.extract_model_family(name) == family


@pytest.mark.parametrize(
    "name,bucket",
    [
        ("llama3.1-70b", "llama70b"),
        ("mixtral-8x7b-instruct-v0.1", "mixtral8x7b"),  # 8x7b layout
        ("granite-3.1-3b-a800m-instruct", "granite3b"),
    ],
)
def test_extract_model_size_bucket(name, bucket):
    assert C.extract_model_size_bucket(name) == bucket


# ---------------------------------------------------------------------------
# Spec-parity feature lookups (the Kavier-augmentation contract)
# ---------------------------------------------------------------------------


def test_llm_spec_features_known_model_matches_published_mistral7b_arch():
    # KNOWN_LLM is Mistral-7B-v0.1. Oracle = its published model card / config
    # (independent of this code): 32 transformer layers, hidden size 4096,
    # 32 attention heads -> head dim 4096/32 = 128; a dense (non-MoE) model so
    # exactly 1 expert.
    feats = C.llm_spec_features(KNOWN_LLM)
    assert set(feats) == set(C.LLM_SPEC_NUMERICAL)
    assert all(np.isfinite(v) for v in feats.values())
    assert feats["llm_n_layers"] == pytest.approx(32.0)
    assert feats["llm_d_model"] == pytest.approx(4096.0)
    assert feats["llm_n_heads"] == pytest.approx(32.0)
    # d_head is the derived per-head width: 4096 / 32 = 128.
    assert feats["llm_d_head"] == pytest.approx(128.0)
    assert feats["llm_num_experts"] == 1.0


def test_llm_spec_features_unknown_model_is_all_nan():
    feats = C.llm_spec_features("___not_a_real_model___")
    assert set(feats) == set(C.LLM_SPEC_NUMERICAL)
    assert all(np.isnan(v) for v in feats.values())


def test_gpu_spec_features_known_gpu_matches_published_a100_datasheet():
    # KNOWN_GPU is the NVIDIA A100-SXM4-80GB. Oracle = the published NVIDIA
    # datasheet (independent of this code's spec table):
    #   FP16 tensor-core peak = 312 TFLOPS; HBM2e memory bandwidth = 2039 GB/s
    #   (2.039 TB/s = 2.039e12 B/s, so bps/1e9 must land on 2039.0); 80 GB; 400 W TDP.
    feats = C.gpu_spec_features(KNOWN_GPU)
    assert set(feats) == set(C.GPU_SPEC_NUMERICAL)
    assert all(np.isfinite(v) for v in feats.values())
    assert feats["gpu_fp16_tflops"] == pytest.approx(312.0)
    assert feats["gpu_mem_bw_gbps"] == pytest.approx(2039.0)
    assert feats["gpu_mem_gb"] == pytest.approx(80.0)
    assert feats["gpu_tdp_w"] == pytest.approx(400.0)


def test_gpu_spec_features_unknown_gpu_is_all_nan():
    feats = C.gpu_spec_features("___not_a_real_gpu___")
    assert all(np.isnan(v) for v in feats.values())


def test_spec_knobs_mfu_and_calibration_are_not_features():
    """Feature parity must EXCLUDE the tuned calibration knobs."""
    leak = {"mfu_factor", "calibration_factor", "mfu_multiplier", "comm_scale"}
    assert leak.isdisjoint(set(C.SPEC_NUMERICAL))
    assert leak.isdisjoint(set(C.GPU_SPEC_NUMERICAL))


# ---------------------------------------------------------------------------
# engineer_features
# ---------------------------------------------------------------------------


def test_engineer_features_adds_expected_columns_and_specs():
    df = make_synthetic_options(n=30, seed=1)
    out = C.engineer_features(df)
    for col in ["model_type", "model_size_bucket", "torch_dtype", "enable_roce", "total_gpus"]:
        assert col in out.columns
    # Spec columns attached for both LLM and GPU.
    for col in C.LLM_SPEC_NUMERICAL + C.GPU_SPEC_NUMERICAL:
        assert col in out.columns
    # total_gpus == number_nodes * number_gpus.
    expected = df["number_nodes"].astype(float) * df["number_gpus"].astype(float)
    np.testing.assert_allclose(out["total_gpus"].to_numpy(), expected.to_numpy())


def test_engineer_features_does_not_mutate_input():
    df = make_synthetic_options(n=10, seed=2)
    before = df.copy(deep=True)
    _ = C.engineer_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_engineer_features_roce_and_dtype_categories():
    df = pd.DataFrame(
        {
            "model_name": ["mistral-7b-v0.1"] * 3,
            "gpu_model": ["L40S"] * 3,
            "number_nodes": [1, 1, 1],
            "number_gpus": [1, 2, 4],
            "torch_dtype": ["BFloat16", None, "nan"],
            "enable_roce": [1, 0, np.nan],
        }
    )
    out = C.engineer_features(df)
    assert list(out["enable_roce"]) == ["1", "0", "unknown"]
    # torch_dtype lower-cased; None / 'nan' -> 'unknown'.
    assert list(out["torch_dtype"]) == ["bfloat16", "unknown", "unknown"]


def test_engineer_features_total_gpus_clips_below_one():
    df = pd.DataFrame(
        {
            "model_name": ["mistral-7b-v0.1"],
            "gpu_model": ["L40S"],
            "number_nodes": [0],
            "number_gpus": [0],
        }
    )
    out = C.engineer_features(df)
    # Both clipped to >=1 -> product 1.0.
    assert out["total_gpus"].iloc[0] == 1.0


# ---------------------------------------------------------------------------
# get_feature_lists
# ---------------------------------------------------------------------------


def test_get_feature_lists_matches_featv3_schema():
    # The featv3 schema is a fixed contract the trained pickles were fit on;
    # spell it out independently of the module's concatenation so a reorder or
    # dropped/renamed column is caught (order matters: column-indexed encoders).
    cat, num = C.get_feature_lists()
    assert cat == [
        "method",
        "gpu_model",
        "model_type",
        "torch_dtype",
        "enable_roce",
        "model_size_bucket",
    ]
    assert num == [
        "number_nodes",
        "number_gpus",
        "tokens_per_sample",
        "batch_size",
        "total_gpus",
        "llm_n_layers",
        "llm_d_model",
        "llm_n_heads",
        "llm_d_head",
        "llm_m_params",
        "llm_active_params",
        "llm_num_experts",
        "llm_active_experts",
        "gpu_fp16_tflops",
        "gpu_mem_bw_gbps",
        "gpu_cores",
        "gpu_mem_gb",
        "gpu_clock_mhz",
        "gpu_tdp_w",
        "gpu_net_bw_gbps",
    ]
    # Categorical and numerical name-spaces must not overlap.
    assert set(cat).isdisjoint(set(num))


# ---------------------------------------------------------------------------
# workload_to_ml_feature_row / curated_series_to_ml_feature_row parity
# ---------------------------------------------------------------------------


def test_workload_feature_row_has_all_model_columns():
    feat = C.workload_to_ml_feature_row(StubWorkload())
    cat, num = C.get_feature_lists()
    for key in cat + num:
        assert key in feat, f"missing feature {key}"


def test_workload_feature_row_values():
    feat = C.workload_to_ml_feature_row(StubWorkload(number_of_nodes=2, gpus_per_node=4, enable_roce=True))
    assert feat["total_gpus"] == 8.0
    assert feat["number_nodes"] == 2 and feat["number_gpus"] == 4
    assert feat["enable_roce"] == "1"
    assert feat["model_type"] == "mistral"
    assert feat["model_size_bucket"] == "mistral7b"
    # Specs populated for a known model/GPU.
    assert np.isfinite(feat["llm_n_layers"])
    assert np.isfinite(feat["gpu_fp16_tflops"])


def test_workload_feature_row_missing_optionals_default_to_unknown():
    feat = C.workload_to_ml_feature_row(StubWorkload(set_optional=False))
    assert feat["torch_dtype"] == "unknown"
    assert feat["enable_roce"] == "unknown"


def test_curated_row_matches_workload_row_for_same_inputs():
    """The two builders feed the same model schema; for identical inputs the
    produced feature dicts must agree (guards against feature drift)."""
    row = pd.Series(
        {
            "model_name": KNOWN_LLM,
            "gpu_model": KNOWN_GPU,
            "method": "full",
            "torch_dtype": "bfloat16",
            "enable_roce": 1,
            "number_nodes": 2,
            "number_gpus": 4,
            "tokens_per_sample": 2048,
            "batch_size": 4,
        }
    )
    from_row = C.curated_series_to_ml_feature_row(row)
    from_wl = C.workload_to_ml_feature_row(
        StubWorkload(
            llm_model=KNOWN_LLM,
            gpu_model=KNOWN_GPU,
            fine_tuning_method="full",
            number_of_nodes=2,
            gpus_per_node=4,
            tokens_per_sample=2048,
            batch_size=4,
            torch_dtype="bfloat16",
            enable_roce=True,
        )
    )
    assert set(from_row) == set(from_wl)
    for k in from_row:
        a, b = from_row[k], from_wl[k]
        if isinstance(a, float) and np.isnan(a):
            assert isinstance(b, float) and np.isnan(b)
        else:
            assert a == b, f"mismatch on {k}: {a!r} != {b!r}"
