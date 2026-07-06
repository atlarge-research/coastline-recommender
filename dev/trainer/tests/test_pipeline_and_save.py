"""End-to-end load/preprocess, encoders/scaler, and conditional artifact save.

``load_and_preprocess_data`` reads ``common.DATA_PATH`` — here it is monkey-
patched to a small synthetic CSV (``patched_data_path`` fixture) so the real,
large curated file and any trained pickles are never touched.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from .. import common as C

# Raw Kavier spec libraries — an INDEPENDENT source from common's own spec lookup
# (llm_spec_features / gpu_spec_features), used to cross-check attached spec values.
try:
    from kavier.sdk.library import GPU_SPEC_LIBRARY, LLM_SPEC_LIBRARY
except ImportError:  # pragma: no cover - specs absent
    LLM_SPEC_LIBRARY, GPU_SPEC_LIBRARY = {}, {}

# Keys known to exist in Kavier's libraries (mirrors conftest's KNOWN_LLM/KNOWN_GPU).
_KNOWN_LLM = "mistral-7b-v0.1"
_KNOWN_GPU = "NVIDIA-A100-SXM4-80GB"

# ---------------------------------------------------------------------------
# load_and_preprocess_data
# ---------------------------------------------------------------------------


def test_load_and_preprocess_shapes_and_no_na(patched_data_path):
    X_cat, X_num, y, cat_features, num_features = C.load_and_preprocess_data()
    n = len(y)
    # The synthetic fixture has exactly 200 rows, all is_valid==1 with strictly
    # positive targets, so the validity mask drops none: n == 200.
    assert n == 200
    assert len(X_cat) == len(X_num) == n
    # Targets: two columns in canonical order.
    assert list(y.columns) == C.get_target_column_names()
    # Feature lists report only columns actually present.
    assert list(X_cat.columns) == cat_features
    assert list(X_num.columns) == num_features
    # NA handling: categoricals filled with 'unknown', numerics median-filled.
    assert not X_cat.isna().any().any()
    assert not X_num.isna().any().any()


def test_load_and_preprocess_filters_invalid_and_nonpositive(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "model_name": ["mistral-7b-v0.1"] * 5,
            "gpu_model": ["L40S"] * 5,
            "method": ["full"] * 5,
            "number_nodes": [1] * 5,
            "number_gpus": [1] * 5,
            "tokens_per_sample": [1024] * 5,
            "batch_size": [4] * 5,
            "is_valid": [1.0, 0.0, 1.0, 1.0, 1.0],  # row 1 invalid
            "dataset_tokens_per_second": [100.0, 100.0, 0.0, np.nan, 200.0],  # rows 2,3 dropped
            "train_runtime": [10.0, 10.0, 10.0, 10.0, -5.0],  # row 4 dropped
        }
    )
    csv = tmp_path / "opts.csv"
    df.to_csv(csv, index=False)
    monkeypatch.setattr(C, "DATA_PATH", csv)
    _, _, y, _, _ = C.load_and_preprocess_data()
    # Only row 0 survives all of: is_valid==1, tput>0 & notna, runtime>0 & notna.
    assert len(y) == 1
    assert y[C.get_primary_target_name()].iloc[0] == 100.0


def test_load_and_preprocess_attaches_kavier_specs_from_library(tmp_path, monkeypatch):
    # Every row uses ONE known model + GPU, so median-fill can never mask an
    # attached spec; cross-check the numeric spec columns against Kavier's raw
    # library (independent of common's own llm_spec_features/gpu_spec_features).
    df = pd.DataFrame(
        {
            "model_name": [_KNOWN_LLM] * 4,
            "gpu_model": [_KNOWN_GPU] * 4,
            "method": ["full"] * 4,
            "number_nodes": [1] * 4,
            "number_gpus": [1] * 4,
            "tokens_per_sample": [1024] * 4,
            "batch_size": [4] * 4,
            "is_valid": [1.0] * 4,
            "dataset_tokens_per_second": [100.0, 200.0, 300.0, 400.0],
            "train_runtime": [10.0] * 4,
        }
    )
    csv = tmp_path / "opts.csv"
    df.to_csv(csv, index=False)
    monkeypatch.setattr(C, "DATA_PATH", csv)
    _, X_num, _, _, num_features = C.load_and_preprocess_data()
    # Contract: every declared spec column is present in the numeric feature list.
    for col in C.LLM_SPEC_NUMERICAL + C.GPU_SPEC_NUMERICAL:
        assert col in num_features
    # Value cross-check: llm_n_layers == library n_layers (mistral-7b-v0.1 -> 32).
    exp_layers = float(LLM_SPEC_LIBRARY[_KNOWN_LLM].n_layers)
    assert (X_num["llm_n_layers"] == exp_layers).all()
    # gpu_cores == library cores (A100-SXM4-80GB -> 6912).
    exp_cores = float(GPU_SPEC_LIBRARY[_KNOWN_GPU].cores)
    assert (X_num["gpu_cores"] == exp_cores).all()


def test_load_and_preprocess_missing_target_raises(tmp_path, monkeypatch):
    df = pd.DataFrame({"model_name": ["m"], "dataset_tokens_per_second": [1.0]})
    csv = tmp_path / "bad.csv"
    df.to_csv(csv, index=False)
    monkeypatch.setattr(C, "DATA_PATH", csv)
    with pytest.raises(ValueError, match="Missing required target column"):
        C.load_and_preprocess_data()


def test_pipeline_split_runs_end_to_end(patched_data_path):
    """Load -> transform -> split, exactly as the trainers do, and confirm the
    dual-target log frame survives with two columns through the split."""
    X_cat, X_num, y, _, _ = C.load_and_preprocess_data()
    y_log = C.transform_targets(y)
    (Xc_tr, Xn_tr, y_tr, yl_tr), (Xc_va, Xn_va, y_va, yl_va), (Xc_te, Xn_te, y_te, yl_te) = C.split_data(
        X_cat, X_num, y, y_log
    )
    # Hand-derived sizes for n=200: test = ceil(200*0.15)=30, temp=170;
    # val = ceil(170*0.176)=ceil(29.92)=30, train=170-30=140.
    assert (len(Xc_tr), len(Xc_va), len(Xc_te)) == (140, 30, 30)
    # Split is a partition: no row lost or duplicated.
    assert len(Xc_tr) + len(Xc_va) + len(Xc_te) == len(y)
    assert yl_tr.shape[1] == 2 and y_tr.shape[1] == 2
    # Round-trip on a split recovers the original-space targets.
    np.testing.assert_allclose(C.inverse_transform_targets(yl_te.to_numpy()), y_te.to_numpy(dtype=float), rtol=1e-9)


# ---------------------------------------------------------------------------
# encode_categorical_features
# ---------------------------------------------------------------------------


def _cat_frames():
    train = pd.DataFrame({"c": ["a", "b", "a", "c"]})
    val = pd.DataFrame({"c": ["a", "b"]})
    test = pd.DataFrame({"c": ["a", "zzz_unseen"]})  # unseen -> 'unknown'
    return train, val, test


def test_encode_categorical_dataframe_path():
    train, val, test = _cat_frames()
    Xtr, Xva, Xte, encoders, vocab = C.encode_categorical_features(train, val, test)
    assert isinstance(Xtr, pd.DataFrame)
    # 'unknown' is always added to the vocabulary -> 3 train classes + unknown.
    assert vocab["c"] == 4
    assert "unknown" in set(str(x) for x in encoders["c"].classes_)
    # Unseen test category collapses to the 'unknown' id.
    unknown_id = int(encoders["c"].transform(["unknown"])[0])
    assert int(Xte["c"].iloc[1]) == unknown_id


def test_encode_categorical_numpy_and_dataframe_paths_agree():
    train, val, test = _cat_frames()
    Xtr_df, _, Xte_df, enc, _ = C.encode_categorical_features(train, val, test)
    Xtr_np, Xva_np, Xte_np, _, _ = C.encode_categorical_features(train, val, test, return_numpy=True)
    assert isinstance(Xtr_np, np.ndarray) and Xtr_np.dtype == int
    # Shapes are (n_rows, 1) from the single-column frames: 4 train, 2 val, 2 test.
    assert (Xtr_np.shape, Xva_np.shape, Xte_np.shape) == ((4, 1), (2, 1), (2, 1))
    # The numpy path must encode identically to the dataframe path (same ids).
    np.testing.assert_array_equal(Xtr_np[:, 0], Xtr_df["c"].to_numpy())
    np.testing.assert_array_equal(Xte_np[:, 0], Xte_df["c"].to_numpy())
    # Unseen test label ('zzz_unseen', row 1) collapses to the 'unknown' id here too.
    unknown_id = int(enc["c"].transform(["unknown"])[0])
    assert Xte_np[1, 0] == unknown_id


def test_encode_categorical_train_ids_are_consistent():
    train = pd.DataFrame({"c": ["x", "y", "x"]})
    Xtr, _, _, encoders, _ = C.encode_categorical_features(train, train, train)
    # Same label -> same encoded id within the column.
    assert Xtr["c"].iloc[0] == Xtr["c"].iloc[2]
    assert Xtr["c"].iloc[0] != Xtr["c"].iloc[1]


# ---------------------------------------------------------------------------
# scale_numerical_features
# ---------------------------------------------------------------------------


def test_scale_numerical_fit_on_train_only():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"n": rng.normal(size=200)})
    val = pd.DataFrame({"n": rng.normal(size=40)})
    test = pd.DataFrame({"n": rng.normal(size=40)})
    Xtr, Xva, Xte, scaler = C.scale_numerical_features(train, val, test)
    # Columns preserved; shape unchanged.
    assert list(Xtr.columns) == ["n"]
    assert Xtr.shape == train.shape
    # Analytic reference: QuantileTransformer(output_distribution='normal') maps the
    # FITTED median to the 0.5 quantile -> normal ppf(0.5) = 0. So a scaler fit on
    # train yields a transformed-train median of ~0 (independent of the impl).
    assert float(np.median(Xtr.to_numpy())) == pytest.approx(0.0, abs=1e-9)
    # Fit-on-train-only cross-check: an INDEPENDENT scaler fit on train alone must
    # reproduce the returned test transform. If val/test had leaked into the fit,
    # this would diverge.
    from sklearn.preprocessing import QuantileTransformer

    ref = QuantileTransformer(output_distribution="normal", random_state=C.SEED).fit(train)
    np.testing.assert_allclose(ref.transform(test), Xte.to_numpy(), rtol=1e-9)


# ---------------------------------------------------------------------------
# Conditional artifact save / skip
# ---------------------------------------------------------------------------


def _artifacts(mdape: float) -> dict:
    return {
        "model": "STUB",  # never a real pickled estimator -> no segfault risk
        "test_metrics": {"original_space": {"mdape": mdape}},
    }


def test_save_when_no_prior_file(tmp_path):
    path = tmp_path / "perf.pkl"
    saved, msg = C.save_pickled_artifact_if_better(path, _artifacts(12.0), 12.0)
    assert saved is True
    assert path.is_file()
    assert "no comparable prior" in msg.lower()


def test_skip_when_new_is_worse_or_equal(tmp_path):
    path = tmp_path / "perf.pkl"
    C.save_pickled_artifact_if_better(path, _artifacts(10.0), 10.0)
    # Strictly-lower rule: equal MdAPE is NOT an improvement -> skip.
    saved_eq, _ = C.save_pickled_artifact_if_better(path, _artifacts(10.0), 10.0)
    assert saved_eq is False
    # Worse MdAPE -> skip, and the on-disk file is unchanged.
    saved_worse, msg = C.save_pickled_artifact_if_better(path, _artifacts(99.0), 99.0)
    assert saved_worse is False
    assert "kept existing" in msg.lower()
    with open(path, "rb") as f:
        on_disk = pickle.load(f)
    assert on_disk["test_metrics"]["original_space"]["mdape"] == 10.0


def test_replace_when_new_is_strictly_better(tmp_path):
    path = tmp_path / "perf.pkl"
    C.save_pickled_artifact_if_better(path, _artifacts(20.0), 20.0)
    saved, msg = C.save_pickled_artifact_if_better(path, _artifacts(8.0), 8.0)
    assert saved is True
    assert "→" in msg or "->" in msg
    with open(path, "rb") as f:
        on_disk = pickle.load(f)
    assert on_disk["test_metrics"]["original_space"]["mdape"] == 8.0


def test_force_overwrites_even_if_worse(tmp_path):
    path = tmp_path / "perf.pkl"
    C.save_pickled_artifact_if_better(path, _artifacts(5.0), 5.0)
    saved, _ = C.save_pickled_artifact_if_better(path, _artifacts(50.0), 50.0, force=True)
    assert saved is True
    with open(path, "rb") as f:
        assert pickle.load(f)["test_metrics"]["original_space"]["mdape"] == 50.0


def test_save_creates_missing_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "deep" / "perf.pkl"
    saved, _ = C.save_pickled_artifact_if_better(path, _artifacts(7.0), 7.0)
    assert saved is True
    assert path.is_file()


# ---------------------------------------------------------------------------
# get_stored_test_throughput_mdape: both supported blob layouts + robustness
# ---------------------------------------------------------------------------


def test_stored_mdape_missing_file_returns_none(tmp_path):
    assert C.get_stored_test_throughput_mdape(tmp_path / "nope.pkl") is None


def test_stored_mdape_reads_test_metrics_layout(tmp_path):
    path = tmp_path / "a.pkl"
    with open(path, "wb") as f:
        pickle.dump({"test_metrics": {"original_space": {"mdape": 13.5}}}, f)
    assert C.get_stored_test_throughput_mdape(path) == pytest.approx(13.5)


def test_stored_mdape_reads_by_target_layout(tmp_path):
    path = tmp_path / "b.pkl"
    blob = {"test_metrics_by_target": {"throughput": {"original_space": {"mdape": 9.25}}}}
    with open(path, "wb") as f:
        pickle.dump(blob, f)
    assert C.get_stored_test_throughput_mdape(path) == pytest.approx(9.25)


def test_stored_mdape_non_dict_or_corrupt_returns_none(tmp_path):
    # Non-dict pickle.
    p1 = tmp_path / "list.pkl"
    with open(p1, "wb") as f:
        pickle.dump([1, 2, 3], f)
    assert C.get_stored_test_throughput_mdape(p1) is None
    # Corrupt / unreadable file.
    p2 = tmp_path / "corrupt.pkl"
    p2.write_bytes(b"not a pickle at all")
    assert C.get_stored_test_throughput_mdape(p2) is None
    # Dict without any recognised metrics key.
    p3 = tmp_path / "empty.pkl"
    with open(p3, "wb") as f:
        pickle.dump({"model": "x"}, f)
    assert C.get_stored_test_throughput_mdape(p3) is None


# ---------------------------------------------------------------------------
# training_force_save env parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "val,expected",
    [
        # Accepted spellings {1, true, yes}, case-insensitive after strip.
        ("1", True),
        ("true", True),
        ("yes", True),
        ("TRUE", True),
        (" Yes ", True),
        # Anything else -> False (would catch a bug that accepts only "true",
        # or that treats any non-empty / any digit as truthy).
        ("0", False),
        ("no", False),
        ("false", False),
        ("2", False),
        ("", False),
    ],
)
def test_training_force_save_accepts_only_1_true_yes(monkeypatch, val, expected):
    monkeypatch.setenv("TRAIN_FORCE_SAVE", val)
    assert C.training_force_save() is expected
