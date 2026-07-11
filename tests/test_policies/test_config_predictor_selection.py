"""Config-driven predictor selection (performance + energy), end-to-end.

The recommender lets ``config.yaml`` choose which engine produces throughput and
which produces energy via the ``predictors:`` section::

    predictors:
      performance: <intelligent | kavier | cache | catboost | xgboost | ...>
      energy:      <kavier_power>

``coastline/sdk/policies/__init__.py`` (``PolicyFactory``) and
``coastline/sdk/io/run_config.py`` (``load_strategy_config``) consume those keys.

Oracle for this module
----------------------
Every assertion here is checked against the **documented config→predictor map**
(an exact, independent specification), not against whatever the code happened to
return.  The map is:

    performance:  kavier/physics/physics_driven -> KavierPredictor
                  cache                          -> RetrievalPredictor
                  intelligent (default)          -> CacheThenPhysicsPredictor
                                                    wired cache=Retrieval, physics=Kavier
                  catboost/xgboost/lightgbm/...   -> <Name>Predictor  (one class *each*)
                  <unknown name>                 -> intelligent fallback
    energy:       kavier_power (default)         -> KavierPowerPredictor (WRAPS_THROUGHPUT_ENGINE)
                  <unknown name>                 -> ValueError

The per-name map is load-bearing because of a real regression documented in
CLAUDE.md: an earlier duplicate resolver *silently collapsed every named model to
CatBoost*.  The parametrized sweep below is the pinned-bug guard — if the resolver
regressed, ``xgboost`` would resolve to ``CatBoostPredictor`` and the test goes red.

Scope / segfault avoidance
--------------------------
We assert the *type / wiring* of the selected predictor only and never call
``.predict()`` on a data-driven (ML) predictor — they lazy-unpickle their model on
first predict and unpickling xgboost & friends in-process segfaults on the host.
*Constructing* them is safe (verified).
"""

from pathlib import Path

import pytest
import yaml

import coastline.sdk.predictors.performance.data_driven.ml_common as ml_common
from coastline.sdk.io.run_config import load_strategy_config
from coastline.sdk.policies import PolicyFactory
from coastline.sdk.predictors.energy import KavierPowerPredictor
from coastline.sdk.predictors.performance.composite import CacheThenPhysicsPredictor
from coastline.sdk.predictors.performance.physics import KavierPredictor
from coastline.sdk.predictors.performance.retrieval.cache_predictor import RetrievalPredictor

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_ml_artifact(monkeypatch):
    """Point trained-artifact lookups at a nonexistent path.

    Named ML predictors resolve their pickle path via
    ``ml_common.performance_trained_model_path`` in ``__init__``; steering it to a
    missing file keeps construction host-independent and prevents any accidental
    load of a committed artifact during these type-only assertions.
    """
    missing = Path("/nonexistent/config_predictor_selection/performance_catboost.pkl")
    monkeypatch.setattr(ml_common, "performance_trained_model_path", lambda stem: missing)


def _make_strategy(predictors: dict | None, strategy_name: str = "multi_objective"):
    """Build a strategy from a config dict whose only varying part is ``predictors``.

    ``predictors=None`` omits the ``predictors:`` section entirely (the strongest
    form of "keys absent").
    """
    config: dict = {
        "strategy": {"name": strategy_name, "preset": "balanced"},
        "grid": {"batch_sizes": [4, 8], "total_gpus": [1, 2], "top_k": 1},
    }
    if predictors is not None:
        config["predictors"] = predictors
    return PolicyFactory.create_strategy(config=config)


# ---------------------------------------------------------------------------
# named ML models: each name selects its OWN class (anti-"collapse to CatBoost")
# ---------------------------------------------------------------------------


class TestNamedMLModelSelection:
    # The documented name->class map. Independent oracle: these are the class
    # names the config contract promises, one distinct class per model. A
    # resolver that collapsed every name to a single class (the CLAUDE.md bug)
    # fails every row except its own.
    _CATALOG = [
        ("catboost", "CatBoostPredictor"),
        ("xgboost", "XGBoostPredictor"),
        ("lightgbm", "LightGBMPredictor"),
        ("random_forest", "RandomForestPredictor"),
        ("svr", "SVRPredictor"),
        ("knn", "KNNPredictor"),
        ("gaussian_process", "GaussianProcessPredictor"),
        ("bayesian_ridge", "BayesianRidgePredictor"),
        ("tabpfn", "TabPFNPredictor"),
    ]

    @pytest.mark.parametrize(("name", "expected_class"), _CATALOG)
    def test_each_named_model_selects_its_own_predictor_class(self, name, expected_class):
        # Config performance=<name> must instantiate <name>'s own predictor class,
        # NOT collapse to CatBoost (regression guard, CLAUDE.md).
        strategy = _make_strategy({"performance": name, "energy": "kavier_power"})
        assert type(strategy.throughput_predictor).__name__ == expected_class

    def test_unknown_performance_name_falls_back_to_intelligent(self):
        # Contract: an unrecognised performance name logs a warning and falls back
        # to the "intelligent" composite (not a crash, not CatBoost).
        strategy = _make_strategy({"performance": "totally-bogus-model"})
        assert isinstance(strategy.throughput_predictor, CacheThenPhysicsPredictor)


# ---------------------------------------------------------------------------
# the two keys are honoured INDEPENDENTLY (the exp4 use case)
# ---------------------------------------------------------------------------


class TestPerformanceAndEnergyAreIndependent:
    def test_exp4_combo_tabpfn_perf_kavier_energy(self):
        # The exact combo exp4 will use: performance=tabpfn, energy=kavier.
        # ("kavier" for energy is exposed in config as "kavier_power".) The energy
        # predictor is the throughput-engine-wrapping variant (WRAPS_THROUGHPUT_ENGINE),
        # which is what lets recommend() reuse one Kavier call for both metrics.
        strategy = _make_strategy({"performance": "tabpfn", "energy": "kavier_power"})
        assert type(strategy.throughput_predictor).__name__ == "TabPFNPredictor"
        assert isinstance(strategy.power_predictor, KavierPowerPredictor)
        assert strategy.power_predictor.WRAPS_THROUGHPUT_ENGINE is True

    def test_setting_only_energy_leaves_performance_at_intelligent_default(self):
        # Only the energy key is set; performance is omitted -> stays at the
        # "intelligent" default (cache→physics). Proves one key does not silently
        # reset the other.
        strategy = _make_strategy({"energy": "kavier_power"})
        assert isinstance(strategy.throughput_predictor, CacheThenPhysicsPredictor)
        assert isinstance(strategy.power_predictor, KavierPowerPredictor)

    def test_unknown_energy_name_raises_value_error(self):
        # Contract: unknown energy names are rejected loudly (not silently defaulted),
        # so a typo in a config can never quietly change the energy model.
        with pytest.raises(ValueError, match="Unknown energy predictor"):
            _make_strategy({"energy": "not-a-real-energy-model"})


# ---------------------------------------------------------------------------
# omitting the keys preserves today's defaults (performance=intelligent,
# energy=kavier_power) — and the intelligent default is wired cache→physics
# ---------------------------------------------------------------------------


class TestDefaultsPreservedWhenKeysOmitted:
    def test_omitted_predictors_yields_intelligent_composite_wired_cache_then_physics(self):
        # No predictors section at all -> the documented default pair, and the
        # "intelligent" throughput predictor is specifically the composite that
        # tries an exact cache hit first, then Kavier physics. We assert the wiring
        # (cache=Retrieval, physics=Kavier) and the spec name string, not just the
        # outer class — a composite wired the wrong way round would pass a bare
        # isinstance but fail here.
        strategy = _make_strategy(None)
        tp = strategy.throughput_predictor
        assert isinstance(tp, CacheThenPhysicsPredictor)
        assert isinstance(tp._cache, RetrievalPredictor)
        assert isinstance(tp._physics, KavierPredictor)
        assert tp.get_name() == "intelligent (cache→kavier)"
        assert isinstance(strategy.power_predictor, KavierPowerPredictor)

    def test_empty_none_and_explicit_intelligent_resolve_identically(self):
        # Three spellings of "use the defaults" must be equivalent: predictors
        # omitted (None), present-but-empty ({}), and spelled out explicitly. If
        # any diverged, an absent config would not reproduce documented behaviour.
        none_s = _make_strategy(None)
        empty_s = _make_strategy({})
        explicit_s = _make_strategy({"performance": "intelligent", "energy": "kavier_power"})
        for other in (empty_s, explicit_s):
            assert type(none_s.throughput_predictor) is type(other.throughput_predictor)
            assert type(none_s.power_predictor) is type(other.power_predictor)


# ---------------------------------------------------------------------------
# the on-disk YAML loader (load_strategy_config) surfaces both keys
# ---------------------------------------------------------------------------


class TestYamlLoaderSurfacesBothKeys:
    def _write(self, tmp_path: Path, predictors: dict) -> Path:
        cfg = {
            "strategy": {"name": "multi_objective", "preset": "balanced"},
            "predictors": predictors,
            "grid": {"batch_sizes": [4, 8], "total_gpus": [1, 2], "top_k": 1},
        }
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return path

    def test_loader_reads_both_keys_from_yaml_file(self, tmp_path):
        # Oracle: the loader must surface exactly the strings written to disk — it
        # does not validate them, so a non-default marker string proves passthrough.
        path = self._write(tmp_path, {"performance": "xgboost", "energy": "custom_energy"})
        loaded = load_strategy_config(path)
        assert loaded["predictors"]["performance"] == "xgboost"
        assert loaded["predictors"]["energy"] == "custom_energy"

    def test_loader_defaults_when_predictors_absent(self, tmp_path):
        # A YAML with no predictors section -> loader supplies the documented
        # defaults verbatim (performance=intelligent, energy=kavier_power).
        cfg = {
            "strategy": {"name": "multi_objective", "preset": "balanced"},
            "grid": {"batch_sizes": [4, 8], "total_gpus": [1, 2], "top_k": 1},
        }
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        loaded = load_strategy_config(path)
        assert loaded["predictors"]["performance"] == "intelligent"
        assert loaded["predictors"]["energy"] == "kavier_power"

    def test_loaded_yaml_drives_strategy_selection(self, tmp_path):
        # End-to-end: YAML on disk -> load_strategy_config -> create_strategy.
        # "kavier" must yield the bare physics predictor, i.e. specifically NOT the
        # intelligent composite you'd get from the default — so this pins that the
        # loaded value actually steered selection.
        path = self._write(tmp_path, {"performance": "kavier", "energy": "kavier_power"})
        loaded = load_strategy_config(path)
        strategy = PolicyFactory.create_strategy(config=loaded)
        assert isinstance(strategy.throughput_predictor, KavierPredictor)
        assert not isinstance(strategy.throughput_predictor, CacheThenPhysicsPredictor)
        assert isinstance(strategy.power_predictor, KavierPowerPredictor)
