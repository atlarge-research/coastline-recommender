"""Predictor selection from the ``predictors:`` section of a config::

    predictors:
      performance: <intelligent | kavier | cache | catboost | xgboost | ...>
      energy:      <kavier_power>

PolicyFactory (coastline.sdk.policies) and load_strategy_config (coastline.sdk.io.run_config)
read these keys. The expected mapping is:

    performance:  kavier, physics, physics_driven  -> KavierPredictor
                  cache                            -> RetrievalPredictor
                  intelligent (default)            -> CacheThenSimulatePredictor (cache, then Kavier)
                  catboost, xgboost, ...           -> a predictor whose get_name() is that name
                  unknown name                     -> ValueError
    energy:       kavier_power (default)           -> KavierPowerPredictor
                  unknown name                     -> ValueError

The tests check the type and wiring of the selected predictor and never call predict() on an ML
predictor: the first predict unpickles the model, and unpickling xgboost and similar models in the
test process can segfault. Constructing them is safe.
"""

from pathlib import Path

import pytest
import yaml

import coastline.sdk.predictors.performance.data_driven.ml_common as ml_common
from coastline.sdk.io.run_config import load_strategy_config
from coastline.sdk.policies import PolicyFactory
from coastline.sdk.predictors.energy import KavierPowerPredictor
from coastline.sdk.predictors.performance.composite import CacheThenSimulatePredictor
from coastline.sdk.predictors.performance.physics import KavierPredictor
from coastline.sdk.predictors.performance.retrieval.cache_predictor import RetrievalPredictor


@pytest.fixture(autouse=True)
def _no_ml_artifact(monkeypatch):
    """Point the trained-model lookup at a missing file.

    Named ML predictors resolve their pickle path in ``__init__``; a missing file keeps
    construction independent of the host and loads no model.
    """
    missing = Path("/nonexistent/config_predictor_selection/performance_catboost.pkl")
    monkeypatch.setattr(ml_common, "performance_trained_model_path", lambda stem: missing)


def _make_strategy(predictors: dict | None, strategy_name: str = "multi_objective"):
    """Build a strategy from a config that differs only in its ``predictors`` section.

    ``predictors=None`` leaves the section out.
    """
    config: dict = {
        "strategy": {"name": strategy_name, "preset": "balanced"},
        "grid": {"batch_sizes": [4, 8], "total_gpus": [1, 2], "top_k": 1},
    }
    if predictors is not None:
        config["predictors"] = predictors
    return PolicyFactory.create_strategy(config=config)


# each named ML model selects its own predictor


class TestNamedMLModelSelection:
    # Expected get_name() per config name. The six portfolio models share one class and differ
    # by name. A resolver that mapped every name to one predictor would fail all rows but one.
    _CATALOG = [
        ("catboost", "catboost"),
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("random_forest", "random_forest"),
        ("svr", "svr"),
        ("knn", "knn"),
        ("gaussian_process", "gaussian_process"),
        ("bayesian_ridge", "bayesian_ridge"),
        ("tabpfn", "TabPFNPredictor"),
    ]

    @pytest.mark.parametrize(("name", "expected_identity"), _CATALOG)
    def test_each_named_model_selects_its_own_predictor_class(self, name, expected_identity):
        # performance=<name> gives the predictor for <name>; xgboost, for example, must not
        # resolve to CatBoost.
        strategy = _make_strategy({"performance": name, "energy": "kavier_power"})
        assert strategy.throughput_predictor.get_name() == expected_identity

    def test_unknown_performance_name_raises(self):
        # An unknown name raises and lists the valid names, so a typo cannot run another model.
        with pytest.raises(ValueError, match="unknown predictor 'totally-bogus-model'"):
            _make_strategy({"performance": "totally-bogus-model"})


# the two keys are read independently (as in Exp4)


class TestPerformanceAndEnergyAreIndependent:
    def test_exp4_combo_tabpfn_perf_kavier_energy(self):
        # The Exp4 combination: performance=tabpfn, energy=kavier_power. The energy predictor
        # wraps the throughput engine (WRAPS_THROUGHPUT_ENGINE), which lets recommend() reuse
        # one Kavier call for both metrics.
        strategy = _make_strategy({"performance": "tabpfn", "energy": "kavier_power"})
        assert strategy.throughput_predictor.get_name() == "TabPFNPredictor"
        assert isinstance(strategy.power_predictor, KavierPowerPredictor)
        assert strategy.power_predictor.WRAPS_THROUGHPUT_ENGINE is True

    def test_setting_only_energy_leaves_performance_at_intelligent_default(self):
        # With only the energy key set, performance stays at the intelligent default (cache,
        # then Kavier).
        strategy = _make_strategy({"energy": "kavier_power"})
        assert isinstance(strategy.throughput_predictor, CacheThenSimulatePredictor)
        assert isinstance(strategy.power_predictor, KavierPowerPredictor)

    def test_unknown_energy_name_raises_value_error(self):
        # A typo in the energy name raises, so it cannot change the energy model.
        with pytest.raises(ValueError, match="Unknown energy predictor"):
            _make_strategy({"energy": "not-a-real-energy-model"})


# without the keys the defaults apply: performance=intelligent (cache, then
# Kavier) and energy=kavier_power


class TestDefaultsPreservedWhenKeysOmitted:
    def test_omitted_predictors_yields_intelligent_composite_wired_cache_then_fallback(self):
        # With no predictors section the throughput predictor is the intelligent composite: an
        # exact cache hit first, then Kavier. The wiring is checked as well as the class, since
        # a composite wired the other way round would pass the isinstance check.
        strategy = _make_strategy(None)
        tp = strategy.throughput_predictor
        assert isinstance(tp, CacheThenSimulatePredictor)
        assert isinstance(tp._cache, RetrievalPredictor)
        assert isinstance(tp._fallback, KavierPredictor)
        assert tp.get_name() == "intelligent (cache->Kavier Physics-Based)"
        assert isinstance(strategy.power_predictor, KavierPowerPredictor)

    def test_empty_none_and_explicit_intelligent_resolve_identically(self):
        # A missing section, an empty one and the defaults written out build the same predictors.
        none_s = _make_strategy(None)
        empty_s = _make_strategy({})
        explicit_s = _make_strategy({"performance": "intelligent", "energy": "kavier_power"})
        for other in (empty_s, explicit_s):
            assert type(none_s.throughput_predictor) is type(other.throughput_predictor)
            assert type(none_s.power_predictor) is type(other.power_predictor)


# load_strategy_config reads both keys from a YAML file


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
        # The loader does not validate the names, so a made-up energy name comes back as written.
        path = self._write(tmp_path, {"performance": "xgboost", "energy": "custom_energy"})
        loaded = load_strategy_config(path)
        assert loaded["predictors"]["performance"] == "xgboost"
        assert loaded["predictors"]["energy"] == "custom_energy"

    def test_loader_defaults_when_predictors_absent(self, tmp_path):
        # Without a predictors section the loader fills in performance=intelligent and
        # energy=kavier_power.
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
        # From a YAML file through load_strategy_config to create_strategy: "kavier" gives the
        # plain KavierPredictor, which differs from the default composite.
        path = self._write(tmp_path, {"performance": "kavier", "energy": "kavier_power"})
        loaded = load_strategy_config(path)
        strategy = PolicyFactory.create_strategy(config=loaded)
        assert isinstance(strategy.throughput_predictor, KavierPredictor)
        assert not isinstance(strategy.throughput_predictor, CacheThenSimulatePredictor)
        assert isinstance(strategy.power_predictor, KavierPowerPredictor)
