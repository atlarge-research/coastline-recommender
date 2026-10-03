"""Tests for the factories that map a config name to a throughput or power predictor.

  * ``PolicyFactory.throughput_predictor`` and ``PolicyFactory.power_predictor``
    (``sdk/policies/__init__.py``): performance names intelligent, cache, kavier and the ML
    models; energy name kavier_power.
  * ``create_physics_driven`` (``sdk/predictors/factory.py``): builds the Kavier predictor.
  * ``workflow._create_throughput_predictor`` and ``_create_power_predictor``, which delegate to
    ``PolicyFactory``.

The tests check the class and wiring of the returned predictor and do not call ``predict()`` on
an ML predictor: those load their pickled model on the first ``predict``, and unpickling xgboost
and similar models in this process can segfault. ``intelligent`` returns an exact cache match (a
measured past run) when there is one, else the Kavier estimate.
"""

import pytest

from coastline.sdk.pipeline import workflow as wf
from coastline.sdk.policies import PolicyFactory, _build_named_ml_predictor
from coastline.sdk.predictors.energy import KavierPowerPredictor
from coastline.sdk.predictors.factory import create_physics_driven
from coastline.sdk.predictors.performance.composite import CacheThenSimulatePredictor
from coastline.sdk.predictors.performance.physics import KavierPredictor
from coastline.sdk.predictors.performance.retrieval.cache_predictor import RetrievalPredictor

# create_physics_driven


class TestPredictorFactory:
    def test_create_physics_driven_returns_kavier(self):
        predictor = create_physics_driven()
        assert isinstance(predictor, KavierPredictor)


# PolicyFactory.throughput_predictor (performance: string-keyed)


class TestThroughputPredictorFactory:
    def test_kavier_returns_kavier_predictor(self):
        predictor = PolicyFactory.throughput_predictor({"performance": "kavier"})
        assert isinstance(predictor, KavierPredictor)

    def test_cache_returns_retrieval_predictor(self):
        predictor = PolicyFactory.throughput_predictor({"performance": "cache"})
        assert isinstance(predictor, RetrievalPredictor)

    def test_intelligent_wires_cache_first_then_fallback(self):
        # "intelligent" returns an exact cache hit of a past run, else the Kavier estimate. Both
        # slots are checked, so a swapped order or two physics predictors fails.
        predictor = PolicyFactory.throughput_predictor({"performance": "intelligent"})
        assert isinstance(predictor, CacheThenSimulatePredictor)
        assert isinstance(predictor._cache, RetrievalPredictor)  # tried first
        assert isinstance(predictor._fallback, KavierPredictor)  # fallback

    def test_intelligent_fallback_model_is_configurable(self):
        # A cache miss uses the configured `fallback` model (Kavier by default); `fallback: xgboost`
        # uses that ML model while the cache stays first. The test does not call .predict.
        default = PolicyFactory.throughput_predictor({"performance": "intelligent"})
        assert isinstance(default._fallback, KavierPredictor)
        custom = PolicyFactory.throughput_predictor({"performance": "intelligent", "fallback": "xgboost"})
        assert isinstance(custom, CacheThenSimulatePredictor)
        assert isinstance(custom._cache, RetrievalPredictor)  # still cache-first
        assert type(custom._fallback).__name__ == "SklearnPortfolioPredictor"  # a miss uses the ML model
        assert custom._fallback.get_name() == "xgboost"

    @pytest.mark.parametrize("bad_fallback", ["intelligent", "cache", "totally-bogus"])
    def test_a_fallback_that_is_not_a_simulation_model_raises(self, bad_fallback):
        # A `fallback` that names a caching predictor ("intelligent", "cache") or an unknown model
        # raises and lists the simulation models, as an unknown `performance` name does. This also
        # keeps a cache from nesting in another or recursing through throughput_predictor.
        with pytest.raises(ValueError, match=f"unknown fallback predictor '{bad_fallback}'") as excinfo:
            PolicyFactory.throughput_predictor({"performance": "intelligent", "fallback": bad_fallback})
        options = str(excinfo.value).split("choose from")[1]
        assert "'kavier'" in options and "'xgboost'" in options
        assert "'cache'" not in options and "'intelligent'" not in options

    @pytest.mark.parametrize("spelling", ["Kavier", " kavier ", "PHYSICS"])
    def test_the_fallback_name_ignores_letter_case_and_spaces(self, spelling):
        predictor = PolicyFactory.throughput_predictor({"performance": "intelligent", "fallback": spelling})
        assert isinstance(predictor._fallback, KavierPredictor)

    def test_a_fallback_model_name_ignores_letter_case(self):
        # 'XGBoost' resolves to xgboost. The test does not call .predict.
        predictor = PolicyFactory.throughput_predictor({"performance": "intelligent", "fallback": "XGBoost"})
        assert predictor._fallback.get_name() == "xgboost"

    def test_a_blank_fallback_keeps_the_kavier_default(self):
        # A key left empty in the YAML (`fallback:`) loads as None.
        predictor = PolicyFactory.throughput_predictor({"performance": "intelligent", "fallback": None})
        assert isinstance(predictor._fallback, KavierPredictor)

    @pytest.mark.parametrize("name", ["xgboost", "catboost"])
    def test_named_ml_model_routes_to_its_own_predictor(self, name):
        # Each name resolves to a predictor that reports that name. xgboost and catboost share
        # SklearnPortfolioPredictor and differ by get_name(). The test does not call .predict.
        predictor = PolicyFactory.throughput_predictor({"performance": name})
        assert predictor.get_name() == name
        # A named model gets the ML predictor itself, with no cache in front.
        assert not isinstance(predictor, CacheThenSimulatePredictor)

    def test_unknown_name_raises(self):
        # An unknown performance name raises and lists the valid names.
        with pytest.raises(ValueError, match="unknown predictor 'totally-bogus'"):
            PolicyFactory.throughput_predictor({"performance": "totally-bogus"})


# _build_named_ml_predictor: name to ML predictor, or None


class TestBuildNamedMlPredictor:
    # Models load on the first .predict, which these tests do not call.
    @pytest.mark.parametrize(
        "name, expected_identity",
        [
            # portfolio models report their canonical name; tabpfn keeps its own get_name.
            ("catboost", "catboost"),
            ("xgboost", "xgboost"),
            ("lightgbm", "lightgbm"),
            ("random_forest", "random_forest"),
            ("tabpfn", "TabPFNPredictor"),
        ],
    )
    def test_known_name_builds_its_own_predictor(self, name, expected_identity):
        predictor = _build_named_ml_predictor(name)
        assert predictor.get_name() == expected_identity

    def test_distinct_names_build_distinct_predictors(self):
        # Distinct names give predictors with distinct get_name() values; the six portfolio models
        # share one class.
        names = ["catboost", "xgboost", "lightgbm", "random_forest", "tabpfn"]
        identities = {_build_named_ml_predictor(n).get_name() for n in names}
        assert len(identities) == len(names)  # 5 names, 5 distinct predictors

    def test_unknown_name_returns_none(self):
        assert _build_named_ml_predictor("not-a-real-model") is None


# PolicyFactory.power_predictor (energy: string-keyed)


class TestPowerPredictorFactory:
    def test_kavier_power_returns_kavier_power_predictor(self):
        predictor = PolicyFactory.power_predictor({"energy": "kavier_power"})
        assert isinstance(predictor, KavierPowerPredictor)

    def test_default_when_energy_key_missing_is_kavier_power(self):
        predictor = PolicyFactory.power_predictor({})
        assert isinstance(predictor, KavierPowerPredictor)

    def test_unknown_energy_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown energy predictor"):
            PolicyFactory.power_predictor({"energy": "totally-bogus"})


# workflow module-level factory functions, used by the pipeline


class TestWorkflowThroughputFactory:
    @pytest.mark.parametrize("alias", ["kavier", "physics", "physics_driven"])
    def test_physics_aliases_all_map_to_kavier(self, alias):
        # All three aliases resolve to the Kavier predictor.
        predictor = wf._create_throughput_predictor({"performance": alias})
        assert isinstance(predictor, KavierPredictor)

    @pytest.mark.parametrize("name", ["kavier", "cache", "intelligent", "xgboost", "tabpfn"])
    def test_workflow_never_diverges_from_policyfactory(self, name):
        # The workflow factory delegates to PolicyFactory, so both give the same class per name.
        via_workflow = type(wf._create_throughput_predictor({"performance": name}))
        via_factory = type(PolicyFactory.throughput_predictor({"performance": name}))
        assert via_workflow is via_factory

    def test_default_is_the_intelligent_composite_not_a_bare_engine(self):
        # An empty config gives the "intelligent" default: the cache-then-Kavier composite.
        predictor = wf._create_throughput_predictor({})
        assert isinstance(predictor, CacheThenSimulatePredictor)
        assert not isinstance(predictor, (KavierPredictor, RetrievalPredictor))


class TestWorkflowPowerFactory:
    # TestPowerPredictorFactory covers kavier_power and the missing key (same wiring); this class
    # checks the workflow function's error path.
    def test_unknown_energy_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown energy predictor"):
            wf._create_power_predictor({"energy": "totally-bogus"})
