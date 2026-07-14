"""Tests for the predictor factory wiring.

Covers the string-keyed factory that maps a config / name to a concrete
performance (throughput) or energy (power) predictor:

  * ``PolicyFactory.throughput_predictor`` / ``PolicyFactory.power_predictor``
    (``recommender/recommendation_policies/__init__.py``) — the canonical factory used by the
    strategy layer (performance: intelligent / cache / kavier / named-ML;
    energy: kavier_power).
  * ``create_physics_driven`` (``recommender/predictor_factory.py``) — the
    low-level physics-predictor constructor the above delegates to.
  * ``coastline.sdk.pipeline.workflow._create_throughput_predictor`` /
    ``_create_power_predictor`` — the workflow's near-twin of the above.

Scope / segfault avoidance
--------------------------
These tests assert the *type / wiring* of the returned predictor only. They
never call ``.predict()`` on a data-driven (ML) predictor: all of them lazy-load
their pickled model on first ``predict``, and unpickling xgboost (and friends)
in-process segfaults on the host. Constructing them is safe, so we assert the
returned class without touching the model.

The ``intelligent`` performance path is a cache→physics composite: an exact
cache match (a measured past run) when one exists, else the Kavier analytical
predictor.
"""

import pytest

from coastline.sdk.pipeline import workflow as wf
from coastline.sdk.policies import PolicyFactory, _build_named_ml_predictor
from coastline.sdk.predictors.energy import KavierPowerPredictor
from coastline.sdk.predictors.factory import create_physics_driven
from coastline.sdk.predictors.performance.composite import CacheThenSimulatePredictor
from coastline.sdk.predictors.performance.physics import KavierPredictor
from coastline.sdk.predictors.performance.retrieval.cache_predictor import RetrievalPredictor

# ---------------------------------------------------------------------------
# PredictorFactory (low-level: physics / power / data-driven)
# ---------------------------------------------------------------------------


class TestPredictorFactory:
    def test_create_physics_driven_returns_kavier(self):
        predictor = create_physics_driven()
        assert isinstance(predictor, KavierPredictor)


# ---------------------------------------------------------------------------
# PolicyFactory.throughput_predictor (performance: string-keyed)
# ---------------------------------------------------------------------------


class TestThroughputPredictorFactory:
    def test_kavier_returns_kavier_predictor(self):
        predictor = PolicyFactory.throughput_predictor({"performance": "kavier"})
        assert isinstance(predictor, KavierPredictor)

    def test_cache_returns_retrieval_predictor(self):
        predictor = PolicyFactory.throughput_predictor({"performance": "cache"})
        assert isinstance(predictor, RetrievalPredictor)

    def test_intelligent_wires_cache_first_then_fallback(self):
        # Contract of "intelligent" (CLAUDE.md): exact cache hit of a real past run,
        # ELSE Kavier physics. The cascade ORDER is the spec, so pin both slots:
        # a bug that swapped them (physics tried first) or wired two physics
        # predictors would pass a bare isinstance check but is caught here.
        predictor = PolicyFactory.throughput_predictor({"performance": "intelligent"})
        assert isinstance(predictor, CacheThenSimulatePredictor)
        assert isinstance(predictor._cache, RetrievalPredictor)  # tried first
        assert isinstance(predictor._fallback, KavierPredictor)  # fallback

    def test_intelligent_fallback_model_is_configurable(self):
        # A cache MISS simulates with the configured `fallback` model, not always Kavier.
        # Default stays Kavier; `fallback: xgboost` swaps the miss branch to that ML model
        # while the cache stays first. Constructing is safe; we never call .predict.
        default = PolicyFactory.throughput_predictor({"performance": "intelligent"})
        assert isinstance(default._fallback, KavierPredictor)
        custom = PolicyFactory.throughput_predictor({"performance": "intelligent", "fallback": "xgboost"})
        assert isinstance(custom, CacheThenSimulatePredictor)
        assert isinstance(custom._cache, RetrievalPredictor)  # still cache-first
        assert type(custom._fallback).__name__ == "SklearnPortfolioPredictor"  # miss -> the ML model
        assert custom._fallback.get_name() == "xgboost"

    @pytest.mark.parametrize("bad_fallback", ["intelligent", "cache", "totally-bogus"])
    def test_intelligent_fallback_guard_degrades_to_kavier(self, bad_fallback):
        # A `fallback` that names a caching predictor ("intelligent"/"cache") or an unknown model
        # must NOT nest another cache or recurse — it degrades to Kavier. Pins the
        # _resolve_simulation_predictor guard so a future refactor that re-routes unknown names
        # (e.g. back through throughput_predictor) can't reintroduce infinite recursion.
        predictor = PolicyFactory.throughput_predictor({"performance": "intelligent", "fallback": bad_fallback})
        assert isinstance(predictor, CacheThenSimulatePredictor)
        assert isinstance(predictor._fallback, KavierPredictor)
        assert not isinstance(predictor._fallback, CacheThenSimulatePredictor)

    @pytest.mark.parametrize("name", ["xgboost", "catboost"])
    def test_named_ml_model_routes_to_its_own_predictor(self, name):
        # Oracle = the documented model catalog (CLAUDE.md lists catboost/xgboost/…).
        # Two DISTINCT names must resolve to predictors that self-report their OWN name
        # (get_name), catching the regression noted in workflow.py where every named
        # model silently collapsed to CatBoost. xgboost/catboost now share one class
        # (SklearnPortfolioPredictor) but stay distinguishable by name. Constructing is
        # safe; we never call .predict.
        predictor = PolicyFactory.throughput_predictor({"performance": name})
        assert predictor.get_name() == name
        # Named models must reach the ML branch, NOT fall through to the intelligent
        # default composite (that fallback is reserved for UNKNOWN names).
        assert not isinstance(predictor, CacheThenSimulatePredictor)

    def test_unknown_name_falls_back_to_intelligent_default(self):
        # Unknown performance names log a warning and fall back to the intelligent
        # default rather than raising (lenient by design).
        predictor = PolicyFactory.throughput_predictor({"performance": "totally-bogus"})
        assert isinstance(predictor, CacheThenSimulatePredictor)


# ---------------------------------------------------------------------------
# _build_named_ml_predictor (name -> individual ML predictor or None)
# ---------------------------------------------------------------------------


class TestBuildNamedMlPredictor:
    # Oracle = the documented data-driven model library (CLAUDE.md / _build_named_ml_predictor
    # map). Parametrized over the whole catalog so behavior varies name-by-name, not just a
    # number. Constructing is safe (models unpickle lazily on first .predict, which we never call).
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
        # Regression guard for the bug called out in workflow.py: an earlier duplicate
        # resolver "silently collapsed every named model — e.g. tabpfn — to CatBoost".
        # Independent oracle: N distinct catalog names must yield N distinguishable
        # predictors. get_name() is the identity that survives the portfolio collapse
        # (six models share one class but keep distinct names).
        names = ["catboost", "xgboost", "lightgbm", "random_forest", "tabpfn"]
        identities = {_build_named_ml_predictor(n).get_name() for n in names}
        assert len(identities) == len(names)  # 5 names -> 5 distinguishable predictors

    def test_unknown_name_returns_none(self):
        assert _build_named_ml_predictor("not-a-real-model") is None


# ---------------------------------------------------------------------------
# PolicyFactory.power_predictor (energy: string-keyed)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# workflow module-level factory functions (the near-twin used by the pipeline)
# ---------------------------------------------------------------------------


class TestWorkflowThroughputFactory:
    @pytest.mark.parametrize("alias", ["kavier", "physics", "physics_driven"])
    def test_physics_aliases_all_map_to_kavier(self, alias):
        # All three spelling aliases (CLAUDE.md config map) resolve to the physics engine.
        predictor = wf._create_throughput_predictor({"performance": alias})
        assert isinstance(predictor, KavierPredictor)

    @pytest.mark.parametrize("name", ["kavier", "cache", "intelligent", "xgboost", "tabpfn"])
    def test_workflow_never_diverges_from_policyfactory(self, name):
        # The workflow factory is documented as delegating to PolicyFactory so the two
        # "can never diverge on what performance resolves to" — the old copy here
        # silently collapsed every named model (e.g. tabpfn) to CatBoost.
        # Independent oracle: for each name the workflow's class == PolicyFactory's class.
        # Reintroducing a private resolver in workflow.py breaks this parity.
        via_workflow = type(wf._create_throughput_predictor({"performance": name}))
        via_factory = type(PolicyFactory.throughput_predictor({"performance": name}))
        assert via_workflow is via_factory

    def test_default_is_the_intelligent_composite_not_a_bare_engine(self):
        # Empty config => "intelligent" default. The anti-twin guard: the default must be
        # the cache→physics COMPOSITE, never a bare Kavier/Retrieval (which is what the old
        # divergent workflow copy produced). Falsifiable: return KavierPredictor() and it reds.
        predictor = wf._create_throughput_predictor({})
        assert isinstance(predictor, CacheThenSimulatePredictor)
        assert not isinstance(predictor, (KavierPredictor, RetrievalPredictor))


class TestWorkflowPowerFactory:
    # kavier_power + missing-key default mirror TestPowerPredictorFactory exactly
    # (both delegate to the same wiring); only the error path of this separate
    # workflow function is exercised here.
    def test_unknown_energy_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown energy predictor"):
            wf._create_power_predictor({"energy": "totally-bogus"})
