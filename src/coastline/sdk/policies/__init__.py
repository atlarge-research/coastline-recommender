"""Policy factory for creating recommendation policies."""

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import yaml

from coastline.sdk.constants import EnergyBackend, SelectionPolicy, Strategy
from coastline.sdk.io.run_config import builtin_default_config
from coastline.sdk.pipeline.feasibility import create_feasibility_checker
from coastline.sdk.pipeline.workflow import GridWorkflowPipeline
from coastline.sdk.policies.base import BaseStrategy
from coastline.sdk.policies.min_gpu import MinGPUStrategy
from coastline.sdk.policies.multi_objective import MultiObjectiveStrategy, PolicyPreset
from coastline.sdk.predictors.base import BasePredictor
from coastline.sdk.predictors.energy import KavierPowerPredictor
from coastline.sdk.predictors.factory import create_physics_driven
from coastline.sdk.predictors.performance.data_driven.sklearn_portfolio import (
    Artifact,
    Const,
    Param,
    PortfolioModel,
)
from coastline.sdk.predictors.performance.retrieval.cache_predictor import RetrievalPredictor

logger = logging.getLogger(__name__)

# The coastline repo root (src/coastline/sdk/policies/ -> parents[4]); holds config/.
_REPO_ROOT = Path(__file__).resolve().parents[4]

# Fallback when no config path given and no YAML found on disk — the one built-in default,
# sourced from the bundled default_experiment.yaml (not a second hardcoded copy).
_BUILTIN_DEFAULT_CONFIG: dict = builtin_default_config()


class PolicyFactory:
    """Factory for creating recommendation policies from config."""

    @staticmethod
    def _default_config_candidates() -> list:
        """The one canonical config to try when none is supplied (env-overridable). Shared with
        the CLI and UI so every door resolves to the same experiment.yaml."""
        from coastline.sdk.io.run_config import default_experiment_path

        return [default_experiment_path()]

    @staticmethod
    def load_config(config_path: Optional[str] = None) -> dict:
        # Explicit path: behave exactly as before (load it, errors propagate).
        if config_path is not None:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
            return config

        # No path given: first existing default file, then built-in.
        for candidate in PolicyFactory._default_config_candidates():
            if not candidate.is_file():
                continue
            with open(candidate, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {candidate}")
            return config

        logger.warning(
            "No strategy config file found (looked for "
            "config/coastline_functionality/experiment.yaml); using built-in default config"
        )
        return copy.deepcopy(_BUILTIN_DEFAULT_CONFIG)  # deep copy: callers may mutate nested dicts

    @staticmethod
    def create_strategy(
        strategy_name: Optional[str] = None,
        preset: Optional[PolicyPreset] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        config: Optional[dict] = None,
    ) -> BaseStrategy:
        if config is None:
            config = PolicyFactory.load_config()

        strategy_config = config.get("strategy", {})
        predictor_config = config.get("predictors", {})

        if strategy_name is None:
            strategy_name = strategy_config.get("name", Strategy.MULTI_OBJECTIVE.value)

        logger.info(f"Creating strategy: {strategy_name}")

        if strategy_name == Strategy.MIN_GPU:
            return PolicyFactory._create_min_gpu_strategy(config, predictor_config)
        elif strategy_name == Strategy.MULTI_OBJECTIVE:
            return PolicyFactory._create_multi_objective_strategy(
                config, strategy_config, predictor_config, preset, alpha, beta
            )
        else:
            raise ValueError(f"Unknown strategy: '{strategy_name}'. Supported: {[s.value for s in Strategy]}")

    @staticmethod
    def _lookup_path(predictor_config: dict) -> Optional[Path]:
        """Resolve ``predictors.lookup``: a measured-runs CSV path, or the literal
        ``default`` for the repo's default lookup DB (jittered sfttrainer sample in
        config/coastline_functionality/). None = the RetrievalPredictor's own
        resolution ($DATA_DIR, then the bundled sample)."""
        lookup = predictor_config.get("lookup")
        if not lookup:
            return None
        if str(lookup).strip().lower() == "default":
            default = _REPO_ROOT / "config" / "coastline_functionality" / "run_database.csv"
            if not default.exists():
                raise FileNotFoundError(
                    "the default run database (config/coastline_functionality/run_database.csv) "
                    "is only available in a repo checkout — pass an explicit lookup CSV path"
                )
            return default
        path = Path(lookup)
        if not path.exists():
            raise FileNotFoundError(f"lookup CSV not found: {path}")
        return path

    @staticmethod
    def _retrieval_predictor(predictor_config: dict, lookup: Optional[Path]):
        """A lookup/cache predictor over ``lookup``, reading throughput/duration from the columns
        named by ``predictors.lookup_throughput_col`` / ``lookup_runtime_col`` (defaults = run DB)."""
        return RetrievalPredictor(
            dataset_path=lookup,
            throughput_col=predictor_config.get("lookup_throughput_col"),
            runtime_col=predictor_config.get("lookup_runtime_col"),
        )

    @staticmethod
    def _resolve_simulation_predictor(name: str):
        """The simulation model used on a cache miss (or directly as ``performance: <name>``):
        Kavier physics or a named ML model. Never resolves to ``intelligent``/``cache`` — a
        fallback that itself cached would nest a second lookup."""
        if name in ("kavier", "physics", "physics_driven"):
            return create_physics_driven()
        named = _build_named_ml_predictor(name)
        if named is not None:
            return named
        logger.warning("Unknown fallback model '%s'; using kavier", name)
        return create_physics_driven()

    @staticmethod
    def throughput_predictor(predictor_config: dict):
        performance_type = predictor_config.get("performance", "intelligent")
        lookup = PolicyFactory._lookup_path(predictor_config)
        if performance_type == "cache":
            return PolicyFactory._retrieval_predictor(predictor_config, lookup)
        if performance_type == "intelligent":
            return PolicyFactory._intelligent_throughput_predictor(predictor_config, lookup)
        if performance_type in ("kavier", "physics", "physics_driven"):
            return create_physics_driven()
        # a specific data-driven model selected by name (catboost, xgboost, …)
        named = _build_named_ml_predictor(performance_type)
        if named is not None:
            return named
        logger.warning("Unknown predictor '%s'; using intelligent default", performance_type)
        return PolicyFactory._intelligent_throughput_predictor(predictor_config, lookup)

    @staticmethod
    def _intelligent_throughput_predictor(predictor_config: dict, lookup: Optional[Path] = None):
        # "intelligent" = an exact cache match (a real measured past run) when one exists for this
        # configuration, else simulate with `fallback` (Kavier by default, or any model by name).
        # A cache miss yields no prediction, so the composite falls through to the fallback per
        # configuration.
        from coastline.sdk.predictors.performance.composite import CacheThenSimulatePredictor

        fallback = predictor_config.get("fallback", "kavier")
        return CacheThenSimulatePredictor(
            cache=PolicyFactory._retrieval_predictor(predictor_config, lookup),
            fallback=PolicyFactory._resolve_simulation_predictor(fallback),
        )

    @staticmethod
    def power_predictor(predictor_config: dict):
        energy_type = predictor_config.get("energy", EnergyBackend.KAVIER_POWER.value)
        if energy_type == EnergyBackend.KAVIER_POWER:
            return KavierPowerPredictor()
        raise ValueError(f"Unknown energy predictor: '{energy_type}'. Supported: {[e.value for e in EnergyBackend]}")

    @staticmethod
    def _create_min_gpu_strategy(config: dict, predictor_config: dict) -> MinGPUStrategy:
        throughput = PolicyFactory.throughput_predictor(predictor_config)
        power = PolicyFactory.power_predictor(predictor_config)
        feasibility = create_feasibility_checker(predictor_config)

        pipeline = GridWorkflowPipeline.from_config(
            config=config,
            selection_policy=SelectionPolicy.MIN_GPU.value,
            strategy_name=Strategy.MIN_GPU.value,
            throughput_predictor=throughput,
            power_predictor=power,
            feasibility_checker=feasibility,
        )
        return MinGPUStrategy(pipeline=pipeline)

    @staticmethod
    def _create_multi_objective_strategy(
        config: dict,
        strategy_config: dict,
        predictor_config: dict,
        preset: Optional[PolicyPreset],
        alpha: Optional[float],
        beta: Optional[float],
    ) -> MultiObjectiveStrategy:
        throughput_predictor = PolicyFactory.throughput_predictor(predictor_config)
        power_predictor = PolicyFactory.power_predictor(predictor_config)

        yaml_preset = strategy_config.get("preset")  # captured before alpha/beta resolution for conflict detection

        if alpha is None and beta is None:
            alpha = strategy_config.get("alpha")
            beta = strategy_config.get("beta")

        # One-sided weight: derive complement so it isn't silently dropped in favour of the preset.
        if alpha is not None and beta is None:
            beta = max(0.0, 1.0 - float(alpha))
            logger.warning(
                "multi_objective: only alpha=%s was set; deriving beta=%s (=1-alpha). "
                "Set both to control the split explicitly.",
                alpha,
                beta,
            )
        elif beta is not None and alpha is None:
            alpha = max(0.0, 1.0 - float(beta))
            logger.warning(
                "multi_objective: only beta=%s was set; deriving alpha=%s (=1-beta). "
                "Set both to control the split explicitly.",
                beta,
                alpha,
            )

        weights_set = alpha is not None and beta is not None

        # Explicit alpha/beta wins over any preset; warn so the author knows the preset was dropped.
        effective_preset = preset if preset is not None else yaml_preset
        if weights_set and effective_preset is not None:
            logger.warning(
                "multi_objective: both explicit alpha/beta (%s/%s) and preset='%s' "
                "given; the explicit weights win and the preset is ignored.",
                alpha,
                beta,
                effective_preset,
            )
            preset = None

        # Only fall back to a preset when no weights were supplied.
        if preset is None and not weights_set:
            preset = strategy_config.get("preset", "balanced")

        return MultiObjectiveStrategy(
            throughput_predictor=throughput_predictor,
            power_predictor=power_predictor,
            preset=preset,
            alpha=alpha,
            beta=beta,
            config=config,
        )


@dataclass(frozen=True)
class _DedicatedModel:
    """A named model with its own predictor class — a distinct runtime (tabpfn,
    deep_learning) or a return_std path (gaussian_process, bayesian_ridge). Imported
    lazily so an unused ML runtime is never pulled in."""

    module: str
    cls: str

    def build(self, name: str) -> BasePredictor:
        import importlib

        module = importlib.import_module(f"coastline.sdk.predictors.performance.data_driven.{self.module}")
        return getattr(module, self.cls)()


# Every data-driven predictor, keyed by public name. The six portfolio models share
# SklearnPortfolioPredictor and differ ONLY in this table (the metadata hyperparameters
# they surface + whether categoricals are native); the rest keep their own class.
# Module-level so list_predictor_names() can advertise them without importing any ML runtime.
_GRADIENT_BOOSTING = Const("algorithm", "gradient_boosting")
_NAMED_ML_PREDICTORS: dict[str, Union[PortfolioModel, _DedicatedModel]] = {
    "catboost": PortfolioModel(
        native_categorical=True,
        metadata=(
            Param("iterations"),
            Param("depth"),
            Param("learning_rate"),
            Param("l2_leaf_reg"),
            _GRADIENT_BOOSTING,
        ),
    ),
    "xgboost": PortfolioModel(
        metadata=(Param("n_estimators"), Param("max_depth"), Param("learning_rate"), _GRADIENT_BOOSTING),
    ),
    "lightgbm": PortfolioModel(
        metadata=(
            Param("n_estimators"),
            Param("max_depth"),
            Param("num_leaves"),
            Param("learning_rate"),
            _GRADIENT_BOOSTING,
        ),
    ),
    "random_forest": PortfolioModel(metadata=(Artifact("oob_score"),)),
    "svr": PortfolioModel(
        metadata=(
            Const("kernel", "rbf"),
            Param("C", "svr__C"),
            Param("epsilon", "svr__epsilon"),
            Param("gamma", "svr__gamma"),
        ),
    ),
    "knn": PortfolioModel(
        metadata=(
            Param("n_neighbors", "knn__n_neighbors"),
            Param("weights", "knn__weights"),
            Param("p", "knn__p"),
            Const("metric", "minkowski"),
        ),
    ),
    "gaussian_process": _DedicatedModel("gaussian_process_predictor", "GaussianProcessPredictor"),
    "bayesian_ridge": _DedicatedModel("bayesian_ridge_predictor", "BayesianRidgePredictor"),
    "tabpfn": _DedicatedModel("tabpfn_predictor", "TabPFNPredictor"),
    "deep_learning": _DedicatedModel("deep_learning_predictor", "DeepLearningPredictor"),
}

# Physics/retrieval/composite performance predictors (not trained-model names).
_SPECIAL_PREDICTORS: tuple[str, ...] = ("intelligent", "kavier", "cache")


def list_predictor_names() -> tuple[str, ...]:
    """Every accepted throughput-predictor name (the specials plus the trained models)."""
    return _SPECIAL_PREDICTORS + tuple(_NAMED_ML_PREDICTORS)


# physics/physics_driven are internal config aliases for the Kavier physics predictor.
_PREDICTOR_ALIASES: frozenset[str] = frozenset({"physics", "physics_driven"})


def normalize_predictor(name: str) -> str:
    """Lowercase a public predictor spelling to its key and validate it, so a typo fails loudly
    (listing the options) rather than silently falling back to the default. The single validator
    shared by the facade and the batch API."""
    key = str(name).strip().lower()
    if key not in set(list_predictor_names()) | _PREDICTOR_ALIASES:
        raise ValueError(f"unknown predictor {name!r}; choose from {list(list_predictor_names())}")
    return key


def _build_named_ml_predictor(name: str):
    """Construct a data-driven predictor by name, or None if unknown. Lazy import avoids pulling all ML runtimes."""
    spec = _NAMED_ML_PREDICTORS.get(name)
    if spec is None:
        return None
    return spec.build(name)
