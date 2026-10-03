"""Predictor and preset names: any letter case resolves, an unknown name raises.

Replacing an unknown name with a default ('intelligent' or 'balanced') would report numbers from
a model nobody asked for.
"""

from __future__ import annotations

import pytest

from coastline.sdk.constants import Preset
from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.recommendation import Prediction
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.policies import PolicyFactory
from coastline.sdk.policies.multi_objective import MultiObjectiveStrategy
from coastline.sdk.predictors.performance.composite import CacheThenSimulatePredictor
from coastline.sdk.predictors.performance.physics import KavierPredictor

_GRID = {"batch_sizes": [4], "total_gpus": [1, 2], "top_k": 2}


class _StubPredictor:
    """Throughput that grows with the GPU count and a fixed per-GPU power; enough to run a strategy."""

    def predict(self, workload: WorkloadSpec, context: SystemContext) -> Prediction:
        return Prediction(
            gpus_per_node=workload.gpus_per_node or 1,
            number_of_nodes=workload.number_of_nodes or 1,
            total_gpus=workload.total_gpus,
            predicted_throughput=100.0 * workload.total_gpus,
            predicted_power=200.0,
        )

    def get_name(self) -> str:
        return "stub"


def _strategy(**kw) -> MultiObjectiveStrategy:
    stub = _StubPredictor()
    return MultiObjectiveStrategy(
        throughput_predictor=stub,
        power_predictor=stub,
        config={"grid": _GRID, "predictors": {"feasibility": "rules"}},
        **kw,
    )


# Predictor names (PolicyFactory.throughput_predictor)
@pytest.mark.parametrize("name", ["Kavier", "KAVIER", " kavier "])
def test_throughput_predictor_ignores_letter_case(name):
    # 'Kavier' differs from the key 'kavier' only in case, so it builds the Kavier predictor.
    assert isinstance(PolicyFactory.throughput_predictor({"performance": name}), KavierPredictor)


@pytest.mark.parametrize("name", ["kavir", "xgboostt", "gpt5"])
def test_throughput_predictor_rejects_an_unknown_name(name):
    with pytest.raises(ValueError, match="unknown predictor"):
        PolicyFactory.throughput_predictor({"performance": name})


def test_an_unknown_name_in_a_config_fails_the_strategy_build():
    # recommend_csv and recommend-job --config build through create_strategy, so a typo in
    # predictors.performance stops the build.
    config = {
        "strategy": {"name": "multi_objective", "preset": "balanced"},
        "predictors": {"performance": "kavir", "energy": "kavier_power", "feasibility": "rules"},
        "grid": _GRID,
    }
    with pytest.raises(ValueError, match="unknown predictor 'kavir'"):
        PolicyFactory.create_strategy(config=config)


def test_an_unset_predictor_still_means_the_intelligent_default():
    # A missing or null 'performance' key is not a typo: it keeps the documented default.
    assert isinstance(PolicyFactory.throughput_predictor({"performance": None}), CacheThenSimulatePredictor)


# Preset names (MultiObjectiveStrategy, and PolicyFactory through it)
@pytest.mark.parametrize(
    "given,canonical,weights",
    [
        ("Performance", "performance", (0.2, 0.8)),
        ("ENERGY", "energy", (0.8, 0.2)),
        (" Balanced ", "balanced", (0.5, 0.5)),
        ("Energy-Frontier", "energy-frontier", (0.8, 0.2)),
        (Preset.ENERGY, "energy", (0.8, 0.2)),
    ],
)
def test_preset_resolves_regardless_of_case(given, canonical, weights):
    strat = _strategy(preset=given)
    assert strat.preset == canonical
    assert (strat.alpha, strat.beta) == pytest.approx(weights)
    assert strat.get_name() == f"multi_objective_{canonical}"


@pytest.mark.parametrize("preset", ["perfomance", "low_energy", "custom", ""])
def test_unknown_preset_raises_and_lists_the_options(preset):
    with pytest.raises(ValueError, match="unknown preset") as excinfo:
        _strategy(preset=preset)
    assert "performance" in str(excinfo.value) and "balanced" in str(excinfo.value)


def test_no_preset_still_means_balanced():
    strat = _strategy()
    assert strat.preset == "balanced"
    assert (strat.alpha, strat.beta) == (0.5, 0.5)


def test_explicit_weights_still_win_over_a_preset():
    # A preset given next to explicit weights is ignored (PolicyFactory logs that), so it is not
    # validated either: the weights decide.
    strat = _strategy(preset="custom", alpha=1.0, beta=3.0)
    assert strat.preset == "custom"
    assert (strat.alpha, strat.beta) == pytest.approx((0.25, 0.75))


def test_normalize_preset_returns_the_canonical_key():
    from coastline.sdk.policies.multi_objective import normalize_preset

    assert normalize_preset("PERFORMANCE-frontier") == "performance-frontier"
    with pytest.raises(ValueError, match="unknown preset 'fast'"):
        normalize_preset("fast")


@pytest.mark.parametrize(
    "preset,expected", [("ENERGY", "multi_objective_energy"), ("Performance", "multi_objective_performance")]
)
def test_config_preset_is_case_insensitive_through_the_factory(preset, expected):
    config = {
        "strategy": {"name": "multi_objective", "preset": preset},
        "predictors": {"performance": "kavier", "energy": "kavier_power", "feasibility": "rules"},
        "grid": _GRID,
    }
    assert PolicyFactory.create_strategy(config=config).get_name() == expected


def test_config_preset_typo_fails_the_strategy_build():
    config = {
        "strategy": {"name": "multi_objective", "preset": "perfomance"},
        "predictors": {"performance": "kavier", "energy": "kavier_power", "feasibility": "rules"},
        "grid": _GRID,
    }
    with pytest.raises(ValueError, match="unknown preset 'perfomance'"):
        PolicyFactory.create_strategy(config=config)


def test_a_case_insensitive_preset_ranks_like_the_canonical_one():
    # The weights reach the ranking: 'Performance' picks what 'performance' picks on the same grid.
    context = SystemContext(
        available_gpu_models=["NVIDIA-A100-SXM4-80GB"],
        max_gpus=8,
        gpu_memory={"NVIDIA-A100-SXM4-80GB": 80},
        constraints=Constraints(max_gpus=8, gpus_per_node=8, max_nodes=1),
    )
    workload = WorkloadSpec(
        llm_model="mistral-7b-v0.1",
        fine_tuning_method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=4,
    )
    upper = _strategy(preset="Performance").recommend(workload, context)
    lower = _strategy(preset="performance").recommend(workload, context)
    assert [(r.total_gpus, r.metadata["alpha"], r.metadata["beta"]) for r in upper] == [
        (r.total_gpus, r.metadata["alpha"], r.metadata["beta"]) for r in lower
    ]
