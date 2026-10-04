"""Multi-objective weights as in the thesis: S = alpha * s_r + beta * s_e.

alpha weights the runtime score s_r and beta the energy score s_e, as in the thesis score and its
preset table: balanced 0.5/0.5, performance 0.8/0.2, energy 0.2/0.8. 'energy-saver', the
thesis name of the energy preset, is accepted in any letter case. The performance preset picks the
fast candidate and the energy preset the low-power one.
"""

from __future__ import annotations

from importlib import resources
from typing import Optional

import pytest

import coastline
from coastline.sdk.constants import PRESET_WEIGHTS
from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.recommendation import Prediction
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.pipeline.selection import EvaluatedCandidate, rank_candidates
from coastline.sdk.policies import PolicyFactory
from coastline.sdk.policies.multi_objective import MultiObjectiveStrategy, normalize_preset
from coastline.sdk.recommend._goals import normalize_goal

GPU = "NVIDIA-A100-SXM4-80GB"

# Two candidates on 2 GPUs: FAST (batch 4) has the higher throughput and the higher power, SLOW
# (batch 8) the reverse. Over the pair, FAST has runtime score 1 and energy score 0.
TABLE = {4: (900.0, 380.0), 8: (200.0, 40.0)}  # batch: (throughput, per-GPU power)
GRID = {"batch_sizes": [4, 8], "total_gpus": [2], "top_k": 2}


class _TablePredictor:
    def predict(self, workload: WorkloadSpec, context: SystemContext) -> Optional[Prediction]:
        throughput, power = TABLE[workload.batch_size]
        return Prediction(
            gpus_per_node=workload.gpus_per_node or 1,
            number_of_nodes=workload.number_of_nodes or 1,
            total_gpus=workload.total_gpus,
            predicted_throughput=throughput,
            predicted_power=power,
        )

    def get_name(self) -> str:  # pragma: no cover - trivial
        return "table"


def _context() -> SystemContext:
    return SystemContext(
        available_gpu_models=[GPU],
        max_gpus=8,
        gpu_memory={GPU: 80},
        constraints=Constraints(max_gpus=8, gpus_per_node=8, max_nodes=1),
    )


def _workload() -> WorkloadSpec:
    return WorkloadSpec(
        llm_model="mistral-7b-v0.1", fine_tuning_method="lora", gpu_model=GPU, tokens_per_sample=1024, batch_size=4
    )


def _strategy(**kw) -> MultiObjectiveStrategy:
    predictor = _TablePredictor()
    return MultiObjectiveStrategy(
        throughput_predictor=predictor,
        power_predictor=predictor,
        config={"grid": GRID, "predictors": {"feasibility": "none"}},
        **kw,
    )


def _pick(**kw) -> int:
    """The batch size of the winner: 4 is the fast candidate, 8 the low-power one."""
    return int(_strategy(**kw).recommend(_workload(), _context())[0].metadata["batch_size"])


def test_preset_weights_are_the_thesis_table():
    assert PRESET_WEIGHTS["balanced"] == (0.5, 0.5)
    assert PRESET_WEIGHTS["performance"] == (0.8, 0.2)
    assert PRESET_WEIGHTS["energy"] == (0.2, 0.8)
    for base in ("balanced", "performance", "energy"):
        assert PRESET_WEIGHTS[f"{base}-frontier"] == PRESET_WEIGHTS[base]


def test_combined_score_is_alpha_times_runtime_plus_beta_times_energy():
    candidate = EvaluatedCandidate(
        gpus_per_node=1,
        number_of_nodes=1,
        total_gpus=1,
        throughput=100.0,
        power=100.0,
        runtime=None,
        throughput_score=0.9,  # runtime score s_r
        power_score=0.6,  # energy score s_e
        combined_score=0.0,
        feasibility_metadata={},
    )
    rank_candidates([candidate], "balanced", alpha=0.7, beta=0.3, top_k=1)
    assert candidate.combined_score == pytest.approx(0.7 * 0.9 + 0.3 * 0.6)


@pytest.mark.parametrize(
    "policy,message",
    [("min_gpu", "ranks energy, balanced or performance"), ("custom", "energy, balanced, performance")],
)
def test_rank_candidates_rejects_a_policy_it_does_not_rank(policy, message):
    candidate = EvaluatedCandidate(
        gpus_per_node=1,
        number_of_nodes=1,
        total_gpus=1,
        throughput=100.0,
        power=100.0,
        runtime=None,
        throughput_score=1.0,
        power_score=1.0,
        combined_score=0.0,
        feasibility_metadata={},
    )
    with pytest.raises(ValueError, match=message):
        rank_candidates([candidate], policy, top_k=1)


def test_alpha_alone_picks_the_fastest_and_beta_alone_the_lowest_power():
    assert _pick(alpha=1.0, beta=0.0) == 4
    assert _pick(alpha=0.0, beta=1.0) == 8
    assert _pick(alpha=0.9, beta=0.1) == 4
    assert _pick(alpha=0.1, beta=0.9) == 8


@pytest.mark.parametrize("preset,winner", [("performance", 4), ("energy", 8)])
def test_preset_winners(preset, winner):
    strategy = _strategy(preset=preset)
    assert (strategy.alpha, strategy.beta) == PRESET_WEIGHTS[preset]
    assert _pick(preset=preset) == winner


@pytest.mark.parametrize("given", ["energy-saver", "Energy-Saver", " ENERGY-SAVER "])
def test_energy_saver_is_the_energy_preset(given):
    assert normalize_preset(given) == "energy"
    strategy = _strategy(preset=given)
    assert strategy.preset == "energy"
    assert (strategy.alpha, strategy.beta) == (0.2, 0.8)
    assert strategy.get_name() == "multi_objective_energy"


@pytest.mark.parametrize("given", ["energy-saver", "Energy-Saver"])
def test_energy_saver_is_the_energy_goal(given):
    assert normalize_goal(given) == "energy"


def _kavier_config(strategy: dict) -> dict:
    return {
        "strategy": {"name": "multi_objective", **strategy},
        "predictors": {"performance": "kavier", "energy": "kavier_power", "feasibility": "rules"},
        "grid": {"batch_sizes": [8], "total_gpus": [1, 8], "top_k": 1},
    }


def _factory_pick(strategy: dict) -> int:
    built = PolicyFactory.create_strategy(config=_kavier_config(strategy))
    return built.recommend(_workload(), _context())[0].total_gpus


def test_yaml_alpha_and_beta_follow_the_thesis_meaning():
    # 8 GPUs run faster than 1; 1 GPU draws less cluster power.
    assert _factory_pick({"alpha": 1.0, "beta": 0.0}) == 8
    assert _factory_pick({"alpha": 0.0, "beta": 1.0}) == 1


def test_yaml_energy_saver_preset():
    built = PolicyFactory.create_strategy(config=_kavier_config({"preset": "Energy-Saver"}))
    assert built.get_name() == "multi_objective_energy"


def test_facade_weights_and_energy_saver_alias():
    rec = coastline.Coastline("kavier", feasibility="rules")
    kw = {"total_gpus": [1, 8], "batch_sizes": [8], "max_gpus": 8, "top_k": 1}
    assert rec.recommend(_workload(), alpha=1.0, beta=0.0, **kw)[0].total_gpus == 8
    assert rec.recommend(_workload(), alpha=0.0, beta=1.0, **kw)[0].total_gpus == 1
    saver = rec.recommend(_workload(), preset="energy-saver", **kw)[0]
    energy = rec.recommend(_workload(), preset="energy", **kw)[0]
    assert saver.metadata["preset"] == "energy"
    assert (saver.total_gpus, saver.metadata["alpha"], saver.metadata["beta"]) == (energy.total_gpus, 0.2, 0.8)


def test_batch_api_energy_saver_goal_matches_energy():
    job = _workload().model_dump(include={"llm_model", "fine_tuning_method", "gpu_model", "tokens_per_sample"})
    job["batch_size"] = 8
    saver = coastline.recommend([job], goal="Energy-Saver", predictor="kavier", feasibility="rules", max_gpus=8)
    energy = coastline.recommend([job], goal="energy", predictor="kavier", feasibility="rules", max_gpus=8)
    assert saver.drop(columns=["goal"], errors="ignore").equals(energy.drop(columns=["goal"], errors="ignore"))


def test_explain_names_alpha_runtime_and_beta_energy(capsys):
    from coastline.cli import main

    base = ["explain", "--model", "mistral-7b-v0.1", "--method", "lora", "--gpu-model", GPU, "--tokens", "1024"]
    base += ["--batch-size", "16", "--predictor", "kavier", "--feasibility", "rules", "--top-k", "1"]
    main([*base, "--preset", "performance"])
    performance = capsys.readouterr().out
    main([*base, "--preset", "Energy-Saver"])
    saver = capsys.readouterr().out

    assert "alpha=0.80 runtime, beta=0.20 energy" in performance
    assert "alpha=0.20 runtime, beta=0.80 energy" in saver
    assert "preset=energy" in saver


def test_recommend_trace_accepts_energy_saver_as_a_goal():
    from coastline.cli.recommend_trace import _build_parser

    args = _build_parser().parse_args(["--input", "in.csv", "--output", "out.csv", "--goal", "Energy-Saver"])
    assert normalize_goal(args.goal) == "energy"


def test_dashboard_accepts_energy_saver_and_labels_the_weights():
    from fastapi.testclient import TestClient

    from coastline.ui.app import app

    body = {
        "llm_model": "mistral-7b-v0.1",
        "fine_tuning_method": "lora",
        "gpu_model": GPU,
        "tokens_per_sample": 1024,
        "batch_size": 8,
        "prediction_model": "kavier",
        "strategy": "multi_objective",
        "preset": "Energy-Saver",
        "total_gpus": 8,
    }
    with TestClient(app) as client:
        resp = client.post("/api/recommend", json=body)
    assert resp.status_code == 200, resp.text
    assert resp.json()["preset"] == "energy"

    ui = resources.files("coastline.ui")
    page = (ui / "templates" / "index.html").read_text(encoding="utf-8")
    script = (ui / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert "Performance (0.8 / 0.2)" in page
    assert "Energy-saver (0.2 / 0.8)" in page
    assert "alpha/beta = 0.8 / 0.2" in script  # performance
    assert "alpha/beta = 0.2 / 0.8" in script  # energy-saver
