"""Tests for the interactive recommender (``coastline.cli.interactive``).

They cover the engine (recommend logic without prompts) and the non-interactive Typer path; no
test needs a TTY. Every test pins the analytical Kavier predictor, so no trained ML model is
unpickled (unpickling xgboost segfaults on some hosts). Pipeline results are checked through
invariants (grid membership, finite values, the runtime and energy identities). The
``recommendation_rationale`` text and the JSON output are compared with values worked out from
known inputs.
"""

from __future__ import annotations

import json
import math

import pytest
from typer.testing import CliRunner

from coastline.cli import interactive as ui_app
from coastline.sdk.io.interface.json_output import save_recommendation_to_json
from coastline.sdk.models.recommendation import Recommendation
from coastline.sdk.recommend import engine

# Bind the original at import time so monkeypatching ``engine.defaults`` (in the
# CLI tests) can't recurse back into this helper.
_ORIG_DEFAULTS = engine.defaults

# The defaults set max_gpus=16 and the grid keeps the GPU_BUDGETS entries <= max_gpus:
# GPU_BUDGETS=(1, 2, 4, 8, 16, 32, ...) filtered to <= 16 gives this set.
_EXPECTED_GPU_BUDGETS = {1, 2, 4, 8, 16}


def _kavier_defaults() -> dict:
    """Non-interactive defaults, pinned to the Kavier predictor + a calibrated workload."""
    answers = _ORIG_DEFAULTS(engine.resolve_options())
    answers["predictor"] = "kavier"  # analytical engine, no ML unpickle
    answers["llm_model"] = "mistral-7b-v0.1"
    answers["gpu_model"] = "NVIDIA-A100-SXM4-80GB"
    return answers


def test_run_pipeline_yields_only_gpu_budget_configs_with_finite_throughput():
    """Every ranked config comes from the GPU-budget grid; Kavier gives a finite
    positive throughput for this supported (model, gpu) pair; top_k is respected."""
    recs, meta = engine.run_pipeline(_kavier_defaults(), top_k=3)
    # answers["predictor"] must flow through build_config into meta unchanged.
    assert meta["predictor"] == "kavier"
    assert 1 <= len(recs) <= 3  # top_k=3 caps the table
    for rec in recs:
        # Grid only enumerates the power-of-two budgets <= max_gpus(16).
        assert rec.total_gpus in _EXPECTED_GPU_BUDGETS
        # The layout multiplies out to total_gpus.
        assert rec.gpus_per_node * rec.number_of_nodes == rec.total_gpus
        # For a supported workload Kavier's throughput is finite and > 0.
        assert rec.predicted_throughput is not None
        assert math.isfinite(rec.predicted_throughput)
        assert rec.predicted_throughput > 0


def test_total_tokens_is_product_of_dataset_epochs_and_seq_len():
    """meta.total_tokens = dataset_size x epochs x tokens_per_sample."""
    answers = _kavier_defaults()
    answers.update({"dataset_size": 10_000, "epochs": 2, "tokens_per_sample": 1024})
    _, meta = engine.run_pipeline(answers, top_k=3)
    # 10_000 x 2 x 1024 = 20_480_000
    assert meta["total_tokens"] == 20_480_000


def test_runtime_and_energy_satisfy_the_physical_identities():
    """runtime_energy gives runtime = tokens / throughput and energy [Wh] = power x gpus x hours.

    Both are checked in rearranged form, so the test does not repeat the implementation's
    expression, and a /1000 in place of /3600 fails."""
    answers = _kavier_defaults()
    answers.update({"dataset_size": 10_000, "epochs": 2, "tokens_per_sample": 1024})
    recs, meta = engine.run_pipeline(answers, top_k=3)
    rec = recs[0]
    total_tokens = meta["total_tokens"]
    runtime, energy = engine.runtime_energy(rec, total_tokens)

    thr = rec.predicted_throughput
    power = rec.metadata["predicted_power_watts"]  # W per GPU
    # runtime x throughput == tokens
    assert runtime * thr == pytest.approx(total_tokens)
    # energy [Wh] = per-GPU power [W] x GPUs x runtime [h]
    hours = runtime / 3600.0
    assert energy == pytest.approx(power * rec.total_gpus * hours)
    # A /1000 in place of /3600 would give this value.
    assert energy != pytest.approx(power * rec.total_gpus * runtime / 1000.0)


def test_save_writes_json_schema_mapping_the_recommendation_fields(tmp_path):
    """The JSON holds the config block, the throughput and a tokens/watt efficiency equal to
    throughput / power."""
    recs, _ = engine.run_pipeline(_kavier_defaults(), top_k=3)
    rec = recs[0]
    out = tmp_path / "rec.json"
    save_recommendation_to_json(rec, out)
    payload = json.loads(out.read_text())

    cfg = payload["configuration"]
    assert cfg["total_gpus"] == rec.total_gpus
    assert cfg["gpus_per_node"] == rec.gpus_per_node
    assert cfg["workers"] == rec.number_of_nodes
    # The written layout still multiplies out to total_gpus.
    assert cfg["gpus_per_node"] * cfg["workers"] == cfg["total_gpus"]
    # The serializer rounds floats to 2 decimals for display.
    assert payload["performance"]["throughput_tokens_per_sec"] == round(rec.predicted_throughput, 2)
    # Efficiency is tokens/watt, computed here from throughput and power.
    power = rec.metadata["predicted_power_watts"]
    assert payload["energy"]["power_watts"] == round(power, 2)
    assert payload["energy"]["efficiency_tokens_per_watt"] == pytest.approx(round(rec.predicted_throughput / power, 2))


def test_cli_no_interactive_with_save_writes_a_valid_config(tmp_path, monkeypatch):
    """`--no-interactive --save` exits 0 and writes a config whose layout multiplies out and
    whose GPU count is on the budget grid."""
    monkeypatch.setattr(engine, "defaults", lambda opts: _kavier_defaults())
    out = tmp_path / "cli_rec.json"
    result = CliRunner().invoke(ui_app.app, ["--no-interactive", "--save", str(out), "--top-k", "3"])
    assert result.exit_code == 0, result.output
    cfg = json.loads(out.read_text())["configuration"]
    assert cfg["total_gpus"] in _EXPECTED_GPU_BUDGETS
    assert cfg["gpus_per_node"] * cfg["workers"] == cfg["total_gpus"]


def test_non_tty_falls_back_to_defaults_without_prompting(tmp_path, monkeypatch):
    """Without a TTY the interactive default uses the defaults and does not prompt.

    _boom stands in for the guided prompts and raises if they run."""
    monkeypatch.setattr(engine, "defaults", lambda opts: _kavier_defaults())

    def _boom(seed=None):
        raise AssertionError("guided prompts should not run without a TTY")

    monkeypatch.setattr(ui_app, "_recommend_inputs", _boom)
    out = tmp_path / "tty_rec.json"
    # Without --no-interactive, interactive stays True, but CliRunner's stdin is not a
    # terminal, so the defaults branch runs.
    result = CliRunner().invoke(ui_app.app, ["--save", str(out)])
    assert result.exit_code == 0, result.output
    assert "not a terminal" in result.output
    assert out.exists()


def test_rationale_names_config_goal_and_runner_up_gap():
    """recommendation_rationale on two known recommendations gives the expected one-line summary.

    The top runs at 1000 tok/s and the runner-up at 800, so the gap is (1000 - 800) / 800 = 25%
    faster. The balanced preset reads 'the best throughput-vs-energy balance'; 2 GPUs take the
    plural."""
    top = Recommendation(
        gpus_per_node=2,
        number_of_nodes=1,
        total_gpus=2,
        strategy="multi_objective",
        predicted_throughput=1000.0,
        metadata={"batch_size": 16},
    )
    runner = Recommendation(
        gpus_per_node=4,
        number_of_nodes=1,
        total_gpus=4,
        strategy="multi_objective",
        predicted_throughput=800.0,
        metadata={"batch_size": 8},
    )
    why = engine.recommendation_rationale([top, runner], {"preset": "balanced"})
    assert why == (
        "2 GPUs (2x1, batch 16) picked for the best throughput-vs-energy balance, "
        "25% faster than the runner-up (4 GPUs, batch 8)."
    )


def test_rationale_uses_singular_gpu_and_strategy_goal_when_alone():
    """Single min_gpu rec: no plural 's', no runner-up clause, goal from strategy_name."""
    only = Recommendation(
        gpus_per_node=1,
        number_of_nodes=1,
        total_gpus=1,
        strategy="min_gpu",
        predicted_throughput=500.0,
        metadata={},
    )
    why = engine.recommendation_rationale([only], {"strategy_name": "min_gpu"})
    # One GPU takes the singular; with no batch in metadata and no runner-up, neither clause appears.
    assert why == "1 GPU (1x1) picked for the fewest GPUs that fit."


def test_rationale_reports_no_feasible_config_for_empty_recs():
    """An empty recommendation list gives the fixed 'no feasible configuration' sentence."""
    assert engine.recommendation_rationale([], {}) == ("No feasible configuration in the search space.")


if __name__ == "__main__":  # pragma: no cover - manual run convenience
    raise SystemExit(pytest.main([__file__, "-q"]))
