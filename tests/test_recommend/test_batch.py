"""End-to-end tests for the CSV-to-CSV batch recommender (Kavier predictor, rules feasibility).

The checks rest on facts that hold whatever numbers Kavier produces:
  * min_gpu ranks feasible configs by (total_gpus asc, throughput desc), so with the runtime
    guard off it picks the fewest GPUs in the grid.
  * The rules checker admits any valid per-device workload (there is no divisibility rule).
  * tokens_per_watt = predicted_throughput / per-GPU power.
  * Per-GPU power lies in [idle, TDP] of the recommended GPU.
No Kavier throughput or power value is pinned.
"""

import csv

import pytest
import yaml

from coastline.sdk.recommend.batch_csv import recommend_csv

CANONICAL_HEADER = ["llm_model", "fine_tuning_method", "gpu_model", "tokens_per_sample", "batch_size"]


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _base_config():
    """min_gpu with the runtime guard on (max_slowdown=3).

    Configs slower than a third of the fastest are dropped, which excludes the slow 1-GPU layout.
    """
    return {
        "strategy": {"name": "min_gpu", "max_slowdown": 3.0},
        "predictors": {"performance": "kavier", "energy": "kavier_power", "feasibility": "rules"},
        "grid": {
            "gpu_models": ["NVIDIA-A100-SXM4-80GB"],
            "batch_sizes": [8, 16],
            "total_gpus": [1, 2, 4, 8],
        },
    }


def _guardless_config():
    """The same grid without max_slowdown, so min_gpu picks the fewest feasible GPUs."""
    cfg = _base_config()
    del cfg["strategy"]["max_slowdown"]
    return cfg


def _run_batch(tmp_path, config, header, rows, *, name="in"):
    """Write the config and the input CSV, run the batch recommender, return the output path and rows."""
    cfg_path = tmp_path / f"{name}_config.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    inp = tmp_path / f"{name}.csv"
    _write_csv(inp, header, rows)
    out = tmp_path / f"{name}_out.csv"
    recommend_csv(cfg_path, inp, out)
    return out, list(csv.DictReader(open(out)))


# One recommendation row per input row, with the input echoed.
def test_one_output_row_per_input_row_with_input_echoed(tmp_path):
    rows_in = [
        ["mistral-7b-v0.1", "lora", "NVIDIA-A100-SXM4-80GB", 1024, 16],
        ["mistral-7b-v0.1", "lora", "NVIDIA-A100-SXM4-80GB", 2048, 8],
    ]
    _, rows = _run_batch(tmp_path, _base_config(), CANONICAL_HEADER, rows_in)

    # One output row per input row: none dropped, none duplicated.
    assert len(rows) == 2
    # The input columns come back unchanged and in order.
    assert [r["llm_model"] for r in rows] == ["mistral-7b-v0.1", "mistral-7b-v0.1"]
    assert [r["tokens_per_sample"] for r in rows] == ["1024", "2048"]
    # total_gpus = gpus_per_node * number_of_nodes.
    for r in rows:
        assert int(r["recommended_total_gpus"]) == int(r["recommended_gpus_per_node"]) * int(
            r["recommended_number_of_nodes"]
        )


# min_gpu selection.
def test_min_gpu_selects_single_gpu_when_runtime_guard_disabled(tmp_path):
    # The rules backend only checks per-device batch >= 1 and total_gpus >= 1, so every
    # total_gpus in [1, 2, 4, 8] is feasible. Without the runtime guard min_gpu ranks by
    # total_gpus ascending and picks 1 GPU on 1 node.
    _, rows = _run_batch(
        tmp_path,
        _guardless_config(),
        CANONICAL_HEADER,
        [["mistral-7b-v0.1", "lora", "NVIDIA-A100-SXM4-80GB", 1024, 16]],
    )
    r = rows[0]
    assert r["feasible"] == "True"
    assert int(r["recommended_total_gpus"]) == 1
    assert int(r["recommended_gpus_per_node"]) == 1
    assert int(r["recommended_number_of_nodes"]) == 1
    # At equal GPU count min_gpu prefers the higher throughput, and Kavier throughput rises
    # with batch size, so batch 16 beats batch 8.
    assert int(r["recommended_batch_size"]) == 16


def test_runtime_guard_forces_faster_config_than_min_gpu_alone(tmp_path):
    # Without the guard min_gpu picks 1 GPU, the slowest feasible config. max_slowdown=3
    # removes every config slower than a third of the fastest. On this A100/mistral grid
    # the 8-GPU config is more than 3x the 1-GPU one, so the pick moves to a larger, faster config.
    row_in = [["mistral-7b-v0.1", "lora", "NVIDIA-A100-SXM4-80GB", 1024, 16]]
    _, off = _run_batch(tmp_path, _guardless_config(), CANONICAL_HEADER, row_in, name="off")
    _, guarded = _run_batch(tmp_path, _base_config(), CANONICAL_HEADER, row_in, name="guard")

    thr_off = float(off[0]["predicted_throughput"])
    thr_guard = float(guarded[0]["predicted_throughput"])
    assert thr_guard > thr_off  # the guard removed the slow pick
    assert int(guarded[0]["recommended_total_gpus"]) > int(off[0]["recommended_total_gpus"])


# Derived metrics and the GPU power range.
def test_tokens_per_watt_equals_throughput_divided_by_per_gpu_power(tmp_path):
    # tokens_per_watt = throughput / per-GPU power. The base config picks a multi-GPU
    # config, where dividing by total power (power * N) would give a value N times lower.
    _, rows = _run_batch(
        tmp_path,
        _base_config(),
        CANONICAL_HEADER,
        [["mistral-7b-v0.1", "lora", "NVIDIA-A100-SXM4-80GB", 1024, 16]],
    )
    r = rows[0]
    thr = float(r["predicted_throughput"])
    power = float(r["predicted_power_watts"])
    tpw = float(r["tokens_per_watt"])
    assert int(r["recommended_total_gpus"]) > 1  # so the two divisions would differ
    assert tpw == pytest.approx(thr / power, rel=1e-9)


def test_per_gpu_power_within_a100_sxm4_envelope(tmp_path):
    # A100-SXM4-80GB datasheet: idle 75 W, TDP 400 W (coastline.sdk.library.hardware).
    # predicted_power_watts is per GPU, so it lies in [75, 400] W for any GPU count. The
    # base config picks a multi-GPU config, whose total power would exceed 400 W.
    _, rows = _run_batch(
        tmp_path,
        _base_config(),
        CANONICAL_HEADER,
        [["mistral-7b-v0.1", "lora", "NVIDIA-A100-SXM4-80GB", 1024, 16]],
    )
    power = float(rows[0]["predicted_power_watts"])
    assert 75.0 <= power <= 400.0


# Column mapping (a column that does not map makes the row feasible=False).
def test_custom_column_override_maps_headers(tmp_path):
    # By default each column is named after its WorkloadSpec field. input.columns adds other
    # spellings: here the_model maps to llm_model and the_gpu to gpu_model. If the mapping
    # were ignored, the row would come back feasible=False.
    cfg = _guardless_config()
    cfg["input"] = {"columns": {"the_model": "llm_model", "the_gpu": "gpu_model"}}
    _, rows = _run_batch(
        tmp_path,
        cfg,
        ["the_model", "fine_tuning_method", "the_gpu", "tokens_per_sample", "batch_size"],
        [["mistral-7b-v0.1", "lora", "NVIDIA-A100-SXM4-80GB", 1024, 16]],
    )
    assert len(rows) == 1
    assert rows[0]["feasible"] == "True"
    assert int(rows[0]["recommended_total_gpus"]) == 1  # mapping applied, workload valid


# Rationale text.
def test_rationale_states_min_gpu_goal_and_recommended_config(tmp_path):
    # For min_gpu the rationale opens with "<total> GPU[s] (...)" and gives the goal as
    # "the fewest GPUs that fit".
    _, rows = _run_batch(
        tmp_path,
        _base_config(),
        CANONICAL_HEADER,
        [["mistral-7b-v0.1", "lora", "NVIDIA-A100-SXM4-80GB", 1024, 16]],
    )
    r = rows[0]
    rationale = r["rationale"]
    total = r["recommended_total_gpus"]
    batch = r["recommended_batch_size"]
    assert rationale.startswith(f"{total} GPU")
    assert "picked for the fewest GPUs that fit" in rationale
    assert f"batch {batch}" in rationale


# The CLI batch mode gives the same result as the API.
def test_cli_entrypoint_matches_direct_api(tmp_path):
    # `coastline recommend-job --input/--output` wraps recommend_csv, so on the same inputs
    # it gives the same recommendation as calling recommend_csv directly.
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(_base_config()))
    inp = tmp_path / "in.csv"
    _write_csv(inp, CANONICAL_HEADER, [["mistral-7b-v0.1", "lora", "NVIDIA-A100-SXM4-80GB", 1024, 16]])

    api_out = tmp_path / "api_out.csv"
    recommend_csv(config, inp, api_out)
    api_row = next(csv.DictReader(open(api_out)))

    from coastline.cli import main

    cli_out = tmp_path / "cli_out.csv"
    main(["recommend-job", "--config", str(config), "--input", str(inp), "--output", str(cli_out)])
    cli_row = next(csv.DictReader(open(cli_out)))

    assert cli_row["feasible"] == "True"
    assert cli_row["recommended_total_gpus"] == api_row["recommended_total_gpus"]
    assert cli_row["predicted_throughput"] == api_row["predicted_throughput"]


# An unknown GPU in one row does not stop the batch.
def test_unknown_gpu_row_is_marked_feasible_false_not_crash(tmp_path):
    """A row with an unknown GPU comes back feasible=False with blank columns, and the valid
    row still gets a recommendation."""
    out, rows = _run_batch(
        tmp_path,
        _base_config(),
        CANONICAL_HEADER,
        [
            ["mistral-7b-v0.1", "lora", "NVIDIA-A100-SXM4-80GB", 1024, 16],
            ["mistral-7b-v0.1", "lora", "NOT-A-REAL-GPU-MODEL", 1024, 16],
        ],
    )
    assert out.exists(), "output file must be written even when a row has an unknown GPU"
    assert len(rows) == 2, "both input rows must appear in the output"
    # The valid row is unaffected by the bad one.
    assert rows[0]["feasible"] == "True"
    assert float(rows[0]["predicted_throughput"]) > 0
    # The unknown-GPU row has no recommendation.
    assert rows[1]["feasible"] == "False"
    assert rows[1]["recommended_total_gpus"] == ""
    assert rows[1]["predicted_throughput"] == ""
