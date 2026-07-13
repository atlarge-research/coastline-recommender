"""End-to-end tests for the CSV -> CSV batch recommender (Kavier path, hermetic).

Every assertion is tied to an independent oracle:
  * min_gpu contract — ranks feasible configs by (total_gpus asc, throughput desc),
    so with the runtime guard OFF it must pick the fewest GPUs in the grid.
  * divisibility feasibility rule — a config is feasible iff batch_size % total_gpus == 0.
  * derived metrics — tokens_per_watt = predicted_throughput / per-GPU power.
  * hardware envelope — per-GPU power lies in [idle, TDP] for the recommended GPU.
The Kavier engine's exact throughput/power magnitudes are treated as black boxes; we
assert invariants, scaling laws and cross-checks, never a pinned engine output.
"""

import csv

import pytest
import yaml

from coastline.sdk.recommend.batch_csv import recommend_csv

CANONICAL_HEADER = ["model_name", "method", "gpu_model", "tokens_per_sample", "batch_size"]


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _base_config():
    """min_gpu with the runtime guard armed (max_slowdown=3): the guard drops feasible
    configs slower than 1/3 of the fastest, so min_gpu cannot settle on the slow 1-GPU
    layout — it is forced onto a faster, larger config."""
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
    """Same grid but NO max_slowdown, so the runtime guard is off. min_gpu then simply
    picks the fewest feasible GPUs — a hand-derivable outcome."""
    cfg = _base_config()
    del cfg["strategy"]["max_slowdown"]
    return cfg


def _run_batch(tmp_path, config, header, rows, *, name="in"):
    """Write config + input CSV, run the batch recommender, return the output rows."""
    cfg_path = tmp_path / f"{name}_config.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    inp = tmp_path / f"{name}.csv"
    _write_csv(inp, header, rows)
    out = tmp_path / f"{name}_out.csv"
    recommend_csv(cfg_path, inp, out)
    return out, list(csv.DictReader(open(out)))


# --------------------------------------------------------------------------- #
# Structural contract: one recommendation row per input row, input echoed.
# --------------------------------------------------------------------------- #
def test_one_output_row_per_input_row_with_input_echoed(tmp_path):
    rows_in = [
        ["mistral-7b-v0.1", "lora", "NVIDIA-A100-SXM4-80GB", 1024, 16],
        ["mistral-7b-v0.1", "lora", "NVIDIA-A100-SXM4-80GB", 2048, 8],
    ]
    _, rows = _run_batch(tmp_path, _base_config(), CANONICAL_HEADER, rows_in)

    # Cardinality contract: exactly one output row per input row (no drop, no fan-out).
    assert len(rows) == 2
    # Input columns are echoed back verbatim, in order, alongside the recommendation.
    assert [r["model_name"] for r in rows] == ["mistral-7b-v0.1", "mistral-7b-v0.1"]
    assert [r["tokens_per_sample"] for r in rows] == ["1024", "2048"]
    # Node layout is internally consistent: total = gpus_per_node * number_of_nodes.
    for r in rows:
        assert int(r["recommended_total_gpus"]) == int(r["recommended_gpus_per_node"]) * int(
            r["recommended_number_of_nodes"]
        )


# --------------------------------------------------------------------------- #
# min_gpu selection contract (hand-derived).
# --------------------------------------------------------------------------- #
def test_min_gpu_selects_single_gpu_when_runtime_guard_disabled(tmp_path):
    # Grid total_gpus = [1,2,4,8], batch_size 16. Divisibility rule: feasible iff
    # 16 % total_gpus == 0 -> ALL of {1,2,4,8} feasible. With the runtime guard off,
    # min_gpu ranks by (total_gpus asc) so it MUST pick the smallest: 1 GPU.
    # A 1-GPU layout on <=8 GPUs/node => gpus_per_node=1, number_of_nodes=1.
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
    # Tie-break: among equal-GPU configs min_gpu sorts by -throughput; at a fixed GPU
    # count Kavier throughput rises with batch size, so batch 16 (the larger grid batch)
    # beats batch 8 -> recommended batch is 16.
    assert int(r["recommended_batch_size"]) == 16


def test_runtime_guard_forces_faster_config_than_min_gpu_alone(tmp_path):
    # With the guard off, min_gpu picks the fewest GPUs = the globally SLOWEST feasible
    # config (1 GPU). Arming max_slowdown=3 removes every config slower than 1/3 of the
    # fastest; that veto can only strip slow configs, so the recommended throughput can
    # only rise. In this A100/mistral grid the 8-GPU config is >3x the 1-GPU one, so the
    # 1-GPU layout is vetoed and min_gpu is pushed onto a strictly larger, faster config.
    row_in = [["mistral-7b-v0.1", "lora", "NVIDIA-A100-SXM4-80GB", 1024, 16]]
    _, off = _run_batch(tmp_path, _guardless_config(), CANONICAL_HEADER, row_in, name="off")
    _, guarded = _run_batch(tmp_path, _base_config(), CANONICAL_HEADER, row_in, name="guard")

    thr_off = float(off[0]["predicted_throughput"])
    thr_guard = float(guarded[0]["predicted_throughput"])
    assert thr_guard > thr_off  # guard removed the slow pick -> faster recommendation
    assert int(guarded[0]["recommended_total_gpus"]) > int(off[0]["recommended_total_gpus"])


# --------------------------------------------------------------------------- #
# Derived-metric and hardware-envelope cross-checks.
# --------------------------------------------------------------------------- #
def test_tokens_per_watt_equals_throughput_divided_by_per_gpu_power(tmp_path):
    # tokens_per_watt is defined as throughput / per-GPU power (NOT per-total-GPU).
    # Cross-check the emitted column against the other two emitted columns. Under the
    # base config the pick is a MULTI-GPU config, so a per-total bug (thr/(power*N))
    # would land ~N-fold lower and this cross-check would catch it.
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
    assert int(r["recommended_total_gpus"]) > 1  # ensure the per-total bug would diverge
    assert tpw == pytest.approx(thr / power, rel=1e-9)


def test_per_gpu_power_within_a100_sxm4_envelope(tmp_path):
    # NVIDIA-A100-SXM4-80GB datasheet: idle 75 W, TDP 400 W (src/.../library/hardware.py).
    # predicted_power_watts is PER GPU, so it must fall in [75, 400] regardless of the
    # recommended GPU count. Under the base config a multi-GPU config is picked; if the
    # column reported *total* power it would exceed 400 W (>=2*215) and fail here.
    _, rows = _run_batch(
        tmp_path,
        _base_config(),
        CANONICAL_HEADER,
        [["mistral-7b-v0.1", "lora", "NVIDIA-A100-SXM4-80GB", 1024, 16]],
    )
    power = float(rows[0]["predicted_power_watts"])
    assert 75.0 <= power <= 400.0


# --------------------------------------------------------------------------- #
# Column-aliasing contracts (a mapping bug => WorkloadSpec invalid => feasible=False).
# --------------------------------------------------------------------------- #
def test_alias_headers_map_to_workload_fields(tmp_path):
    # llm_model/peft/gpu/seq_len/batch are canonical aliases for
    # llm_model/fine_tuning_method/gpu_model/tokens_per_sample/batch_size. If ANY alias
    # were unmapped the WorkloadSpec would be missing a required field -> feasible=False.
    # With the guard off the same workload the canonical headers produce (1 GPU) must
    # come back, proving every alias resolved.
    _, rows = _run_batch(
        tmp_path,
        _guardless_config(),
        ["llm_model", "peft", "gpu", "seq_len", "batch"],
        [["mistral-7b-v0.1", "lora", "NVIDIA-A100-SXM4-80GB", 1024, 16]],
    )
    assert len(rows) == 1
    assert rows[0]["feasible"] == "True"
    assert int(rows[0]["recommended_total_gpus"]) == 1  # matches canonical-header run


def test_custom_column_override_maps_headers(tmp_path):
    # input.columns overlays extra spellings onto the default alias map; here
    # the_model->llm_model and the_gpu->gpu_model, while method/tokens_per_sample/
    # batch_size keep their canonical spellings. If the override were ignored, the_model/
    # the_gpu would not resolve and the row would be feasible=False.
    cfg = _guardless_config()
    cfg["input"] = {"columns": {"the_model": "llm_model", "the_gpu": "gpu_model"}}
    _, rows = _run_batch(
        tmp_path,
        cfg,
        ["the_model", "method", "the_gpu", "tokens_per_sample", "batch_size"],
        [["mistral-7b-v0.1", "lora", "NVIDIA-A100-SXM4-80GB", 1024, 16]],
    )
    assert len(rows) == 1
    assert rows[0]["feasible"] == "True"
    assert int(rows[0]["recommended_total_gpus"]) == 1  # override honored -> valid workload


# --------------------------------------------------------------------------- #
# Infeasible rows are marked, not dropped.
# --------------------------------------------------------------------------- #
def test_infeasible_row_marked_false_and_retained_with_blanks(tmp_path):
    # Divisibility rule: feasible iff batch_size % total_gpus == 0. Grid batch=3 over
    # total_gpus {2,4,8}: 3 % 2, 3 % 4, 3 % 8 are all != 0 -> NO feasible config, so the
    # pipeline raises and the row is caught. It must still appear (marked, not dropped),
    # with feasible=False and every recommendation column blank.
    cfg = _base_config()
    cfg["grid"]["batch_sizes"] = [3]
    cfg["grid"]["total_gpus"] = [2, 4, 8]
    _, rows = _run_batch(
        tmp_path,
        cfg,
        CANONICAL_HEADER,
        [["mistral-7b-v0.1", "lora", "NVIDIA-A100-SXM4-80GB", 1024, 8]],
    )
    assert len(rows) == 1  # retained, not dropped
    r = rows[0]
    assert r["model_name"] == "mistral-7b-v0.1"  # input still echoed
    assert r["feasible"] == "False"
    assert r["recommended_total_gpus"] == ""
    assert r["predicted_throughput"] == ""
    assert r["tokens_per_watt"] == ""


# --------------------------------------------------------------------------- #
# Rationale text (hand-derived from recommendation_rationale + _GOAL_RATIONALE).
# --------------------------------------------------------------------------- #
def test_rationale_states_min_gpu_goal_and_recommended_config(tmp_path):
    # For strategy min_gpu, recommendation_rationale uses _GOAL_RATIONALE["min_gpu"] =
    # "the fewest GPUs that fit" and opens with "<total> GPU[s] (...)". So the line must
    # start with the recommended GPU count, name the recommended batch, and cite the goal.
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


# --------------------------------------------------------------------------- #
# CLI entry point wires config/input/output through to the same result.
# --------------------------------------------------------------------------- #
def test_cli_entrypoint_matches_direct_api(tmp_path):
    # The `coastline recommend-job --input/--output` batch mode is a thin wrapper over
    # recommend_csv; running it must yield the SAME recommendation the direct API produces
    # for identical inputs. Any arg-wiring bug (swapped input/output, dropped config) would
    # diverge from this oracle.
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


# --------------------------------------------------------------------------- #
# Regression: an unknown GPU in one row must not abort the whole batch.
# --------------------------------------------------------------------------- #
def test_unknown_gpu_row_is_marked_feasible_false_not_crash(tmp_path):
    """A CSV row with an unknown GPU must produce feasible=False in the output,
    NOT crash the whole batch (regression for UnsupportedGPUError escaping the
    per-row try/except when SystemContext.for_gpus was called outside the try).

    Oracle: the two rows are independent — the valid row must still yield a positive
    throughput while the bad row is isolated to feasible=False with blank columns.
    """
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
    # Valid row: feasible with a real prediction, unaffected by the bad row's failure.
    assert rows[0]["feasible"] == "True"
    assert float(rows[0]["predicted_throughput"]) > 0
    # Unknown-GPU row: isolated to feasible=False, no recommendation leaked.
    assert rows[1]["feasible"] == "False"
    assert rows[1]["recommended_total_gpus"] == ""
    assert rows[1]["predicted_throughput"] == ""
