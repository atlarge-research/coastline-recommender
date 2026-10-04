"""End-to-end tests of the fine-tuning trace pipeline.

A synthetic trace CSV goes through recommend_trace() into an enriched CSV with the recommended
layout and an estimated_duration column. Expected values come from arithmetic
(estimated_duration = total_tokens / throughput), a scaling law (duration is linear in runtime),
the metric definitions (tokens_per_watt = throughput / power, energy_wh = power x gpus x
runtime / 3600), and a comparison of the autoconf and rules paths. Only the Kavier predictor is
used (analytical and deterministic, no ML pickles).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

# The fixture job ran at 2500 tok/s for 3600 s, so it processed 2500 * 3600 = 9_000_000 tokens,
# whatever the configuration.
_FIXTURE_TPS = 2500.0
_FIXTURE_RUNTIME = 3600.0
_FIXTURE_TOTAL_TOKENS = 9_000_000.0  # = _FIXTURE_TPS * _FIXTURE_RUNTIME


def _trace_row(runtime: float = _FIXTURE_RUNTIME) -> dict:
    """One trace row in the trace schema; ``runtime`` drives total-token work."""
    return {
        # workload identity
        "metadata.model_name": "mistral-7b-v0.1",
        "metadata.method": "full",
        "resources.gpu_model": "NVIDIA-A100-SXM4-80GB",
        # layout
        "metadata.tokens_per_sample": 1024,
        "metadata.batch_size": 8,
        "resources.num_gpus_per_node": 1,
        "resources.num_nodes": 1,
        # measured performance; total tokens = tps * runtime
        "metadata.output.train_tokens_per_second": _FIXTURE_TPS,
        "metadata.train_runtime": runtime,
        # measured job duration
        "metadata.output.extrapolated_duration": 3600.0,
    }


def _write_trace(tmp_path, rows) -> str:
    path = tmp_path / "trace.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _rules_throughput() -> float:
    """The throughput Kavier gives the fixture workload on the rules path, from a separate
    recommend() call."""
    import coastline

    wl = {
        "llm_model": "mistral-7b-v0.1",
        "fine_tuning_method": "full",
        "gpu_model": "NVIDIA-A100-SXM4-80GB",
        "tokens_per_sample": 1024,
        "batch_size": 8,
    }
    top = coastline.recommend([wl], predictor="kavier", goal="min_gpu", max_gpus=1, top_k=1, feasibility="rules").iloc[
        0
    ]
    return float(top["throughput_tok_s"])


# recommend_trace: estimated_duration is total_tokens / recommended throughput


def test_recommend_trace_rules_estimate_equals_total_tokens_over_throughput(tmp_path):
    """recommend_trace writes estimated_duration = job_total_tokens / recommended throughput.

    total_tokens is 2500 * 3600 = 9_000_000 and the throughput comes from a separate recommend()
    call, so est * throughput gives back 9_000_000. Using tokens_per_sample for the total, or an
    inverted division, fails this.
    """
    from coastline.sdk.trace.recommend import recommend_trace

    in_csv = _write_trace(tmp_path, [_trace_row()])
    out_csv = str(tmp_path / "enriched.csv")

    df = recommend_trace(in_csv, out_csv, method="kavier", goal="min_gpu", feasibility="rules")

    # enriched CSV round-trips to disk with the single input row
    assert (tmp_path / "enriched.csv").exists(), "enriched CSV not written"
    on_disk = pd.read_csv(out_csv)
    assert len(on_disk) == len(df) == 1, "row count must match the 1-row input"

    # min_gpu: the rules check admits the job's total batch of 8 on 1 GPU.
    assert int(on_disk["resources.num_nodes"].iloc[0]) == 1
    assert int(on_disk["resources.num_gpus_per_node"].iloc[0]) == 1

    est = float(pd.to_numeric(df["metadata.estimated_duration_kavier"], errors="coerce").iloc[0])
    thr = _rules_throughput()
    # est = 9_000_000 / thr, so est * thr = 9_000_000 tokens
    assert est == pytest.approx(_FIXTURE_TOTAL_TOKENS / thr)
    assert est * thr == pytest.approx(_FIXTURE_TOTAL_TOKENS)


def test_recommend_trace_estimated_duration_scales_linearly_with_runtime(tmp_path):
    """Two workloads that differ only in train_runtime (3600 s and 7200 s) get the same config and
    throughput, so estimated_duration = tps * runtime / throughput doubles with the runtime."""
    from coastline.sdk.trace.recommend import recommend_trace

    in_csv = _write_trace(tmp_path, [_trace_row(runtime=3600.0), _trace_row(runtime=7200.0)])
    out_csv = str(tmp_path / "enriched_scale.csv")

    df = recommend_trace(in_csv, out_csv, method="kavier", feasibility="rules")
    est = pd.to_numeric(df["metadata.estimated_duration_kavier"], errors="coerce")

    # Row 1 has twice the runtime of row 0, so twice the tokens and twice the duration.
    assert est.iloc[1] == pytest.approx(2.0 * est.iloc[0])


def test_recommend_trace_autoconf_default_does_not_fall_back_to_rules(tmp_path):
    """The default feasibility='autoconf' runs the AutoConf OOM check, which rejects a full
    fine-tune of a 7B model on one 80GB A100 that the 'rules' path admits.

    min_gpu then moves on to 2 GPUs, which give a higher throughput and a shorter
    estimated_duration. Equal estimates would mean autoconf fell back to the rules path.
    """
    from coastline.sdk.trace.recommend import recommend_trace

    in_csv = _write_trace(tmp_path, [_trace_row()])

    # feasibility defaults to autoconf.
    df_auto = recommend_trace(in_csv, str(tmp_path / "auto.csv"), method="kavier", goal="min_gpu")
    df_rules = recommend_trace(
        in_csv, str(tmp_path / "rules.csv"), method="kavier", goal="min_gpu", feasibility="rules"
    )

    est_auto = float(pd.to_numeric(df_auto["metadata.estimated_duration_kavier"], errors="coerce").iloc[0])
    est_rules = float(pd.to_numeric(df_rules["metadata.estimated_duration_kavier"], errors="coerce").iloc[0])

    assert math.isfinite(est_auto) and est_auto > 0, "autoconf path produced no positive estimate"

    def gpus(df: pd.DataFrame) -> int:
        return int(df["resources.num_gpus_per_node"].iloc[0]) * int(df["resources.num_nodes"].iloc[0])

    # The OOM check rejects 1 GPU, so the autoconf pick has more GPUs and a shorter estimate.
    assert gpus(df_rules) == 1
    assert gpus(df_auto) > 1, "autoconf must not collapse onto the rules pick"
    assert est_auto < est_rules, "autoconf must not collapse onto the rules estimate"


# recommend(feasibility='rules') without AutoConf / without the fallback env


def test_recommend_feasibility_rules_works_without_autoconf(monkeypatch):
    """``coastline.recommend(..., feasibility='rules')`` runs without AutoConf and without
    COASTLINE_ALLOW_RULES_FALLBACK, and its derived metrics follow their definitions:
      tokens_per_watt = throughput / power_per_gpu
      energy_wh       = power_per_gpu * total_gpus * runtime_s / 3600
    """
    monkeypatch.delenv("COASTLINE_ALLOW_RULES_FALLBACK", raising=False)
    import coastline

    batch = [
        {
            "llm_model": "mistral-7b-v0.1",
            "fine_tuning_method": "lora",
            "gpu_model": "NVIDIA-A100-SXM4-80GB",
            "tokens_per_sample": 1024,
            "batch_size": 32,
        }
    ]
    df = coastline.recommend(batch, predictor="kavier", feasibility="rules", top_k=1)
    assert len(df) == 1
    # The rules path is accepted without AutoConf or the env var.
    assert bool(df.iloc[0]["feasible"]), f"row rejected: {df.iloc[0].get('error')}"

    row = df.iloc[0]
    thr = float(row["throughput_tok_s"])
    power = float(row["power_w"])
    gpus = int(row["total_gpus"])
    runtime = float(row["runtime_s"])

    assert thr > 0 and power > 0  # both divide below

    # tokens_per_watt is the throughput per watt of per-GPU power.
    assert float(row["tokens_per_watt"]) == pytest.approx(thr / power)
    # energy [Wh] = per-GPU power [W] * GPUs * runtime [s] / 3600
    assert float(row["energy_wh"]) == pytest.approx(power * gpus * runtime / 3600.0)
