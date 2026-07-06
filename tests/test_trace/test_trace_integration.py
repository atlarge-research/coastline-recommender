"""End-to-end integration test for the fine-tuning trace pipeline.

Covers the full flow:
  1. synthetic trace CSV matching the trace schema
  2. enrich_trace() -> enriched CSV with layout + estimated_duration column
  3. plot_trace_performance() -> PNG + impact stats (jobs / pct_faster / median_speedup)

Every assertion carries an INDEPENDENT oracle: hand-derived arithmetic
(estimated_duration = total_tokens / throughput, impact stats), a scaling law
(duration scales linearly with runtime), a derived-metric cross-check
(tokens_per_watt = throughput / power, energy_wh = power x gpus x runtime / 3600),
or a cross-method contract check (autoconf must not silently fall back to rules).
Uses only the kavier predictor (analytical physics, deterministic, no ML pickles).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

# Ground-truth work in the fixture: the job ran at 2500 tok/s for 3600 s.
# By hand, tokens processed = 2500 * 3600 = 9_000_000 (config-independent "work").
_FIXTURE_TPS = 2500.0
_FIXTURE_RUNTIME = 3600.0
_FIXTURE_TOTAL_TOKENS = 9_000_000.0  # = _FIXTURE_TPS * _FIXTURE_RUNTIME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        # measured performance -> total tokens = tps * runtime
        "metadata.output.train_tokens_per_second": _FIXTURE_TPS,
        "metadata.train_runtime": runtime,
        # ground-truth job duration (used by trace_plot for the impact scatter)
        "metadata.output.extrapolated_duration": 3600.0,
    }


def _write_trace(tmp_path, rows) -> str:
    path = tmp_path / "trace.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _rules_throughput() -> float:
    """Throughput kavier assigns to the fixture workload on the divisibility-only
    (rules) path — fetched via a SEPARATE recommend() call so it is an independent
    oracle for enrich_trace's estimated_duration = total_tokens / throughput."""
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


# ---------------------------------------------------------------------------
# 1. enrich_trace — estimated_duration is total_tokens / recommended throughput
# ---------------------------------------------------------------------------


def test_enrich_trace_rules_estimate_equals_total_tokens_over_throughput(tmp_path):
    """enrich_trace writes estimated_duration = job_total_tokens / recommended
    throughput. Oracle: total_tokens = 2500*3600 = 9_000_000 (hand-derived), and
    the throughput is obtained from an INDEPENDENT recommend() call, so the product
    est * throughput must recover the 9_000_000 tokens of work.

    This falsifies a total-token bug (e.g. using tokens_per_sample instead of
    tps*runtime) or an inverted division.
    """
    from coastline.sdk.trace.enrich import enrich_trace

    in_csv = _write_trace(tmp_path, [_trace_row()])
    out_csv = str(tmp_path / "enriched.csv")

    df = enrich_trace(in_csv, out_csv, method="kavier", feasibility="rules")

    # enriched CSV round-trips to disk with the single input row
    assert (tmp_path / "enriched.csv").exists(), "enriched CSV not written"
    on_disk = pd.read_csv(out_csv)
    assert len(on_disk) == len(df) == 1, "row count must match the 1-row input"

    # min_gpu with max_gpus = num_gpus_per_node * num_nodes = 1*1 = 1 leaves exactly
    # one feasible layout: a single GPU on a single node. Oracle: nodes==1, gpn==1.
    assert int(on_disk["resources.num_nodes"].iloc[0]) == 1
    assert int(on_disk["resources.num_gpus_per_node"].iloc[0]) == 1

    est = float(pd.to_numeric(df["metadata.estimated_duration_kavier"], errors="coerce").iloc[0])
    thr = _rules_throughput()
    # est = 9_000_000 / thr  <=>  est * thr = 9_000_000 (total tokens of work)
    assert est == pytest.approx(_FIXTURE_TOTAL_TOKENS / thr)
    assert est * thr == pytest.approx(_FIXTURE_TOTAL_TOKENS)


def test_enrich_trace_estimated_duration_scales_linearly_with_runtime(tmp_path):
    """Two identical workloads differing only in train_runtime (3600 s vs 7200 s)
    get the SAME recommended config -> the SAME throughput, so estimated_duration
    (= tps*runtime / throughput) must scale linearly with runtime: doubling the
    runtime doubles the estimate. Independent of the throughput magnitude.

    Falsifies any bug where the estimate ignores runtime / total tokens.
    """
    from coastline.sdk.trace.enrich import enrich_trace

    in_csv = _write_trace(tmp_path, [_trace_row(runtime=3600.0), _trace_row(runtime=7200.0)])
    out_csv = str(tmp_path / "enriched_scale.csv")

    df = enrich_trace(in_csv, out_csv, method="kavier", feasibility="rules")
    est = pd.to_numeric(df["metadata.estimated_duration_kavier"], errors="coerce")

    # runtime doubled from row 0 to row 1 -> tokens doubled -> duration doubled.
    assert est.iloc[1] == pytest.approx(2.0 * est.iloc[0])


def test_enrich_trace_autoconf_default_does_not_fall_back_to_rules(tmp_path):
    """The default feasibility='autoconf' runs the real OOM check, which caps a
    full fine-tune of a 7B model on a single 80GB A100 to a SMALLER batch than the
    divisibility-only ('rules') path admits. Different batch -> different throughput
    -> a DIFFERENT (here: larger, more conservative) estimated_duration.

    Oracle/contract: est_autoconf must be positive AND must differ from est_rules.
    Equality would mean autoconf silently degraded to the rules path — the exact
    regression the docstring warns about.
    """
    from coastline.sdk.trace.enrich import enrich_trace

    in_csv = _write_trace(tmp_path, [_trace_row()])

    df_auto = enrich_trace(in_csv, str(tmp_path / "auto.csv"), method="kavier")  # default = autoconf
    df_rules = enrich_trace(in_csv, str(tmp_path / "rules.csv"), method="kavier", feasibility="rules")

    est_auto = float(pd.to_numeric(df_auto["metadata.estimated_duration_kavier"], errors="coerce").iloc[0])
    est_rules = float(pd.to_numeric(df_rules["metadata.estimated_duration_kavier"], errors="coerce").iloc[0])

    assert math.isfinite(est_auto) and est_auto > 0, "autoconf path produced no positive estimate"
    # OOM-aware batch cap -> lower throughput -> longer estimated duration than rules.
    assert est_auto > est_rules, "autoconf must not collapse onto the rules estimate"


# ---------------------------------------------------------------------------
# 2. recommend(feasibility='rules') without AutoConf / without the fallback env
# ---------------------------------------------------------------------------


def test_recommend_feasibility_rules_works_without_autoconf(monkeypatch):
    """``coastline.recommend(..., feasibility='rules')`` succeeds with no AutoConf
    install and no COASTLINE_ALLOW_RULES_FALLBACK — the divisibility-only path must
    not refuse. Beyond the no-refusal contract, cross-check the derived metrics
    against their definitions (a form independent of the engine's internals):
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
    # Contract: the rules path is accepted (never refused) with no autoconf/env.
    assert bool(df.iloc[0]["feasible"]), f"row rejected: {df.iloc[0].get('error')}"

    row = df.iloc[0]
    thr = float(row["throughput_tok_s"])
    power = float(row["power_w"])
    gpus = int(row["total_gpus"])
    runtime = float(row["runtime_s"])

    assert thr > 0 and power > 0  # guards for the ratios below

    # tokens_per_watt is throughput per watt-per-GPU (different form than storing it).
    assert float(row["tokens_per_watt"]) == pytest.approx(thr / power)
    # energy [Wh] = per-GPU power * #GPUs * runtime[s] / 3600 s-per-h (catches a /1000 unit bug).
    assert float(row["energy_wh"]) == pytest.approx(power * gpus * runtime / 3600.0)


# ---------------------------------------------------------------------------
# 3. plot_trace_performance — impact stats hand-derived from known est/actual
# ---------------------------------------------------------------------------


def test_plot_trace_impact_stats_are_hand_derived(tmp_path):
    """plot_trace_performance masks rows to (est>0 & actual>0), then reports
    jobs / pct_faster (est<actual) / median_speedup (actual/est). Build an enriched
    CSV with KNOWN columns and hand-compute every stat.

    Kept rows (est, actual): (100,200)->speedup 2, (400,200)->0.5, (100,300)->3.
    Masked: (0,500) est not >0; (NaN,100) est missing.
      jobs            = 3
      faster (est<act)= rows 1 & 3 -> 2 of 3 -> pct_faster = 200/3 = 66.667%
      speedups sorted = [0.5, 2.0, 3.0] -> median = 2.0
    """
    pytest.importorskip("matplotlib", reason="matplotlib not installed; pip install coastline[plot]")

    from coastline.sdk.trace.plot import plot_trace_performance

    enriched = tmp_path / "enriched.csv"
    pd.DataFrame(
        {
            "metadata.estimated_duration_kavier": [100.0, 400.0, 100.0, 0.0, float("nan")],
            "metadata.output.extrapolated_duration": [200.0, 200.0, 300.0, 500.0, 100.0],
        }
    ).to_csv(enriched, index=False)

    out_png = tmp_path / "perf.png"
    result = plot_trace_performance(str(enriched), str(out_png), method="kavier")

    # PNG written (guard for the plotting side effect)
    assert out_png.exists() and out_png.stat().st_size > 0, "PNG not written"

    assert result["jobs"] == 3  # two of five rows masked out
    assert result["pct_faster"] == pytest.approx(200.0 / 3.0)  # 2 faster of 3 -> 66.667%
    assert result["median_speedup"] == pytest.approx(2.0)  # median of [0.5, 2.0, 3.0]
