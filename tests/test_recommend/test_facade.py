"""Tests for the importable ``coastline`` facade:

    import coastline
    rec = coastline(predictor="Kavier")   # or "tabpfn"
    results = rec(workload, total_gpus=[1, 2, 4, 8])

Covers name normalization, model id canonicalization, top-k truncation, preset and weight
steering, and input validation. Kavier's numbers are checked only through invariants.
"""

import math

import pytest

import coastline
from coastline import Coastline
from coastline.sdk.constants import EMPIRICAL_OOM_TOKEN_BUDGET
from coastline.sdk.models.recommendation import Recommendation

# Preset to (alpha = runtime weight, beta = energy weight), as in the thesis, copied from
# coastline.sdk.pipeline.selection.PRESET_WEIGHTS so that a change to that table fails here.
SPEC_PRESET_WEIGHTS = {"energy": (0.2, 0.8), "balanced": (0.5, 0.5), "performance": (0.8, 0.2)}


def _workload():
    return {
        "llm_model": "mistral-7b-v0.1",
        "fine_tuning_method": "lora",
        "gpu_model": "NVIDIA-A100-SXM4-80GB",
        "tokens_per_sample": 1024,
        "batch_size": 32,
    }


def test_predictor_is_normalized():
    # Normalization is strip + lower, so "Kavier" and "TabPFN" become their lowercase keys.
    assert Coastline(predictor="Kavier").predictor == "kavier"
    assert Coastline(predictor="TabPFN").predictor == "tabpfn"
    assert Coastline(predictor="tabpfn").predictor == "tabpfn"  # lowercase is kept


def test_module_is_callable_returns_configured_instance():
    # The module itself is callable (see _CallableModule) and returns a Coastline built with
    # the normalized predictor name.
    rec = coastline(predictor="Kavier")
    assert isinstance(rec, Coastline)
    assert rec.predictor == "kavier"


def test_recommend_truncates_to_stable_ranked_prefix():
    """top_k=2 returns the first two entries of the top_k=5 result, in the same order."""
    rec = Coastline("kavier")
    budget = [1, 2, 4, 8]
    top2 = rec.recommend(_workload(), total_gpus=budget, top_k=2)
    top5 = rec.recommend(_workload(), total_gpus=budget, top_k=5)

    assert len(top2) == 2 and len(top5) == 5

    def key(r):
        return (r.total_gpus, r.metadata["batch_size"])

    assert [key(r) for r in top2] == [key(r) for r in top5[:2]]

    for r in top5:
        assert isinstance(r, Recommendation)
        # Every pick comes from the requested GPU budget
        assert r.total_gpus in budget
        # and has a finite, positive throughput.
        assert r.predicted_throughput is not None and math.isfinite(r.predicted_throughput)
        assert r.predicted_throughput > 0


def test_total_throughput_increases_with_gpu_count():
    """With the workload and batch fixed, Kavier's total throughput rises with every added GPU."""
    rec = Coastline("kavier")
    # A large top_k returns every feasible config, and one batch size leaves only the GPU
    # count varying.
    recs = rec.recommend(_workload(), total_gpus=[1, 2, 4, 8], batch_sizes=[8], top_k=99)
    by_gpu = {r.total_gpus: r.predicted_throughput for r in recs}
    assert set(by_gpu) == {1, 2, 4, 8}, "every GPU count in the budget should be feasible at batch 8"
    ordered = [by_gpu[g] for g in (1, 2, 4, 8)]
    assert ordered == sorted(ordered), "aggregate throughput must be monotonic in GPU count"
    assert all(a < b for a, b in zip(ordered, ordered[1:])), "adding GPUs must strictly raise throughput"


def test_call_alias_reproduces_recommend_exactly():
    """``rec(...)`` returns the same ranked configs and throughputs as ``rec.recommend(...)``."""
    rec = Coastline("kavier")
    args = dict(total_gpus=[1, 2, 4], top_k=3)
    via_call = rec(_workload(), **args)
    via_method = rec.recommend(_workload(), **args)
    assert len(via_call) == len(via_method) == 3
    for a, b in zip(via_call, via_method):
        assert (a.total_gpus, a.metadata["batch_size"]) == (b.total_gpus, b.metadata["batch_size"])
        assert a.predicted_throughput == b.predicted_throughput


def test_dict_and_workloadspec_inputs_are_equivalent():
    """A dict and the equivalent WorkloadSpec give the same throughput on a fixed 2-GPU budget."""
    from coastline.sdk.models.workload import WorkloadSpec

    rec = Coastline("kavier")
    as_dict = rec(_workload(), total_gpus=[2], top_k=1)
    as_spec = rec(WorkloadSpec(**_workload()), total_gpus=[2], top_k=1)
    assert as_dict[0].total_gpus == as_spec[0].total_gpus == 2
    assert as_dict[0].predicted_throughput == as_spec[0].predicted_throughput


def test_huggingface_model_id_is_canonicalized_and_recommends(monkeypatch):
    """A HuggingFace id such as ``mistralai/Mistral-7B-v0.1`` becomes Kavier's key
    ``mistral-7b-v0.1`` and gives the same pick and throughput as the short id.

    Uses ``feasibility="rules"``, so no AutoConf install is needed.
    """
    monkeypatch.setenv("COASTLINE_ALLOW_RULES_FALLBACK", "1")
    rec = Coastline("kavier", feasibility="rules")
    hf = {**_workload(), "llm_model": "mistralai/Mistral-7B-v0.1"}
    out = rec(hf, total_gpus=[1, 2, 4], batch_sizes=[8], top_k=1)
    assert out, "HF model id should yield a recommendation"
    assert out[0].predicted_throughput and out[0].predicted_throughput > 0
    # The short id gives the same pick and throughput.
    short = rec({**_workload(), "llm_model": "mistral-7b-v0.1"}, total_gpus=[1, 2, 4], batch_sizes=[8], top_k=1)
    assert short and out[0].total_gpus == short[0].total_gpus
    assert out[0].predicted_throughput == short[0].predicted_throughput


def test_workloadspec_canonicalizes_huggingface_model_id():
    """WorkloadSpec drops the org prefix and lowercases the model id, and leaves a short id unchanged."""
    from coastline.sdk.models.workload import WorkloadSpec, canonical_model_name

    assert canonical_model_name("mistralai/Mistral-7B-v0.1") == "mistral-7b-v0.1"
    assert canonical_model_name("mistral-7b-v0.1") == "mistral-7b-v0.1"  # already short
    assert WorkloadSpec(**{**_workload(), "llm_model": "mistralai/Mistral-7B-v0.1"}).llm_model == "mistral-7b-v0.1"


def test_csv_path_reads_field_name_columns(tmp_path):
    # The CSV columns are the WorkloadSpec field names. Each cell lands in its field, and the
    # two numeric columns become int.
    from coastline.sdk.recommend.facade import _coerce_workload

    csv = tmp_path / "workload.csv"
    csv.write_text(
        "llm_model,fine_tuning_method,gpu_model,tokens_per_sample,batch_size\n"
        "mistral-7b-v0.1,lora,NVIDIA-A100-SXM4-80GB,1024,16\n"
    )
    wl = _coerce_workload(str(csv))
    assert wl.llm_model == "mistral-7b-v0.1"
    assert wl.gpu_model == "NVIDIA-A100-SXM4-80GB"
    assert wl.fine_tuning_method == "lora"
    assert wl.tokens_per_sample == 1024 and wl.batch_size == 16
    assert isinstance(wl.tokens_per_sample, int) and isinstance(wl.batch_size, int)


def test_default_context_max_nodes_uses_ceil_not_floor():
    # max_nodes = ceil(max_gpus / 8). Floor division would give a 12-GPU budget one node and
    # drop the 9 to 12 GPU configs.
    from coastline.sdk.models.workload import WorkloadSpec
    from coastline.sdk.recommend.facade import _default_context

    wl = WorkloadSpec(**_workload())
    assert _default_context(wl, 8).constraints.max_nodes == 1  # exact multiple
    assert _default_context(wl, 12).constraints.max_nodes == 2  # floor would give 1
    assert _default_context(wl, 16).constraints.max_nodes == 2  # exact multiple
    assert _default_context(wl, 20).constraints.max_nodes == 3  # floor would give 2


# Presets and alpha/beta: energy and performance pick different configs


def _top_total_gpus(recs):
    return recs[0].total_gpus


def test_energy_preset_favors_fewer_gpus_than_performance_preset():
    """The energy preset picks fewer GPUs than the performance preset.

    power_cost = per-GPU watts x GPU count, so an energy weight (beta) of 0.8 favours fewer GPUs.
    """
    rec = Coastline("kavier")
    budget = [1, 2, 4, 8]
    energy = rec.recommend(_workload(), total_gpus=budget, preset="energy", top_k=1)
    performance = rec.recommend(_workload(), total_gpus=budget, preset="performance", top_k=1)
    assert energy and performance
    assert _top_total_gpus(energy) <= _top_total_gpus(performance)
    assert _top_total_gpus(energy) != _top_total_gpus(performance)
    # Each pick records its preset's (alpha, beta), so the named preset was applied.
    assert (energy[0].metadata["alpha"], energy[0].metadata["beta"]) == SPEC_PRESET_WEIGHTS["energy"]
    assert (performance[0].metadata["alpha"], performance[0].metadata["beta"]) == SPEC_PRESET_WEIGHTS["performance"]


def test_explicit_alpha_beta_override_preset_and_reproduce_extremes():
    """Given alpha and beta override the preset: beta=0.8 picks like the energy preset,
    alpha=0.8 like the performance preset, and the preset is recorded as 'custom'."""
    rec = Coastline("kavier")
    budget = [1, 2, 4, 8]
    energy_like = rec.recommend(_workload(), total_gpus=budget, alpha=0.2, beta=0.8, top_k=1)
    perf_like = rec.recommend(_workload(), total_gpus=budget, alpha=0.8, beta=0.2, top_k=1)
    assert energy_like and perf_like
    assert _top_total_gpus(energy_like) <= _top_total_gpus(perf_like)
    assert _top_total_gpus(energy_like) != _top_total_gpus(perf_like)
    # 0.8 + 0.2 = 1, so normalization leaves the weights unchanged.
    assert (energy_like[0].metadata["alpha"], energy_like[0].metadata["beta"]) == (0.2, 0.8)
    assert energy_like[0].metadata["preset"] == "custom"
    assert (perf_like[0].metadata["alpha"], perf_like[0].metadata["beta"]) == (0.8, 0.2)


# Input validation: an empty CSV raises ValueError, a wrong type TypeError


def test_empty_csv_raises_value_error(tmp_path):
    csv = tmp_path / "empty.csv"
    # A header with no data rows.
    csv.write_text("llm_model,fine_tuning_method,gpu_model,tokens_per_sample,batch_size\n")
    rec = Coastline("kavier")
    with pytest.raises(ValueError):
        rec.recommend(str(csv))


def test_unsupported_workload_type_raises_type_error():
    rec = Coastline("kavier")
    # An int is neither a WorkloadSpec, a dict nor a CSV path.
    with pytest.raises(TypeError):
        rec.recommend(12345)


def test_max_gpus_zero_raises_clear_value_error():
    """max_gpus=0 raises a ValueError that says max_gpus must be >= 1."""
    rec = Coastline("kavier")
    with pytest.raises(ValueError, match="max_gpus must be >= 1"):
        rec.recommend(_workload(), max_gpus=0)


def test_max_gpus_negative_raises_clear_value_error():
    """A negative max_gpus raises the same error."""
    rec = Coastline("kavier")
    with pytest.raises(ValueError, match="max_gpus must be >= 1"):
        rec.recommend(_workload(), max_gpus=-1)


# feasibility='rules' constructor path (under COASTLINE_ALLOW_RULES_FALLBACK=1)


def test_rules_feasibility_admits_all_per_device_configs(monkeypatch):
    """``Coastline(feasibility="rules")`` admits every GPU count in the budget, 3 included.

    batch_size is per device, so it need not divide the GPU count.
    """
    monkeypatch.setenv("COASTLINE_ALLOW_RULES_FALLBACK", "1")
    rec = Coastline("kavier", feasibility="rules")
    assert rec.feasibility == "rules"
    out = rec.recommend(_workload(), total_gpus=[1, 2, 3, 4, 8], batch_sizes=[8], top_k=99)
    assert out, "rules feasibility should still yield recommendations"
    assert all(isinstance(r, Recommendation) for r in out)
    admitted = {r.total_gpus for r in out}
    assert admitted == {1, 2, 3, 4, 8}


def test_empirical_oom_guard_vetoes_over_budget_per_device_loads(monkeypatch):
    """``empirical_oom_guard=True`` adds the per-device token limit to the rules checker.

    A ``batch_size x tokens_per_sample`` above EMPIRICAL_OOM_TOKEN_BUDGET is rejected at every GPU
    count and accepted with the guard off; a load below the limit passes with the guard on.
    """
    monkeypatch.setenv("COASTLINE_ALLOW_RULES_FALLBACK", "1")
    over, under = 32 * 2048, 32 * 1024
    assert over > EMPIRICAL_OOM_TOKEN_BUDGET > under, "test workloads must straddle the budget"
    over_wl = {**_workload(), "tokens_per_sample": 2048, "batch_size": 32}
    under_wl = {**_workload(), "tokens_per_sample": 1024, "batch_size": 32}
    budgets = [1, 2, 4, 8]

    unguarded = Coastline("kavier", feasibility="rules")
    assert unguarded.empirical_oom_guard is False
    assert unguarded.recommend(over_wl, total_gpus=budgets, batch_sizes=[32], top_k=99)

    guarded = Coastline("kavier", feasibility="rules", empirical_oom_guard=True)
    assert guarded.empirical_oom_guard is True
    with pytest.raises(RuntimeError, match="no feasible"):
        guarded.recommend(over_wl, total_gpus=budgets, batch_sizes=[32], top_k=99)
    out = guarded.recommend(under_wl, total_gpus=budgets, batch_sizes=[32], top_k=99)
    assert {r.total_gpus for r in out} == set(budgets)
