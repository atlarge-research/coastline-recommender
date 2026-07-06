"""Tests for the importable `coastline` facade (the supervisor's sketch):

    import coastline
    rec = coastline(throughput_estim="Kavier")   # or "tabpfn"
    results = rec(workload, total_gpus=[1, 2, 4, 8])

Kavier throughput/power is an analytical black box: these tests assert the facade
CONTRACT (normalization, canonicalization, top-k truncation, ranking-prefix
stability, preset/weight steering, input validation) with independent oracles, and
only INVARIANTS (monotonicity, budget membership, equivalence of input forms) over
the engine's numbers — never a pinned magic throughput value.
"""

import math

import pytest

import coastline
from coastline import Coastline
from coastline.sdk.models.recommendation import Recommendation

# Preset -> (alpha=power weight, beta=throughput weight), the canonical spec the
# selection layer indexes on (coastline.sdk.pipeline.selection.PRESET_WEIGHTS).
# Re-stated here as an INDEPENDENT reference so a silent edit to the weights table
# is caught rather than mirrored.
SPEC_PRESET_WEIGHTS = {"energy": (0.8, 0.2), "balanced": (0.5, 0.5), "performance": (0.2, 0.8)}


def _workload():
    return {
        "llm_model": "mistral-7b-v0.1",
        "fine_tuning_method": "lora",
        "gpu_model": "NVIDIA-A100-SXM4-80GB",
        "tokens_per_sample": 1024,
        "batch_size": 32,
    }


def test_throughput_estim_is_normalized():
    # Oracle: normalization is pure case-folding (strip+lower), independent of any
    # lookup table. "Kavier"/"TabPFN" differ from their keys only in letter case, so
    # a correct normalizer must return the all-lowercase spelling; a normalizer that
    # did nothing (or upper-cased) would leave "Kavier" != "kavier" and fail.
    assert Coastline(throughput_estim="Kavier").throughput_estim == "kavier"
    assert Coastline(throughput_estim="TabPFN").throughput_estim == "tabpfn"
    assert Coastline(throughput_estim="tabpfn").throughput_estim == "tabpfn"  # idempotent on lowercase


def test_module_is_callable_returns_configured_instance():
    # Contract: the module object itself is callable (PEP 562 _CallableModule) and
    # forwards throughput_estim into a real Coastline. Oracle: the returned object is
    # a Coastline whose estimator is the normalized spelling of what we passed — a
    # plain module (non-callable) would raise TypeError here.
    rec = coastline(throughput_estim="Kavier")
    assert isinstance(rec, Coastline)
    assert rec.throughput_estim == "kavier"


def test_recommend_truncates_to_stable_ranked_prefix():
    """top_k must return the first k of the FULL ranking, not an arbitrary/pre-sort
    subset. Oracle: the top-2 result must equal the first two entries of the top-5
    result (same configs, same order) — a stable ranking followed by an
    ``ordered[:k]`` slice. A bug that truncated before sorting, or reshuffled per
    call, would break this prefix identity."""
    rec = Coastline("kavier")
    budget = [1, 2, 4, 8]
    top2 = rec.recommend(_workload(), total_gpus=budget, top_k=2)
    top5 = rec.recommend(_workload(), total_gpus=budget, top_k=5)

    assert len(top2) == 2 and len(top5) == 5  # top_k respected exactly

    def key(r):
        return (r.total_gpus, r.metadata["batch_size"])

    assert [key(r) for r in top2] == [key(r) for r in top5[:2]]  # prefix identity

    for r in top5:
        assert isinstance(r, Recommendation)
        # Every pick is a real config drawn from the requested GPU budget...
        assert r.total_gpus in budget
        # ...and Kavier throughput for a SUPPORTED workload is finite and positive.
        assert r.predicted_throughput is not None and math.isfinite(r.predicted_throughput)
        assert r.predicted_throughput > 0


def test_total_throughput_increases_with_gpu_count():
    """Scaling invariant for the Kavier engine: with batch and workload fixed, total
    (aggregate) throughput must strictly increase as GPUs are added — 2 GPUs out-
    produce 1, 4 out-produce 2, etc. No magic value pinned; only the ordering. A
    predictor that ignored GPU count (constant output) would flatten this and fail."""
    rec = Coastline("kavier")
    # top_k large so every feasible config is returned; fixed batch so only the GPU
    # axis varies. batch_sizes=[8] divides all of 1/2/4/8 -> all feasible.
    recs = rec.recommend(_workload(), total_gpus=[1, 2, 4, 8], batch_sizes=[8], top_k=99)
    by_gpu = {r.total_gpus: r.predicted_throughput for r in recs}
    assert set(by_gpu) == {1, 2, 4, 8}, "every GPU count in the budget should be feasible at batch 8"
    ordered = [by_gpu[g] for g in (1, 2, 4, 8)]
    assert ordered == sorted(ordered), "aggregate throughput must be monotonic in GPU count"
    assert all(a < b for a, b in zip(ordered, ordered[1:])), "adding GPUs must strictly raise throughput"


def test_call_alias_reproduces_recommend_exactly():
    """``rec(...)`` is documented as an alias of ``rec.recommend(...)``. Oracle: with
    identical args both paths must return the SAME ranked configs and the SAME engine
    numbers (deterministic), proving __call__ dispatches to recommend and does not
    re-parameterize anything."""
    rec = Coastline("kavier")
    args = dict(total_gpus=[1, 2, 4], top_k=3)
    via_call = rec(_workload(), **args)
    via_method = rec.recommend(_workload(), **args)
    assert len(via_call) == len(via_method) == 3
    for a, b in zip(via_call, via_method):
        assert (a.total_gpus, a.metadata["batch_size"]) == (b.total_gpus, b.metadata["batch_size"])
        assert a.predicted_throughput == b.predicted_throughput  # identical engine output


def test_dict_and_workloadspec_inputs_are_equivalent():
    """A dict workload and the equivalent WorkloadSpec must be coerced to the same
    spec and therefore yield the IDENTICAL prediction. Oracle: same top config AND
    the exact same throughput number (not merely 'both positive'). Budget [2] forces
    total_gpus=2, so the load-bearing check is the throughput equality — a divergence
    in the dict vs spec code path would surface as different floats."""
    from coastline.sdk.models.workload import WorkloadSpec

    rec = Coastline("kavier")
    as_dict = rec(_workload(), total_gpus=[2], top_k=1)
    as_spec = rec(WorkloadSpec(**_workload()), total_gpus=[2], top_k=1)
    assert as_dict[0].total_gpus == as_spec[0].total_gpus == 2
    assert as_dict[0].predicted_throughput == as_spec[0].predicted_throughput


def test_huggingface_model_id_is_canonicalized_and_recommends(monkeypatch):
    """Regression (BLOCKER 1): a real HuggingFace id like ``mistralai/Mistral-7B-v0.1``
    must be canonicalized to the short key Kavier indexes on (``mistral-7b-v0.1``) and
    return a recommendation — NOT raise ``RuntimeError: no feasible candidates``. Pinned
    to ``feasibility="rules"`` so it is deterministic and needs no AutoConf install.

    Oracle: canonicalization means the HF id and the already-short id are the SAME
    workload, so they must produce the identical top pick AND identical throughput.
    """
    monkeypatch.setenv("COASTLINE_ALLOW_RULES_FALLBACK", "1")
    rec = Coastline("kavier", feasibility="rules")
    hf = {**_workload(), "llm_model": "mistralai/Mistral-7B-v0.1"}
    out = rec(hf, total_gpus=[1, 2, 4], batch_sizes=[8], top_k=1)
    assert out, "HF model id should yield a recommendation, not RuntimeError"
    assert out[0].predicted_throughput and out[0].predicted_throughput > 0
    # The canonical short key must produce the SAME pick and SAME number as the HF id.
    short = rec({**_workload(), "llm_model": "mistral-7b-v0.1"}, total_gpus=[1, 2, 4], batch_sizes=[8], top_k=1)
    assert short and out[0].total_gpus == short[0].total_gpus
    assert out[0].predicted_throughput == short[0].predicted_throughput


def test_workloadspec_canonicalizes_huggingface_model_id():
    """The WorkloadSpec field validator drops the org prefix + lowercases, and is
    idempotent on the already-short form. Oracle: the transform is
    ``split('/')[-1].lower()`` applied by hand to a known id."""
    from coastline.sdk.models.workload import WorkloadSpec, canonical_model_name

    # By hand: "mistralai/Mistral-7B-v0.1" -> drop "mistralai/" -> lowercase -> "mistral-7b-v0.1".
    assert canonical_model_name("mistralai/Mistral-7B-v0.1") == "mistral-7b-v0.1"
    assert canonical_model_name("mistral-7b-v0.1") == "mistral-7b-v0.1"  # idempotent
    assert WorkloadSpec(**{**_workload(), "llm_model": "mistralai/Mistral-7B-v0.1"}).llm_model == "mistral-7b-v0.1"


def test_csv_path_accepts_flexible_column_spellings(tmp_path):
    # The CSV reader accepts flexible spellings (model/gpu/batch), not only the trace
    # convention (model_name/gpu_model), via the shared canonical alias map. Oracle:
    # each written cell maps to its named WorkloadSpec field verbatim (with the two
    # numeric columns coerced to int). A broken alias map would drop or mis-route a
    # column, changing one of these known values.
    from coastline.sdk.recommend.facade import _coerce_workload

    csv = tmp_path / "workload.csv"
    csv.write_text(
        "model,method,gpu,tokens_per_sample,batch_size\nmistral-7b-v0.1,lora,NVIDIA-A100-SXM4-80GB,1024,16\n"
    )
    wl = _coerce_workload(str(csv))
    assert wl.llm_model == "mistral-7b-v0.1"
    assert wl.gpu_model == "NVIDIA-A100-SXM4-80GB"
    assert wl.fine_tuning_method == "lora"
    assert wl.tokens_per_sample == 1024 and wl.batch_size == 16
    assert isinstance(wl.tokens_per_sample, int) and isinstance(wl.batch_size, int)


def test_default_context_max_nodes_uses_ceil_not_floor():
    # Regression: the facade derived max_nodes with `max_gpus // 8` (floor), so a 12-GPU
    # budget allowed only 1 node (8 GPUs) and silently dropped the 9-12 GPU configs that
    # need a 2nd node. ceil makes the whole budget explorable (matches grid.py's layout).
    # Oracle by hand at 8 GPUs/node: ceil(8/8)=1, ceil(12/8)=2, ceil(16/8)=2, ceil(20/8)=3;
    # the floor bug would give 1, 1, 2, 2 respectively.
    from coastline.sdk.models.workload import WorkloadSpec
    from coastline.sdk.recommend.facade import _default_context

    wl = WorkloadSpec(**_workload())
    assert _default_context(wl, 8).constraints.max_nodes == 1  # exact multiple: floor==ceil
    assert _default_context(wl, 12).constraints.max_nodes == 2  # floor gave 1 -> dropped 9-12
    assert _default_context(wl, 16).constraints.max_nodes == 2  # exact multiple: floor==ceil
    assert _default_context(wl, 20).constraints.max_nodes == 3  # floor gave 2 -> dropped 17-20


# --------------------------------------------------------------------------- #
# Preset / alpha-beta steering: energy vs performance pick different configs
# --------------------------------------------------------------------------- #


def _top_total_gpus(recs):
    return recs[0].total_gpus


def test_energy_preset_favors_fewer_gpus_than_performance_preset():
    """Direction invariant: because power_cost = per-GPU watts x GPU count, the
    energy preset (power weight 0.8) is pulled toward FEWER GPUs while the
    performance preset (throughput weight 0.8) is pulled toward MORE. So the energy
    pick must use <= the GPUs of the performance pick, and on this budget the two
    presets must not collapse to the same config (the weights genuinely steer)."""
    rec = Coastline("kavier")
    budget = [1, 2, 4, 8]
    energy = rec.recommend(_workload(), total_gpus=budget, preset="energy", top_k=1)
    performance = rec.recommend(_workload(), total_gpus=budget, preset="performance", top_k=1)
    assert energy and performance
    assert _top_total_gpus(energy) <= _top_total_gpus(performance)
    assert _top_total_gpus(energy) != _top_total_gpus(performance)
    # Independent oracle on the steering weights: each pick records the preset's
    # canonical (alpha, beta) from SPEC_PRESET_WEIGHTS, proving the named preset was
    # actually applied rather than silently defaulting to balanced (0.5, 0.5).
    assert (energy[0].metadata["alpha"], energy[0].metadata["beta"]) == SPEC_PRESET_WEIGHTS["energy"]
    assert (performance[0].metadata["alpha"], performance[0].metadata["beta"]) == SPEC_PRESET_WEIGHTS["performance"]


def test_explicit_alpha_beta_override_preset_and_reproduce_extremes():
    """Explicit alpha/beta override the preset and reproduce the preset extremes:
    alpha-heavy (power) -> the energy-like pick; beta-heavy (throughput) -> the
    perf-like pick. Oracle: same direction invariant as the named presets, PLUS the
    recorded weights equal the (already-normalized-to-1) inputs and preset='custom',
    confirming explicit weights win over the default 'balanced' preset."""
    rec = Coastline("kavier")
    budget = [1, 2, 4, 8]
    energy_like = rec.recommend(_workload(), total_gpus=budget, alpha=0.8, beta=0.2, top_k=1)
    perf_like = rec.recommend(_workload(), total_gpus=budget, alpha=0.2, beta=0.8, top_k=1)
    assert energy_like and perf_like
    assert _top_total_gpus(energy_like) <= _top_total_gpus(perf_like)
    assert _top_total_gpus(energy_like) != _top_total_gpus(perf_like)
    # 0.8/0.2 already sums to 1, so normalization leaves them unchanged; preset dropped.
    assert (energy_like[0].metadata["alpha"], energy_like[0].metadata["beta"]) == (0.8, 0.2)
    assert energy_like[0].metadata["preset"] == "custom"
    assert (perf_like[0].metadata["alpha"], perf_like[0].metadata["beta"]) == (0.2, 0.8)


# --------------------------------------------------------------------------- #
# Input validation: empty CSV -> ValueError, bad type -> TypeError
# --------------------------------------------------------------------------- #


def test_empty_csv_raises_value_error(tmp_path):
    csv = tmp_path / "empty.csv"
    # Header only, no data rows -> pandas reads an empty frame -> facade must reject
    # it with a ValueError rather than IndexError on row 0 / returning [].
    csv.write_text("model,method,gpu,tokens_per_sample,batch_size\n")
    rec = Coastline("kavier")
    with pytest.raises(ValueError):
        rec.recommend(str(csv))


def test_unsupported_workload_type_raises_type_error():
    rec = Coastline("kavier")
    # An int is neither a WorkloadSpec, dict, nor CSV path -> TypeError (contract),
    # distinct from the ValueError raised for a well-typed-but-empty CSV above.
    with pytest.raises(TypeError):
        rec.recommend(12345)


def test_max_gpus_zero_raises_clear_value_error():
    """max_gpus=0 must raise a clear ValueError with the documented message, NOT a raw
    pydantic ValidationError leaking from deep in the context builder."""
    rec = Coastline("kavier")
    with pytest.raises(ValueError, match="max_gpus must be >= 1"):
        rec.recommend(_workload(), max_gpus=0)


def test_max_gpus_negative_raises_clear_value_error():
    """max_gpus < 0 is equally invalid; confirm the same guard fires (boundary is
    < 1, so -1 and 0 are both rejected while 1 would pass)."""
    rec = Coastline("kavier")
    with pytest.raises(ValueError, match="max_gpus must be >= 1"):
        rec.recommend(_workload(), max_gpus=-1)


# --------------------------------------------------------------------------- #
# feasibility='rules' constructor path (under COASTLINE_ALLOW_RULES_FALLBACK=1)
# --------------------------------------------------------------------------- #


def test_rules_feasibility_admits_only_batch_divisible_configs(monkeypatch):
    """``Coastline(feasibility="rules")`` drives the divisibility-only checker end to
    end. Oracle: the rules checker admits a config iff batch_size % total_gpus == 0.
    With batch_sizes=[8] and budget [1,2,4,8], the divisor set of 8 is {1,2,4,8}, so
    EXACTLY those GPU counts must appear (and 3/5/6/7 would be rejected if present).
    A checker that ignored divisibility would let a non-divisor through and fail the
    modulo assertion."""
    monkeypatch.setenv("COASTLINE_ALLOW_RULES_FALLBACK", "1")
    rec = Coastline("kavier", feasibility="rules")
    assert rec.feasibility == "rules"
    out = rec.recommend(_workload(), total_gpus=[1, 2, 4, 8], batch_sizes=[8], top_k=99)
    assert out, "rules feasibility should still yield recommendations"
    assert all(isinstance(r, Recommendation) for r in out)
    admitted = {r.total_gpus for r in out}
    # By hand: 8 is divisible by exactly 1, 2, 4, 8 among the budget -> all admitted.
    assert admitted == {1, 2, 4, 8}
    for r in out:
        assert 8 % r.total_gpus == 0
