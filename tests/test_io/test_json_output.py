"""Tests for coastline.sdk.io.interface.json_output (the two save helpers).

These helpers own no arithmetic; their contract is a *schema mapping* that
downstream consumers (CLI/report/verifier) read by name. So the independent
oracle here is the spec of that schema: which output key each input field lands
under, the rename `workers <- number_of_nodes`, and the conditional-block logic
(energy iff power is truthy, metadata iff the flag, rationale iff truthy).

To make the mapping oracle able to catch a field swap, every source field is
given a DISTINCT value: e.g. gpus_per_node=8, number_of_nodes=2, total_gpus=16,
throughput/power/efficiency all mutually distinct. A test that read the wrong
source field would then produce a different number and go red.

Self-contained: synthetic Recommendations -> tmp_path, no data/model artifacts.
"""

import json

from coastline.sdk.io.interface.json_output import (
    save_batch_recommendations,
    save_recommendation_to_json,
)
from coastline.sdk.models.recommendation import Recommendation


def _make_rec(
    *,
    gpus_per_node=8,
    number_of_nodes=2,
    total_gpus=16,
    strategy="min_gpu",
    predicted_throughput=1234.5,
    predicted_runtime_seconds=600.0,
    metadata=None,
):
    return Recommendation(
        gpus_per_node=gpus_per_node,
        number_of_nodes=number_of_nodes,
        total_gpus=total_gpus,
        strategy=strategy,
        predicted_throughput=predicted_throughput,
        predicted_runtime_seconds=predicted_runtime_seconds,
        metadata=metadata if metadata is not None else {},
    )


def _read(path):
    with open(path) as f:
        return json.load(f)


def test_single_schema_maps_each_source_field_to_its_named_key(tmp_path):
    """Schema mapping oracle: each input field lands under its documented key,
    including the rename workers <- number_of_nodes. Source values are mutually
    distinct (16, 8, 2, 1234.5, 450.0, 2.74) so a swapped mapping (e.g. workers
    <- gpus_per_node would give 8, not 2) is caught."""
    rec = _make_rec(metadata={"predicted_power_watts": 450.0, "tokens_per_watt": 2.74})
    out = tmp_path / "rec.json"
    save_recommendation_to_json(rec, out)
    data = _read(out)
    # workers is number_of_nodes(=2), NOT gpus_per_node(=8) nor total_gpus(=16)
    assert data["configuration"] == {"total_gpus": 16, "gpus_per_node": 8, "workers": 2}
    assert data["performance"]["throughput_tokens_per_sec"] == 1234.5
    assert data["strategy"] == "min_gpu"
    # energy mirrors metadata keys under renamed output keys
    assert data["energy"]["power_watts"] == 450.0
    assert data["energy"]["efficiency_tokens_per_watt"] == 2.74


def test_single_energy_block_present_iff_power_truthy(tmp_path):
    """Conditional-block logic: the energy block is gated on the *truthiness* of
    predicted_power_watts (the code uses metadata.get(...) as a bool), not merely
    its presence. Oracle: present for 300.0; absent when the key is missing."""
    out_with = tmp_path / "with.json"
    save_recommendation_to_json(_make_rec(metadata={"predicted_power_watts": 300.0}), out_with)
    assert "energy" in _read(out_with)

    out_without = tmp_path / "without.json"
    save_recommendation_to_json(_make_rec(metadata={}), out_without)
    assert "energy" not in _read(out_without)


def test_single_energy_block_present_when_power_is_zero(tmp_path):
    """A present-but-zero power (predicted_power_watts == 0.0) is a real measurement, so the
    energy block IS emitted with power_watts == 0.0. The gate keys on presence (`is not None`),
    not truthiness. Regression: the old `if metadata.get('predicted_power_watts')` truthiness
    gate treated 0.0 as falsy and silently dropped the whole energy block."""
    out = tmp_path / "zero.json"
    save_recommendation_to_json(_make_rec(metadata={"predicted_power_watts": 0.0, "tokens_per_watt": 0.0}), out)
    energy = _read(out)["energy"]
    assert energy["power_watts"] == 0.0  # would be a missing "energy" key under the old truthiness gate


def test_single_efficiency_defaults_to_zero_without_tokens_per_watt(tmp_path):
    """Documented default: power present but no tokens_per_watt -> efficiency 0
    (the metadata.get(..., 0) fallback), not a missing key nor None."""
    out = tmp_path / "rec.json"
    save_recommendation_to_json(_make_rec(metadata={"predicted_power_watts": 300.0}), out)
    energy = _read(out)["energy"]
    assert "efficiency_tokens_per_watt" in energy
    assert energy["efficiency_tokens_per_watt"] == 0


def test_single_metadata_block_toggles_with_flag_and_is_independent_of_energy(tmp_path):
    """include_metadata gates ONLY the metadata block; energy is derived from the
    same metadata but on a separate branch, so it must survive either way.
    Oracle: default True -> metadata block equals the input metadata verbatim;
    False -> no metadata block; both keep energy (power present)."""
    md = {"predicted_power_watts": 450.0, "tokens_per_watt": 2.74, "note": "x"}

    out_default = tmp_path / "default.json"
    save_recommendation_to_json(_make_rec(metadata=md), out_default)
    data_default = _read(out_default)
    assert data_default["metadata"] == md  # full round-trip of the input metadata
    assert "energy" in data_default

    out_off = tmp_path / "off.json"
    save_recommendation_to_json(_make_rec(metadata=md), out_off, include_metadata=False)
    data_off = _read(out_off)
    assert "metadata" not in data_off
    assert "energy" in data_off  # energy independent of the metadata flag


def test_single_rationale_present_iff_truthy(tmp_path):
    """rationale ('why this config') is added on a truthiness branch: a non-empty
    string appears verbatim; None/empty omits the key entirely."""
    out_with = tmp_path / "why.json"
    save_recommendation_to_json(_make_rec(), out_with, rationale="fewest GPUs")
    assert _read(out_with)["rationale"] == "fewest GPUs"

    out_default = tmp_path / "no_why.json"
    save_recommendation_to_json(_make_rec(), out_default)  # rationale defaults to None
    assert "rationale" not in _read(out_default)


def test_batch_rank_is_one_indexed_in_input_order(tmp_path):
    """Ranking invariant: N recs -> ranks exactly 1..N (one-indexed, no gaps) in
    the order given. Distinct strategies s0,s1,s2 pin that the enumerate order is
    preserved rather than reordered, and that rank starts at 1 not 0."""
    out = tmp_path / "batch.json"
    save_batch_recommendations([_make_rec(strategy=f"s{i}") for i in range(3)], out)
    data = _read(out)
    entries = data["recommendations"]
    assert data["count"] == 3  # count == len(input)
    assert [e["rank"] for e in entries] == [1, 2, 3]
    assert [e["strategy"] for e in entries] == ["s0", "s1", "s2"]


def test_batch_entry_flattens_config_and_gates_energy_per_entry(tmp_path):
    """Each batch entry flattens config/perf into a flat dict (workers <-
    number_of_nodes rename again), and the energy fields are gated PER ENTRY on
    that entry's own power. Distinct values (12,4,3,999,500,1.5) catch a swap;
    the second entry (no power) must lack power_watts/efficiency."""
    with_e = _make_rec(
        gpus_per_node=4,
        number_of_nodes=3,
        total_gpus=12,
        strategy="multi_objective",
        predicted_throughput=999.0,
        metadata={"predicted_power_watts": 500.0, "tokens_per_watt": 1.5},
    )
    without_e = _make_rec(strategy="no-energy", metadata={})
    out = tmp_path / "batch.json"
    save_batch_recommendations([with_e, without_e], out)
    e0, e1 = _read(out)["recommendations"]
    # workers is number_of_nodes(=3), not gpus_per_node(=4)
    assert (e0["total_gpus"], e0["gpus_per_node"], e0["workers"]) == (12, 4, 3)
    assert e0["throughput"] == 999.0
    assert e0["power_watts"] == 500.0 and e0["efficiency"] == 1.5
    assert "power_watts" not in e1 and "efficiency" not in e1


def test_batch_empty_list_yields_zero_count_and_no_entries(tmp_path):
    """Boundary: empty input -> count 0 and an empty (not missing) list, so the
    count == len(recommendations) invariant holds at N=0."""
    out = tmp_path / "batch.json"
    save_batch_recommendations([], out)
    data = _read(out)
    assert data["count"] == 0
    assert data["recommendations"] == []
