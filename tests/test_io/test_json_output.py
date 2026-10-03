"""Tests for ``save_recommendation_to_json`` (coastline.sdk.io.interface.json_output).

The helper does no arithmetic; it maps recommendation fields to named JSON keys that other code
reads. The tests check the key each field lands under, that number_of_nodes is written as
``workers``, and when the energy, metadata and rationale blocks appear. Every source field has a
different value (gpus_per_node=8, number_of_nodes=2, total_gpus=16, and so on), so a swapped
field shows up as a wrong number. The recommendations are synthetic and written to tmp_path.
"""

import json

from coastline.sdk.io.interface.json_output import (
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
    """Each field lands under its documented key, and number_of_nodes is written as workers.

    The source values (16, 8, 2, 1234.5, 450.0, 2.74) all differ, so a swapped field shows up."""
    rec = _make_rec(metadata={"predicted_power_watts": 450.0, "tokens_per_watt": 2.74})
    out = tmp_path / "rec.json"
    save_recommendation_to_json(rec, out)
    data = _read(out)
    # workers is number_of_nodes (2); gpus_per_node is 8 and total_gpus 16
    assert data["configuration"] == {"total_gpus": 16, "gpus_per_node": 8, "workers": 2}
    assert data["performance"]["throughput_tokens_per_sec"] == 1234.5
    assert data["strategy"] == "min_gpu"
    # the energy block holds the metadata values under its own key names
    assert data["energy"]["power_watts"] == 450.0
    assert data["energy"]["efficiency_tokens_per_watt"] == 2.74


def test_single_energy_block_present_iff_power_truthy(tmp_path):
    """The energy block is written when predicted_power_watts is set (300.0) and left out when
    the key is missing."""
    out_with = tmp_path / "with.json"
    save_recommendation_to_json(_make_rec(metadata={"predicted_power_watts": 300.0}), out_with)
    assert "energy" in _read(out_with)

    out_without = tmp_path / "without.json"
    save_recommendation_to_json(_make_rec(metadata={}), out_without)
    assert "energy" not in _read(out_without)


def test_single_energy_block_present_when_power_is_zero(tmp_path):
    """A power of 0.0 W is a value, so the energy block is written with power_watts 0.0."""
    out = tmp_path / "zero.json"
    save_recommendation_to_json(_make_rec(metadata={"predicted_power_watts": 0.0, "tokens_per_watt": 0.0}), out)
    energy = _read(out)["energy"]
    assert energy["power_watts"] == 0.0  # 0.0 is falsy, so a truthiness check would drop the block


def test_single_efficiency_defaults_to_zero_without_tokens_per_watt(tmp_path):
    """With power but no tokens_per_watt, efficiency_tokens_per_watt is written as 0."""
    out = tmp_path / "rec.json"
    save_recommendation_to_json(_make_rec(metadata={"predicted_power_watts": 300.0}), out)
    energy = _read(out)["energy"]
    assert "efficiency_tokens_per_watt" in energy
    assert energy["efficiency_tokens_per_watt"] == 0


def test_single_metadata_block_toggles_with_flag_and_is_independent_of_energy(tmp_path):
    """include_metadata controls the metadata block alone; the energy block is written either way.

    By default the metadata block equals the input metadata; with include_metadata=False it is
    left out."""
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
    """A non-empty rationale ('why this config') is written as given; None or an empty string
    leaves the key out."""
    out_with = tmp_path / "why.json"
    save_recommendation_to_json(_make_rec(), out_with, rationale="fewest GPUs")
    assert _read(out_with)["rationale"] == "fewest GPUs"

    out_default = tmp_path / "no_why.json"
    save_recommendation_to_json(_make_rec(), out_default)  # rationale defaults to None
    assert "rationale" not in _read(out_default)
