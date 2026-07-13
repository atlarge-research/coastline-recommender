"""Phase 2: the canonical schema + format-adapter boundary.

Adapters own FORMAT only (foreign columns <-> canonical workload rows). These tests pin the
canonical vocabulary, the identity + ibm_trace mappings and their round-trip, the registry,
and that the trace column constants moved into the ibm_trace adapter without changing value
(behavior-preserving consolidation — trace.recommend + trace.to_runs share them).
"""

from __future__ import annotations

import pandas as pd

from coastline.sdk.io import schema
from coastline.sdk.io.adapters import adapter_names, get_adapter
from coastline.sdk.io.adapters import ibm_trace as ibm


def test_schema_workload_aliases_resolve_to_canonical_fields():
    m = schema.workload_col_to_field()
    assert m["model"] == "llm_model"
    assert m["model_name"] == "llm_model"
    assert m["gpu"] == "gpu_model"
    assert m["seq_len"] == "tokens_per_sample"
    assert m["number_gpus"] == "gpus_per_node"  # per-node, never total
    assert "total_gpus" not in m  # derived-only, never a canonical input column


def test_schema_knob_aliases_separate_from_workload():
    k = schema.knob_col_to_field()
    assert k["gpu_budget"] == "max_gpus"
    assert k["goal_label"] == "goal"
    assert k["runtime_guard_k"] == "max_slowdown"
    # knobs and workload fields are disjoint vocabularies
    assert set(k).isdisjoint(schema.workload_col_to_field())


def test_coastline_identity_adapter_normalizes_aliases():
    adapter = get_adapter("coastline")
    df = pd.DataFrame([{"model": "mistral-7b-v0.1", "gpu": "NVIDIA-A100-SXM4-80GB", "batch": 8}])
    canon = adapter.to_canonical(df)
    assert list(canon.columns) == ["llm_model", "gpu_model", "batch_size"]
    # from_canonical is pass-through (result already canonical output)
    recs = pd.DataFrame([{"total_gpus": 1, "feasible": True}])
    assert adapter.from_canonical(recs, df).equals(recs)


def test_ibm_trace_adapter_to_canonical_projects_workload_columns():
    df = pd.DataFrame(
        [
            {
                ibm.MODEL: "granite-3.3-8b",
                ibm.METHOD: "lora",
                ibm.GPU: "NVIDIA-A100-SXM4-80GB",
                ibm.TOKENS: 1024,
                ibm.BATCH: 16,
                ibm.GPN: 8,
                ibm.NODES: 1,
                "metadata.submission_time_issue_85_rescaled": 0.0,  # ignored non-workload col
            }
        ]
    )
    canon = get_adapter("ibm_trace").to_canonical(df)
    assert set(canon.columns) == {
        "llm_model", "fine_tuning_method", "gpu_model",
        "tokens_per_sample", "batch_size", "gpus_per_node", "number_of_nodes",
    }
    assert canon.iloc[0]["llm_model"] == "granite-3.3-8b"
    assert canon.iloc[0]["gpus_per_node"] == 8


def test_ibm_trace_adapter_round_trips_layout():
    original = pd.DataFrame([{ibm.NODES: 1, ibm.GPN: 8, ibm.BATCH: 16, "metadata.model_name": "x"}])
    recommended = pd.DataFrame([{"number_of_nodes": 2, "gpus_per_node": 4, "batch_size": 32}])
    out = get_adapter("ibm_trace").from_canonical(recommended, original)
    assert out.iloc[0][ibm.NODES] == 2
    assert out.iloc[0][ibm.GPN] == 4
    assert out.iloc[0][ibm.BATCH] == 32
    assert out.iloc[0]["metadata.model_name"] == "x"  # untouched original column preserved


def test_registry_resolves_builtin_adapters_and_rejects_unknown():
    assert "coastline" in adapter_names() and "ibm_trace" in adapter_names()
    assert get_adapter(None).name == "coastline"  # default
    try:
        get_adapter("nope")
    except ValueError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown adapter")


def test_trace_constants_consolidated_without_value_change():
    """trace.recommend + trace.to_runs now source their dotted-column names from the
    ibm_trace adapter — the values must be identical to the pre-move spellings."""
    from coastline.sdk.trace import recommend as tr
    from coastline.sdk.trace import to_runs

    assert tr._MODEL == ibm.MODEL == "metadata.model_name"
    assert tr._GPN == ibm.GPN == "resources.num_gpus_per_node"
    assert tr._ACT_TPS == ibm.ACT_TPS
    # to_runs builds its flat mapping off the same trace constants (imported via recommend)
    assert to_runs._TRACE_TO_FLAT[ibm.MODEL] == "model_name"
