"""The one canonical Coastline CSV vocabulary.

A *canonical* workload row uses the ``WorkloadSpec`` field names for the workload and the
engine-knob names for the recommendation controls. Every foreign spelling an input surface
accepts resolves here through :data:`WORKLOAD_ALIASES` / :data:`KNOB_ALIASES`; every
recommendation output uses :data:`OUTPUT_COLUMNS`. This is the single vocabulary the batch
surfaces and the format adapters (``coastline.sdk.io.adapters``) share.

Foreign CSV shapes (the IBM fine-tuning trace, the flat measured-runs schema) are handled
*only* at the boundary by an adapter that maps them to/from this canonical shape — the
recommend core never sees anything but canonical rows.

Two invariants the whole codebase relies on, restated here as the contract:

* ``total_gpus`` is **derived only** (``gpus_per_node × number_of_nodes``) — never a canonical
  input column.
* ``number_gpus`` / ``num_gpus_per_node`` always mean GPUs **per node**, not the cluster total.
"""

from __future__ import annotations

from coastline.sdk.models.aliases import WORKLOAD_FIELD_ALIASES

# Workload-field aliases are owned by ``models.aliases`` (shared with WorkloadSpec
# construction); re-exported here so the canonical vocabulary has a single import site.
WORKLOAD_ALIASES: dict[str, tuple[str, ...]] = WORKLOAD_FIELD_ALIASES

# Recommendation-control (engine-knob) aliases — the non-workload columns a batch row may
# carry. Kept separate from workload fields because they configure the search, not the job.
KNOB_ALIASES: dict[str, tuple[str, ...]] = {
    "dataset_size": ("dataset_size", "num_samples"),
    "epochs": ("epochs",),
    "max_gpus": ("max_gpus", "gpu_budget"),
    "goal": ("goal", "goal_label"),
    "predictor": ("predictor", "throughput_estim"),
    "lookup": ("lookup", "lookup_csv"),
    "max_slowdown": ("max_slowdown", "runtime_guard_k"),
}

# Canonical recommendation output columns (the Kavier-style spelling the batch API emits).
# Surface-specific renamings (``recommended_*`` / ``predicted_*`` for the CSV writer, nested
# JSON for the run artifact) are presentation wrappers over this set.
OUTPUT_COLUMNS: tuple[str, ...] = (
    "rank",
    "total_gpus",
    "gpus_per_node",
    "number_of_nodes",
    "batch_size",
    "throughput_tok_s",
    "runtime_s",
    "energy_wh",
    "energy_kwh",
    "tokens_per_watt",
    "power_w",
    "feasible",
    "error",
    "rationale",
)


def _flatten(aliases: dict[str, tuple[str, ...]]) -> dict[str, str]:
    """``{field: (spellings...)}`` → ``{spelling: field}`` for column-driven readers."""
    return {col: field for field, cols in aliases.items() for col in cols}


def workload_col_to_field() -> dict[str, str]:
    """Every accepted workload-column spelling → its canonical ``WorkloadSpec`` field."""
    return _flatten(WORKLOAD_ALIASES)


def knob_col_to_field() -> dict[str, str]:
    """Every accepted engine-knob spelling → its canonical knob name."""
    return _flatten(KNOB_ALIASES)


def canonicalize_columns(columns: list[str]) -> dict[str, str]:
    """Map a set of input column names to their canonical spellings (workload + knobs).
    Columns with no known alias are left out (the caller decides whether to keep them)."""
    lookup = {**workload_col_to_field(), **knob_col_to_field()}
    return {col: lookup[col] for col in columns if col in lookup}
