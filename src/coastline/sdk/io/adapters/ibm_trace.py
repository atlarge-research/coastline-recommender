"""The IBM fine-tuning-trace adapter: dotted ``metadata.*`` / ``resources.*`` ↔ canonical.

This module is the single home for the trace column names (previously defined inline in
``sdk/trace/recommend.py`` and re-imported by ``sdk/trace/to_runs.py``). ``to_canonical``
projects a trace onto canonical workload rows; ``from_canonical`` writes a recommended
layout back onto the original trace's ``resources.*`` / ``metadata.batch_size`` columns.

Only the layout columns are round-tripped here — the recommend-trace *policy* (estimated
duration, kept-unchanged rows, ``recommendation_note``) lives in ``sdk/trace/recommend.py``,
not in this format adapter.
"""

from __future__ import annotations

import pandas as pd

from coastline.sdk.io.adapters.base import register

# --- trace schema column names (canonical home; imported by trace.recommend + trace.to_runs) ---
MODEL = "metadata.model_name"
METHOD = "metadata.method"
GPU = "resources.gpu_model"
TOKENS = "metadata.tokens_per_sample"  # the int the user wants; nominal seq length is fine here
BATCH = "metadata.batch_size"
GPN = "resources.num_gpus_per_node"  # GPUs PER NODE, never the cluster total
NODES = "resources.num_nodes"
# ground-truth work (config-independent): tokens the job actually processed
ACT_TPS = "metadata.output.train_tokens_per_second"
ACT_RUNTIME = "metadata.train_runtime"
# observed job duration — the fallback when no recommendation can be made
ACT_DURATION = "metadata.output.extrapolated_duration"

# dotted trace column -> canonical WorkloadSpec field
_TRACE_TO_CANONICAL: dict[str, str] = {
    MODEL: "llm_model",
    METHOD: "fine_tuning_method",
    GPU: "gpu_model",
    TOKENS: "tokens_per_sample",
    BATCH: "batch_size",
    GPN: "gpus_per_node",
    NODES: "number_of_nodes",
}

# canonical recommended-layout column -> the dotted trace column it is written back to
_CANONICAL_LAYOUT_TO_TRACE: dict[str, str] = {
    "number_of_nodes": NODES,
    "gpus_per_node": GPN,
    "batch_size": BATCH,
}


class IBMTraceAdapter:
    name = "ibm_trace"

    def to_canonical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Project the dotted trace onto canonical workload columns (present ones only)."""
        present = {src: dst for src, dst in _TRACE_TO_CANONICAL.items() if src in df.columns}
        return df[list(present)].rename(columns=present)

    def from_canonical(self, recommended: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
        """Overwrite the original trace's layout columns with the recommended ones."""
        out = original.copy()
        for canonical_col, trace_col in _CANONICAL_LAYOUT_TO_TRACE.items():
            if canonical_col in recommended.columns:
                out[trace_col] = recommended[canonical_col].to_numpy()
        return out


register(IBMTraceAdapter())
