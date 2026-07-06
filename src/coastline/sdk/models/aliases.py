"""Canonical workload-field aliases — one source of truth for the flexible column
spellings accepted by the CSV batch recommender and the facade.

Keyed by WorkloadSpec field, each value lists the accepted input spellings (the field
name itself first). ``col_to_field_map`` flattens this for column-driven readers.
"""

WORKLOAD_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "llm_model": ("llm_model", "model_name", "model"),
    "fine_tuning_method": ("fine_tuning_method", "method", "peft"),
    "gpu_model": ("gpu_model", "gpu"),
    "tokens_per_sample": ("tokens_per_sample", "seq_len", "max_tokens", "tokens"),
    "batch_size": ("batch_size", "batch"),
    "gpus_per_node": ("gpus_per_node", "number_gpus", "num_gpus"),
    "number_of_nodes": ("number_of_nodes", "number_nodes", "nodes"),
}


def col_to_field_map(aliases: dict[str, tuple[str, ...]] = WORKLOAD_FIELD_ALIASES) -> dict[str, str]:
    """Flatten ``{field: (spellings...)}`` into ``{spelling: field}`` for readers that
    map input columns onto WorkloadSpec fields."""
    return {col: field for field, cols in aliases.items() for col in cols}
