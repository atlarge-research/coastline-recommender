"""One home for Coastline's closed-set vocabularies and the default search space.

Values on the wire (YAML / JSON / CSV) stay plain strings; the enums here use a ``str`` base so
they compare and serialize as that wire string (``FeasibilityMode.RULES == "rules"``) while giving
the code a single, typed source of truth. The default lists are the fallback search space used
only when a config's ``grid`` doesn't specify its own.
"""

from __future__ import annotations

from enum import Enum


class FeasibilityMode(str, Enum):
    """How a candidate configuration's feasibility is checked."""

    AUTOCONF = "autoconf"  # OOM-aware AutoConf model (default)
    RULES = "rules"  # divisibility rules only
    NONE = "none"  # no feasibility check


class EnergyBackend(str, Enum):
    """Power/energy predictor backend."""

    KAVIER_POWER = "kavier_power"


class Method(str, Enum):
    """PEFT fine-tuning method."""

    FULL = "full"
    LORA = "lora"
    GPTQ_LORA = "gptq-lora"
    QLORA = "qlora"


class Strategy(str, Enum):
    """Recommendation policy family."""

    MULTI_OBJECTIVE = "multi_objective"
    MIN_GPU = "min_gpu"


class Preset(str, Enum):
    """Multi-objective weight preset. The ``*_FRONTIER`` variants share their base weights and
    differ only in the score-normalization set (the non-dominated frontier)."""

    ENERGY = "energy"
    BALANCED = "balanced"
    PERFORMANCE = "performance"
    ENERGY_FRONTIER = "energy-frontier"
    BALANCED_FRONTIER = "balanced-frontier"
    PERFORMANCE_FRONTIER = "performance-frontier"


class SelectionPolicy(str, Enum):
    """How the winning candidate is chosen: ``min_gpu`` = fewest feasible GPUs; the rest rank on
    the weighted throughput↔energy score."""

    MIN_GPU = "min_gpu"
    PERFORMANCE = "performance"
    ENERGY = "energy"
    BALANCED = "balanced"


class NormalizationMode(str, Enum):
    """Score-normalization set: over all feasible candidates (``grid``) or the non-dominated frontier."""

    GRID = "grid"
    FRONTIER = "frontier"


# α (power weight), β (throughput weight) per base preset. The -frontier variants are derived
# (same weights, different normalization) rather than re-listed.
_BASE_PRESET_WEIGHTS: dict[str, tuple[float, float]] = {
    Preset.ENERGY: (0.8, 0.2),
    Preset.BALANCED: (0.5, 0.5),
    Preset.PERFORMANCE: (0.2, 0.8),
}
PRESET_WEIGHTS: dict[str, tuple[float, float]] = {
    **{p.value: w for p, w in _BASE_PRESET_WEIGHTS.items()},
    **{f"{p.value}-frontier": w for p, w in _BASE_PRESET_WEIGHTS.items()},
}

# Base preset -> ranking policy; the -frontier variants map to the same policy (derived).
_BASE_PRESET_TO_POLICY: dict[str, "SelectionPolicy"] = {
    Preset.ENERGY: SelectionPolicy.ENERGY,
    Preset.BALANCED: SelectionPolicy.BALANCED,
    Preset.PERFORMANCE: SelectionPolicy.PERFORMANCE,
}
PRESET_TO_POLICY: dict[str, "SelectionPolicy"] = {
    **{p.value: pol for p, pol in _BASE_PRESET_TO_POLICY.items()},
    **{f"{p.value}-frontier": pol for p, pol in _BASE_PRESET_TO_POLICY.items()},
}


# The standard node topology (a DGX-style node holds 8 GPUs); a context/config may override it.
DEFAULT_GPUS_PER_NODE: int = 8

# --- default search space (a config's ``grid`` overrides these per run) ---
GPU_BUDGETS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DEFAULT_BATCH_SIZES: list[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256]
DEFAULT_TOKENS_PER_SAMPLE: list[int] = [512, 1024, 2048, 4096, 8192]

# The AutoConf OOM model version used when a config doesn't pin one.
DEFAULT_AUTOCONF_MODEL_VERSION = "3.1.0"

#: The AutoConf model whose classifier may decide a whole chunk of candidates in one call.
#: Verified on this version only: batched and per-row predicts agree on every verdict over
#: 3,339 real grid candidates and 81,928 synthetic ones, with probabilities drifting at most
#: 6e-7 against a 1.8e-3 margin to the decision threshold. Another version falls back to one
#: call per candidate. COASTLINE_NO_AUTOCONF_BATCH=1 opts out entirely.
BATCHABLE_AUTOCONF_MODEL_VERSION = "3.1.0"

# The empirical OOM guard's per-device token ceiling, re-derived from the nine Zurich campaigns
# (135 jobs: 49 OOM, 86 completed). 60,224 is the UNIQUE accuracy maximum at 89.63% (121/135),
# catching 36 of 49 OOMs with a single false alarm.
#
# The comparison is STRICTLY greater-than and that is load-bearing: 14 jobs sit at exactly
# 60,224 tokens/device and every one of them completed. Using >= drops accuracy to 79.3%.
#
# This rule cannot do better. Completed jobs run up to 79,200 tokens/device while OOMs start at
# 15,104, so the classes overlap over 87 of the 135 jobs; 89.63% is the provable ceiling for any
# single tokens/device threshold. The 13 missed OOMs separate on gradient_checkpointing instead.
# Treat it as a guard, not a classifier: the threshold was selected on the same campaigns it is
# scored against, so true out-of-sample accuracy is lower.
EMPIRICAL_OOM_TOKEN_BUDGET = 60_224
