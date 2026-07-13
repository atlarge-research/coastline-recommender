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


# --- default search space (a config's ``grid`` overrides these per run) ---
GPU_BUDGETS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DEFAULT_BATCH_SIZES: list[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256]
DEFAULT_TOKENS_PER_SAMPLE: list[int] = [512, 1024, 2048, 4096, 8192]

# The AutoConf OOM model version used when a config doesn't pin one.
DEFAULT_AUTOCONF_MODEL_VERSION = "3.1.0"
