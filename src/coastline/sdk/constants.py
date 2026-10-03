"""Closed-set vocabularies and the default search space.

Values in YAML, JSON and CSV are plain strings. The enums subclass ``str``, so they compare and
serialize as those strings (``FeasibilityMode.RULES == "rules"``). The default lists are the
search space used when a config's ``grid`` does not set its own.
"""

from __future__ import annotations

from enum import Enum


class FeasibilityMode(str, Enum):
    """How a candidate configuration's feasibility is checked."""

    AUTOCONF = "autoconf"  # OOM-aware AutoConf model (default)
    RULES = "rules"  # structural sanity guards only; no memory model (see EMPIRICAL_OOM_TOKEN_BUDGET)
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
    the weighted throughput and energy score."""

    MIN_GPU = "min_gpu"
    PERFORMANCE = "performance"
    ENERGY = "energy"
    BALANCED = "balanced"


class NormalizationMode(str, Enum):
    """Score-normalization set: over all feasible candidates (``grid``) or the non-dominated frontier."""

    GRID = "grid"
    FRONTIER = "frontier"


# (alpha, beta) per base preset: alpha weights power, beta weights throughput. The -frontier
# variants reuse these weights with frontier normalization.
_BASE_PRESET_WEIGHTS: dict[str, tuple[float, float]] = {
    Preset.ENERGY: (0.8, 0.2),
    Preset.BALANCED: (0.5, 0.5),
    Preset.PERFORMANCE: (0.2, 0.8),
}
PRESET_WEIGHTS: dict[str, tuple[float, float]] = {
    **{p.value: w for p, w in _BASE_PRESET_WEIGHTS.items()},
    **{f"{p.value}-frontier": w for p, w in _BASE_PRESET_WEIGHTS.items()},
}

# Ranking policy per base preset; each -frontier variant uses its base preset's policy.
_BASE_PRESET_TO_POLICY: dict[str, "SelectionPolicy"] = {
    Preset.ENERGY: SelectionPolicy.ENERGY,
    Preset.BALANCED: SelectionPolicy.BALANCED,
    Preset.PERFORMANCE: SelectionPolicy.PERFORMANCE,
}
PRESET_TO_POLICY: dict[str, "SelectionPolicy"] = {
    **{p.value: pol for p, pol in _BASE_PRESET_TO_POLICY.items()},
    **{f"{p.value}-frontier": pol for p, pol in _BASE_PRESET_TO_POLICY.items()},
}


# GPUs per node (a DGX-style node holds 8); a context or config may override it.
DEFAULT_GPUS_PER_NODE: int = 8

# Default search space; a config's ``grid`` overrides it.
GPU_BUDGETS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DEFAULT_BATCH_SIZES: list[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256]
DEFAULT_TOKENS_PER_SAMPLE: list[int] = [512, 1024, 2048, 4096, 8192]

# The AutoConf OOM model version used when a config doesn't pin one.
DEFAULT_AUTOCONF_MODEL_VERSION = "3.1.0"

#: AutoConf model versions whose classifier may judge a chunk of candidates in one call. Each was
#: compared with the one-call-per-candidate path on 3,339 real grid candidates and an 81,928-row
#: adversarial grid, at chunk sizes 1, 8, 35, 250 and 1000 (verdicts, metadata, probabilities):
#:
#:   3.1.0 (CatBoost + WeightedEnsemble_L2): bitwise identical.
#:   3.0.0 (NeuralNetTorch + WeightedEnsemble_L2): same verdicts; probabilities differ by up to
#:          1.4e-6, and the candidate closest to the 0.5 threshold is 3.4e-5 away (~25x margin).
#:
#: Other versions were not measured and get one call per candidate.
#: COASTLINE_NO_AUTOCONF_BATCH=1 turns batching off.
BATCHABLE_AUTOCONF_MODEL_VERSIONS = frozenset({"3.1.0", "3.0.0"})

# Per-device token ceiling of the empirical OOM guard, fitted on the nine Zurich campaigns
# (135 jobs: 49 OOM, 86 completed). 60,224 is the only threshold with the top accuracy, 89.63%
# (121/135): it catches 36 of 49 OOMs with one false alarm.
#
# The guard rejects strictly more tokens/device than this. 14 jobs ran at exactly 60,224
# tokens/device and all completed; with >= the accuracy drops to 79.3%.
#
# No single tokens/device threshold does better: completed jobs reach 79,200 tokens/device and
# OOMs start at 15,104, so the classes overlap on 87 of the 135 jobs. The 13 missed OOMs differ
# in gradient_checkpointing. The threshold was chosen on the same campaigns it is scored on, so
# out-of-sample accuracy is lower.
EMPIRICAL_OOM_TOKEN_BUDGET = 60_224
