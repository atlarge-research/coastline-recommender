"""Find and load the recommendation-policy YAML (strategy, predictors, grid)."""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

from coastline.sdk.pipeline.parallel import (
    DEFAULT_CLI_WORKERS,
    RUNTIME_SECTION,
    WORKERS_KEY,
    resolve_workers,
)

# Built-in policy (multi_objective, balanced) for the CLI, the Python API and the UI, used when
# no config file is found.
_BUILTIN_DEFAULT_PATH = Path(__file__).parent / "default_experiment.yaml"


@lru_cache(maxsize=1)
def _load_builtin_default() -> dict[str, Any]:
    with open(_BUILTIN_DEFAULT_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def builtin_default_config() -> dict[str, Any]:
    """A deep copy of the built-in policy (``strategy``, ``predictors``, ``grid``) from the bundled
    ``default_experiment.yaml``, used when no config file is present."""
    return copy.deepcopy(_load_builtin_default())


# The default that a loaded config file is merged over.
_DEFAULT_STRATEGY_CONFIG: dict[str, Any] = builtin_default_config()

# The repo's policy config, used by every entry point unless EXPERIMENT_CONFIG points elsewhere.
# parents[4] of this file is the repo root.
_CONFIG_ENV_KEY = "EXPERIMENT_CONFIG"
_CANONICAL_CONFIG = Path(__file__).resolve().parents[4] / "config" / "coastline_functionality" / "experiment.yaml"


def default_experiment_path() -> Path:
    """The policy config path used when none is given.

    ``EXPERIMENT_CONFIG`` if set, returned even when the file is missing so the caller can report
    it; else the repo's ``experiment.yaml``; else (in an installed wheel, which has no ``config/``)
    the bundled ``default_experiment.yaml`` that :func:`builtin_default_config` reads.
    """
    override = os.environ.get(_CONFIG_ENV_KEY)
    if override:
        return Path(override)
    return _CANONICAL_CONFIG if _CANONICAL_CONFIG.is_file() else _BUILTIN_DEFAULT_PATH


def load_runtime_workers(path: str | Path | None = None) -> Optional[int]:
    """``runtime.parallel_workers`` from the policy YAML, or None when it is not set.

    Separate from :func:`load_strategy_config` because the trace and batch paths build their
    config from arguments but still use the configured worker count. An invalid value is ignored.
    """
    path = Path(path) if path is not None else default_experiment_path()
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    section = loaded.get(RUNTIME_SECTION)
    if not isinstance(section, dict):
        return None
    try:
        return int(section[WORKERS_KEY])
    except (KeyError, TypeError, ValueError):
        return None


def resolve_cli_workers(flag: Optional[int] = None, path: str | Path | None = None) -> int:
    """Worker count for a command: ``--workers``, else the policy YAML, else 4.

    The SDK default stays 1: a library call runs inside someone else's program, and moving its
    work into subprocesses would bypass that program's monkeypatches, state and log handlers.
    """
    if flag is not None:
        return resolve_workers(flag)
    from_file = load_runtime_workers(path)
    return resolve_workers(from_file if from_file is not None else DEFAULT_CLI_WORKERS)


def _merge_dict(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_strategy_config(path: str | Path, *, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a recommendation-policy YAML merged under a base default (``strategy``/``predictors``/``grid``).

    ``default`` overrides the base config merged under the file; when omitted, the built-in
    default is used. An absent file returns the base default unchanged.
    """
    config = copy.deepcopy(default if default is not None else _DEFAULT_STRATEGY_CONFIG)
    path = Path(path)
    if not path.is_file():
        return config

    with open(path, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    for section in ("strategy", "predictors", "grid", RUNTIME_SECTION):
        if section in loaded:
            if isinstance(loaded[section], dict):
                config[section] = _merge_dict(config.get(section, {}), loaded[section])
            else:
                config[section] = loaded[section]

    return config
