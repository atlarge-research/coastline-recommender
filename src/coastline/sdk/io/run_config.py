"""Resolve + load the recommendation-policy YAML (strategy / predictors / grid)."""

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

# The one built-in recommendation-policy default (multi_objective/balanced), loaded from the
# bundled ``default_experiment.yaml`` so every surface (CLI, facade/API, UI) shares a single
# source instead of parallel hardcoded dicts. Used only when no config file is found.
_BUILTIN_DEFAULT_PATH = Path(__file__).parent / "default_experiment.yaml"


@lru_cache(maxsize=1)
def _load_builtin_default() -> dict[str, Any]:
    with open(_BUILTIN_DEFAULT_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def builtin_default_config() -> dict[str, Any]:
    """The one built-in recommendation-policy default (``strategy``/``predictors``/``grid``),
    loaded from the bundled ``default_experiment.yaml``. Every surface falls back to this when no
    config file is present. Returns a fresh deep copy — callers may mutate it freely."""
    return copy.deepcopy(_load_builtin_default())


# Module-level default dict (the deep copy every door merges under). Kept as a name for the
# surfaces + tests that reference it; its content is the bundled YAML, never a second literal.
_DEFAULT_STRATEGY_CONFIG: dict[str, Any] = builtin_default_config()

# The single canonical recommendation-policy config file. Every door falls back to this one
# ``experiment.yaml`` (there is no separate ``default.yaml``/``config.yaml`` any more); the
# ``EXPERIMENT_CONFIG`` env var lets a deployment point elsewhere. Repo root: io/ -> sdk/ ->
# coastline/ -> src/ -> repo.
_CONFIG_ENV_KEY = "EXPERIMENT_CONFIG"
_CANONICAL_CONFIG = Path(__file__).resolve().parents[4] / "config" / "coastline_functionality" / "experiment.yaml"


def default_experiment_path() -> Path:
    """The one recommendation-policy config every surface resolves to when none is given.
    The ``EXPERIMENT_CONFIG`` env var wins; else the repo's ``experiment.yaml``. The path may not
    exist (stripped wheel) — callers then fall back to :func:`builtin_default_config`."""
    override = os.environ.get(_CONFIG_ENV_KEY)
    return Path(override) if override else _CANONICAL_CONFIG


def load_runtime_workers(path: str | Path | None = None) -> Optional[int]:
    """``runtime.parallel_workers`` from the policy YAML, or None when it is not declared.

    Read separately from :func:`load_strategy_config` because the trace and batch paths build
    their config from arguments rather than from the file, yet still need the operator's worker
    count. A malformed value is ignored rather than failing a run over a preference.
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
    """The worker count a command should use: ``--workers`` wins, then the policy YAML, then 4.

    The default lives here rather than in the SDK on purpose. A command is a whole run the
    operator asked for, so spending the machine on it is what they want; a library call is a
    step inside someone else's program, where silently moving work into subprocesses would take
    their monkeypatches, their in-process state and their logging handlers away from them.
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
