"""Unit tests for ``load_strategy_config`` in ``coastline.sdk.io.run_config``.

``load_strategy_config(path)`` loads an experiment YAML and merges it over the default
``strategy`` / ``predictors`` / ``grid`` sections that ``PolicyFactory`` reads.

Covered:
  - loading a valid config (sections pass through or merge over the defaults);
  - defaults for a missing, empty or non-file config;
  - a partial section keeps the default keys it does not set;
  - the module default is never shared with or changed by a caller.

The loader reads no environment variables; ``test_module_reads_no_environment_variables``
checks this.
"""

from __future__ import annotations

import copy
import textwrap
from pathlib import Path

import pytest
import yaml

from coastline.sdk.io.run_config import (
    _DEFAULT_STRATEGY_CONFIG,
    load_strategy_config,
)


def _write_yaml(tmp_path: Path, data, name: str = "config.yaml") -> Path:
    """Dump ``data`` (any YAML-serialisable object) to a temp file, return path."""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _write_text(tmp_path: Path, text: str, name: str = "config.yaml") -> Path:
    """Write raw YAML text (for empty-file / literal-content cases)."""
    path = tmp_path / name
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _restore_default_config():
    """Restore the contents of ``_DEFAULT_STRATEGY_CONFIG`` in place after every test.

    The loader deep-copies the default, so tests cannot change it; if that breaks, the restore
    keeps the other tests independent of order. ``TestNoGlobalMutation`` checks the isolation.
    """
    snapshot = copy.deepcopy(_DEFAULT_STRATEGY_CONFIG)
    yield
    _DEFAULT_STRATEGY_CONFIG.clear()
    _DEFAULT_STRATEGY_CONFIG.update(snapshot)


class TestValidConfig:
    def test_loads_valid_full_config(self, tmp_path):
        """A complete config is loaded and its sections honoured."""
        payload = {
            "strategy": {"name": "multi_objective", "preset": "energy_saver"},
            "predictors": {
                "performance": "kavier",
                "energy": "kavier_power",
                "feasibility": "autoconf",
            },
            "grid": {
                "batch_sizes": [8, 16],
                "total_gpus": [1, 2, 4],
                "top_k": 3,
            },
        }
        cfg = load_strategy_config(_write_yaml(tmp_path, payload))

        # The input sets both strategy keys, so the merged section equals the input.
        assert cfg["strategy"] == {"name": "multi_objective", "preset": "energy_saver"}
        assert cfg["predictors"]["performance"] == "kavier"
        assert cfg["grid"]["batch_sizes"] == [8, 16]
        assert cfg["grid"]["top_k"] == 3

    def test_accepts_str_path_as_well_as_pathlib(self, tmp_path):
        """The signature is ``str | Path``; a plain string path must work too."""
        path = _write_yaml(tmp_path, {"strategy": {"name": "min_gpu"}})
        cfg = load_strategy_config(str(path))
        assert cfg["strategy"]["name"] == "min_gpu"

    def test_strategy_partial_merge_keeps_default_preset_keys(self, tmp_path):
        """A strategy section with only ``name`` merges over the default.

        The default strategy is ``{name: multi_objective, preset: balanced}``, so
        ``preset: balanced`` is kept.
        """
        payload = {"strategy": {"name": "multi_objective"}}
        cfg = load_strategy_config(_write_yaml(tmp_path, payload))
        assert cfg["strategy"] == {"name": "multi_objective", "preset": "balanced"}

    def test_grid_partial_merge_keeps_other_default_keys(self, tmp_path):
        """Overriding one grid key keeps the remaining default grid keys."""
        payload = {"grid": {"top_k": 10}}
        cfg = load_strategy_config(_write_yaml(tmp_path, payload))
        assert cfg["grid"]["top_k"] == 10  # overridden
        # The other keys keep their defaults. The expected grid is written out here instead of
        # read from _DEFAULT_STRATEGY_CONFIG, the constant the loader copies.
        assert cfg["grid"]["batch_sizes"] == [4, 8, 16, 32, 64]
        assert cfg["grid"]["total_gpus"] == [1, 2, 4, 8, 16, 32]

    def test_non_dict_section_replaces_default_wholesale(self, tmp_path):
        """A recognised section that is not a dict replaces the default as it is, without a merge."""
        payload = {"grid": [1, 2, 3]}  # a list where a dict is expected
        cfg = load_strategy_config(_write_yaml(tmp_path, payload))
        assert cfg["grid"] == [1, 2, 3]


# Defaults for missing / empty configs
class TestDefaults:
    def test_missing_file_returns_full_defaults(self, tmp_path):
        """A missing file returns the default strategy config."""
        cfg = load_strategy_config(tmp_path / "does_not_exist.yaml")
        assert cfg["strategy"]["name"] == "multi_objective"
        assert cfg["strategy"]["preset"] == "balanced"
        assert cfg["predictors"]["performance"] == "intelligent"
        assert cfg["predictors"]["energy"] == "kavier_power"
        assert cfg["predictors"]["feasibility"] == "autoconf"
        assert cfg["grid"]["top_k"] == 5

    def test_directory_path_treated_as_missing(self, tmp_path):
        """A directory path (``is_file()`` False) returns the defaults."""
        cfg = load_strategy_config(tmp_path)
        assert cfg["strategy"]["name"] == "multi_objective"

    def test_empty_file_returns_defaults(self, tmp_path):
        """An empty YAML file (``safe_load`` returns None) gives all the defaults."""
        cfg = load_strategy_config(_write_text(tmp_path, ""))
        assert cfg["strategy"]["name"] == "multi_objective"
        assert cfg["predictors"]["performance"] == "intelligent"
        assert cfg["grid"]["batch_sizes"] == [4, 8, 16, 32, 64]

    def test_unrelated_keys_are_ignored_defaults_kept(self, tmp_path):
        """Unknown top-level keys are ignored; recognised defaults remain intact."""
        payload = {"totally_unknown": {"x": 1}, "another": 2}
        cfg = load_strategy_config(_write_yaml(tmp_path, payload))
        assert cfg["strategy"] == {"name": "multi_objective", "preset": "balanced"}
        assert "totally_unknown" not in cfg


# Environment variables: the loader reads none
class TestEnvScoping:
    def test_module_reads_no_environment_variables(self, tmp_path, monkeypatch):
        """Setting CONFIG_FILE, RUN_ID and DATA_DIR leaves the loader's output for a file unchanged."""
        payload = {"strategy": {"name": "multi_objective"}}
        path = _write_yaml(tmp_path, payload)
        baseline = load_strategy_config(path)

        for var in ("CONFIG_FILE", "RUN_ID", "DATA_DIR"):
            monkeypatch.setenv(var, "/some/override/value")
        after = load_strategy_config(path)

        # strategy.name comes from the file and preset from the default.
        assert baseline["strategy"] == {"name": "multi_objective", "preset": "balanced"}
        # The environment variables leave the output unchanged.
        assert after == baseline


# Isolation of the module default
class TestNoGlobalMutation:
    def test_loading_does_not_mutate_module_default_constant(self, tmp_path):
        """Loading a config leaves ``_DEFAULT_STRATEGY_CONFIG`` unchanged."""
        snapshot = copy.deepcopy(_DEFAULT_STRATEGY_CONFIG)
        load_strategy_config(_write_yaml(tmp_path, {"grid": {"top_k": 99}, "predictors": {"performance": "kavier"}}))
        assert _DEFAULT_STRATEGY_CONFIG == snapshot

    def test_mutating_returned_default_must_not_leak_into_module_constant(self, tmp_path):
        """Changing a section of an all-default result leaves the module default unchanged.

        ``load_strategy_config`` deep-copies ``_DEFAULT_STRATEGY_CONFIG``, so the returned
        nested dicts are separate objects.
        """
        snapshot = copy.deepcopy(_DEFAULT_STRATEGY_CONFIG)
        cfg = load_strategy_config(tmp_path / "missing.yaml")  # all-default path

        # The returned nested dict is a distinct object from the module default.
        assert cfg["grid"] is not _DEFAULT_STRATEGY_CONFIG["grid"]

        # Mutate a nested value of the returned config.
        cfg["grid"]["top_k"] = 123456

        # The module-level default must be unaffected.
        assert _DEFAULT_STRATEGY_CONFIG == snapshot

    def test_two_loads_are_independent_objects(self, tmp_path):
        """Two loads of a missing file share no nested dicts, so changing one leaves the other as is."""
        path = tmp_path / "missing.yaml"
        a = load_strategy_config(path)
        b = load_strategy_config(path)
        # Distinct top-level dicts and distinct nested dicts.
        assert a is not b
        assert a["grid"] is not b["grid"]
        a["grid"]["top_k"] = -999
        assert b["grid"]["top_k"] != -999

    def test_full_config_returns_fresh_nested_objects(self, tmp_path):
        """A supplied section is merged into a new dict (``_merge_dict``), so changing it leaves
        the module default unchanged."""
        cfg = load_strategy_config(_write_yaml(tmp_path, {"grid": {"top_k": 7}}))
        assert cfg["grid"] is not _DEFAULT_STRATEGY_CONFIG["grid"]
        before = _DEFAULT_STRATEGY_CONFIG["grid"]["top_k"]
        cfg["grid"]["top_k"] = -1
        assert _DEFAULT_STRATEGY_CONFIG["grid"]["top_k"] == before
