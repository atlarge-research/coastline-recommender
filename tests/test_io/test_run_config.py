"""
Unit tests for ``coastline/run_config.py``.

The single public function under test is ``load_strategy_config(path)``: it loads
an experiment YAML for the CLI and normalises it into the ``strategy`` /
``predictors`` / ``grid`` / ``output`` shape the ``PolicyFactory`` consumes,
including translating the *legacy* ``orchestrator:`` section into the new
``predictors:`` section.

What is covered:
  - loading a valid config (new-style sections pass through / shallow-merge);
  - the legacy old -> new mapping, in particular
    ``orchestrator.predictor: cache_first`` -> ``predictors.performance:
    intelligent`` (plus ``physics``/``physics_driven`` -> ``kavier``,
    ``ensemble`` -> ``intelligent``, and unknown values passed through);
  - defaults for missing/empty/non-file configs;
  - per-section shallow merge keeps untouched default keys.

Env-var overrides: ``run_config.py`` itself reads **no** environment variables.
The only env handling (``CONFIG_FILE``/``RUN_ID``/``DATA_DIR``) lives in
``coastline_recommender/cli.py``, *outside* the unit under test. We pin that scoping
explicitly (``test_module_reads_no_environment_variables``) rather than invent
override behaviour the loader does not have.

These are NEW tests; no production code is modified.

Run:
  cd <repo> && PYTHONPATH=coastline:coastline/common:kavier/src \
    DATA_DIR=./trace-archive .venv/bin/python -m pytest \
    coastline/tests/test_run_config.py -q
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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
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
    """Defensive snapshot/restore of ``_DEFAULT_STRATEGY_CONFIG`` around every test.

    ``load_strategy_config`` currently ``copy.deepcopy``-es the default before use,
    so no test *should* be able to mutate the shared module constant. This autouse
    fixture is a safety net: should that isolation ever regress (a return to a
    shallow copy), it keeps the suite order-independent by restoring the constant's
    contents in place (so any held reference also sees pristine values) instead of
    letting one test poison the rest. The ``TestNoGlobalMutation`` cases assert the
    isolation invariant directly.
    """
    snapshot = copy.deepcopy(_DEFAULT_STRATEGY_CONFIG)
    yield
    _DEFAULT_STRATEGY_CONFIG.clear()
    _DEFAULT_STRATEGY_CONFIG.update(snapshot)


# ===========================================================================
# Valid config: new-style sections
# ===========================================================================
class TestValidNewStyleConfig:
    def test_loads_valid_full_config(self, tmp_path):
        """A complete new-style config is loaded and its sections honoured."""
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

        # strategy: input supplies BOTH keys, so the shallow-merge over the default
        # {name: min_gpu, preset: balanced} is fully overridden -> equals the input.
        assert cfg["strategy"] == {"name": "multi_objective", "preset": "energy_saver"}
        assert cfg["predictors"]["performance"] == "kavier"
        assert cfg["grid"]["batch_sizes"] == [8, 16]
        assert cfg["grid"]["top_k"] == 3

    def test_accepts_str_path_as_well_as_pathlib(self, tmp_path):
        """The signature is ``str | Path``; a plain string path must work too."""
        path = _write_yaml(tmp_path, {"strategy": {"name": "min_gpu"}})
        cfg = load_strategy_config(str(path))
        assert cfg["strategy"]["name"] == "min_gpu"

    def test_output_section_passes_through(self, tmp_path):
        """The ``output`` section is one of the recognised sections and is kept."""
        payload = {"output": {"dir": "/tmp/out", "format": "json"}}
        cfg = load_strategy_config(_write_yaml(tmp_path, payload))
        assert cfg["output"] == {"dir": "/tmp/out", "format": "json"}

    def test_strategy_partial_merge_keeps_default_preset_keys(self, tmp_path):
        """A strategy section with only ``name`` shallow-merges over the default.

        The default strategy is ``{name: multi_objective, preset: balanced}``; supplying
        only ``name`` must retain ``preset: balanced`` from the default.
        """
        payload = {"strategy": {"name": "multi_objective"}}
        cfg = load_strategy_config(_write_yaml(tmp_path, payload))
        assert cfg["strategy"] == {"name": "multi_objective", "preset": "balanced"}

    def test_grid_partial_merge_keeps_other_default_keys(self, tmp_path):
        """Overriding one grid key keeps the remaining default grid keys."""
        payload = {"grid": {"top_k": 10}}
        cfg = load_strategy_config(_write_yaml(tmp_path, payload))
        assert cfg["grid"]["top_k"] == 10  # overridden
        # Untouched keys survive the shallow merge. Oracle = the documented default
        # grid (spec literals), NOT a read-back of _DEFAULT_STRATEGY_CONFIG — pinning
        # the literals keeps this independent of the very constant the loader copies.
        assert cfg["grid"]["batch_sizes"] == [4, 8, 16, 32, 64]
        assert cfg["grid"]["total_gpus"] == [1, 2, 4, 8, 16]

    def test_non_dict_section_replaces_default_wholesale(self, tmp_path):
        """If a recognised section is not a dict, it replaces the default as-is.

        The loader's branch: ``isinstance(loaded[section], dict)`` is False, so
        the scalar/list value is assigned directly (no merge attempted).
        """
        payload = {"grid": [1, 2, 3]}  # deliberately a list, not a dict
        cfg = load_strategy_config(_write_yaml(tmp_path, payload))
        assert cfg["grid"] == [1, 2, 3]


# ===========================================================================
# Legacy orchestrator: -> new predictors: translation
# ===========================================================================
class TestLegacyOrchestratorMapping:
    @pytest.mark.parametrize(
        ("predictor_value", "expected_performance"),
        [
            ("cache_first", "intelligent"),  # the headline mapping
            ("physics", "kavier"),
            ("physics_driven", "kavier"),
            ("ensemble", "intelligent"),
            ("intelligent", "intelligent"),  # already-new value is preserved
        ],
    )
    def test_orchestrator_predictor_maps_to_performance(self, tmp_path, predictor_value, expected_performance):
        """Legacy ``orchestrator.predictor`` is translated to ``predictors.performance``."""
        payload = {"orchestrator": {"predictor": predictor_value}}
        cfg = load_strategy_config(_write_yaml(tmp_path, payload))
        assert cfg["predictors"]["performance"] == expected_performance

    def test_unknown_orchestrator_predictor_passes_through(self, tmp_path):
        """An unrecognised predictor name is passed through unchanged (no mapping)."""
        payload = {"orchestrator": {"predictor": "some_future_model"}}
        cfg = load_strategy_config(_write_yaml(tmp_path, payload))
        assert cfg["predictors"]["performance"] == "some_future_model"

    def test_orchestrator_without_predictor_defaults_to_intelligent(self, tmp_path):
        """Missing ``orchestrator.predictor`` defaults performance to ``intelligent``."""
        payload = {"orchestrator": {"energy": "kavier_power"}}
        cfg = load_strategy_config(_write_yaml(tmp_path, payload))
        assert cfg["predictors"]["performance"] == "intelligent"

    def test_orchestrator_energy_and_feasibility_carried_over(self, tmp_path):
        """``orchestrator.energy``/``feasibility`` populate the predictors section."""
        payload = {
            "orchestrator": {
                "predictor": "cache_first",
                "energy": "custom_energy",
                "feasibility": "memory_aware",
            }
        }
        cfg = load_strategy_config(_write_yaml(tmp_path, payload))
        assert cfg["predictors"]["energy"] == "custom_energy"
        assert cfg["predictors"]["feasibility"] == "memory_aware"

    def test_orchestrator_missing_energy_feasibility_use_defaults(self, tmp_path):
        """Absent orchestrator energy/feasibility fall back to documented defaults."""
        payload = {"orchestrator": {"predictor": "physics"}}
        cfg = load_strategy_config(_write_yaml(tmp_path, payload))
        assert cfg["predictors"]["energy"] == "kavier_power"
        assert cfg["predictors"]["feasibility"] == "autoconf"

    def test_empty_orchestrator_section_yields_all_defaults(self, tmp_path):
        """``orchestrator:`` present but null/empty still produces default predictors."""
        cfg = load_strategy_config(_write_text(tmp_path, "orchestrator:\n"))
        assert cfg["predictors"] == {
            "performance": "intelligent",
            "energy": "kavier_power",
            "feasibility": "autoconf",
        }

    def test_strategy_section_coexists_with_legacy_orchestrator(self, tmp_path):
        """A new ``strategy`` section and a legacy ``orchestrator`` section combine.

        ``strategy`` is taken from its own section while ``predictors`` is derived
        from the legacy ``orchestrator`` block.
        """
        payload = {
            "strategy": {"name": "multi_objective", "preset": "energy_saver"},
            "orchestrator": {"predictor": "physics_driven"},
        }
        cfg = load_strategy_config(_write_yaml(tmp_path, payload))
        assert cfg["strategy"] == {"name": "multi_objective", "preset": "energy_saver"}
        assert cfg["predictors"]["performance"] == "kavier"

    def test_explicit_predictors_win_over_leftover_orchestrator(self, tmp_path):
        """When both a new ``predictors`` block and a legacy ``orchestrator`` block
        are present, the explicit predictors section wins (mirrors ``api/main.py``:
        the orchestrator translation only applies when ``predictors`` is absent).
        """
        payload = {
            "predictors": {"performance": "kavier", "energy": "kavier_power"},
            "orchestrator": {"predictor": "cache_first"},  # ignored: predictors present
        }
        cfg = load_strategy_config(_write_yaml(tmp_path, payload))
        assert cfg["predictors"]["performance"] == "kavier"

    def test_explicit_predictors_not_replaced_by_orchestrator_defaults(self, tmp_path):
        """Regression (issue #5): an explicit predictors block must survive a
        leftover orchestrator block untouched — previously the translation ran
        unconditionally and silently replaced user predictors with
        ``{performance: intelligent, energy: kavier_power, feasibility: autoconf}``.
        """
        payload = {
            "predictors": {"performance": "catboost", "feasibility": "rules"},
            "orchestrator": {"predictor": "cache_first", "feasibility": "autoconf"},
        }
        cfg = load_strategy_config(_write_yaml(tmp_path, payload))
        assert cfg["predictors"]["performance"] == "catboost"
        assert cfg["predictors"]["feasibility"] == "rules"
        # keys the user did not set still come from the documented defaults
        assert cfg["predictors"]["energy"] == "kavier_power"

    def test_orchestrator_still_translated_when_no_predictors_section(self, tmp_path):
        """Sanity: without a ``predictors`` section the legacy translation applies."""
        payload = {"orchestrator": {"predictor": "cache_first", "feasibility": "none"}}
        cfg = load_strategy_config(_write_yaml(tmp_path, payload))
        assert cfg["predictors"]["performance"] == "intelligent"
        assert cfg["predictors"]["feasibility"] == "none"


# ===========================================================================
# Defaults for missing / empty configs
# ===========================================================================
class TestDefaults:
    def test_missing_file_returns_full_defaults(self, tmp_path):
        """A non-existent path returns the one default strategy config (no crash)."""
        cfg = load_strategy_config(tmp_path / "does_not_exist.yaml")
        assert cfg["strategy"]["name"] == "multi_objective"
        assert cfg["strategy"]["preset"] == "balanced"
        assert cfg["predictors"]["performance"] == "intelligent"
        assert cfg["predictors"]["energy"] == "kavier_power"
        assert cfg["predictors"]["feasibility"] == "autoconf"
        assert cfg["grid"]["top_k"] == 5

    def test_directory_path_treated_as_missing(self, tmp_path):
        """A path that is a directory (``is_file()`` False) yields defaults, not an error."""
        cfg = load_strategy_config(tmp_path)
        assert cfg["strategy"]["name"] == "multi_objective"

    def test_empty_file_returns_defaults(self, tmp_path):
        """An empty YAML file (``safe_load`` -> None) falls back to all defaults."""
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


# ===========================================================================
# Env vars: out of scope for this module (scoping pin, not invented behaviour)
# ===========================================================================
class TestEnvScoping:
    def test_module_reads_no_environment_variables(self, tmp_path, monkeypatch):
        """``load_strategy_config`` is env-agnostic.

        The CLI env vars (CONFIG_FILE/RUN_ID/DATA_DIR) are consumed by
        ``coastline_recommender/cli.py``, not by the loader. Setting them must not change
        the loader's output for a given file. This documents the boundary instead
        of asserting an override the loader does not implement.
        """
        payload = {"strategy": {"name": "multi_objective"}}
        path = _write_yaml(tmp_path, payload)
        baseline = load_strategy_config(path)

        for var in ("CONFIG_FILE", "RUN_ID", "DATA_DIR"):
            monkeypatch.setenv(var, "/some/override/value")
        after = load_strategy_config(path)

        # Content oracle first: the file (not env, not a hard-coded constant) drives
        # the result -- strategy.name comes from the payload, preset from the default.
        assert baseline["strategy"] == {"name": "multi_objective", "preset": "balanced"}
        # Invariant: setting the CLI env vars leaves the loader's output unchanged.
        assert after == baseline


# ===========================================================================
# State-isolation regression (BUG): shallow copy of the module default
# ===========================================================================
class TestNoGlobalMutation:
    def test_loading_does_not_mutate_module_default_constant(self, tmp_path):
        """Loading any config must never mutate ``_DEFAULT_STRATEGY_CONFIG``.

        Normal-use guard: even with full configs the shared module constant must
        stay pristine across calls.
        """
        snapshot = copy.deepcopy(_DEFAULT_STRATEGY_CONFIG)
        load_strategy_config(_write_yaml(tmp_path, {"grid": {"top_k": 99}, "orchestrator": {"predictor": "physics"}}))
        assert _DEFAULT_STRATEGY_CONFIG == snapshot

    def test_mutating_returned_default_must_not_leak_into_module_constant(self, tmp_path):
        """A caller mutating a returned (defaulted) section must not poison globals.

        ``load_strategy_config`` now ``copy.deepcopy``-es ``_DEFAULT_STRATEGY_CONFIG``,
        so even on the all-default path the returned nested dicts are independent of
        the module constant. Mutating the result must NOT change the default.
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
        """Two separate loads must not share mutable nested state.

        With the deep copy in place, a missing-file load no longer aliases the
        module constant, so two such results are fully independent down to their
        nested dicts: mutating one does not affect the other.
        """
        path = tmp_path / "missing.yaml"
        a = load_strategy_config(path)
        b = load_strategy_config(path)
        # Distinct top-level dicts and distinct nested dicts.
        assert a is not b
        assert a["grid"] is not b["grid"]
        a["grid"]["top_k"] = -999
        assert b["grid"]["top_k"] != -999

    def test_full_config_returns_fresh_nested_objects(self, tmp_path):
        """When a section IS supplied, the merge yields fresh dicts (no aliasing).

        This is the safe path: ``_merge_dict`` builds a new dict, so the returned
        ``grid`` is a different object from the module default and can be mutated
        freely. Pins that the bug is confined to the all-default path.
        """
        cfg = load_strategy_config(_write_yaml(tmp_path, {"grid": {"top_k": 7}}))
        assert cfg["grid"] is not _DEFAULT_STRATEGY_CONFIG["grid"]
        before = _DEFAULT_STRATEGY_CONFIG["grid"]["top_k"]
        cfg["grid"]["top_k"] = -1
        assert _DEFAULT_STRATEGY_CONFIG["grid"]["top_k"] == before
