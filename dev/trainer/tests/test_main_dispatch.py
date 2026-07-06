"""CLI dispatch / orchestration for the ML trainer entry points.

Covers ``…trainer.main`` (the ``--all`` / ``--model`` / ``--evaluate`` arg
routing and the ``_MODEL_TRAINERS`` registry) and ``…trainer.train_all``
(the model-name -> module mapping it imports).

NO real training, model loading, or data reading happens here. The actual
per-model train functions are heavyweight (xgboost / torch / sklearn, and they
read the curated CSV), so every dispatch path is exercised against *stubs*:

* For the single-``--model`` path, ``main._run_single_model`` resolves the
  trainer with ``importlib.import_module(".<mod>", package=__package__)`` and
  ``getattr(mod, attr)``. We pre-seed ``sys.modules[f'{_PKG}.<mod>']`` with a
  fake module whose callable is a spy, so importlib returns the cached fake and
  the real (heavy) module is never executed.
* For the ``--all`` path, ``train_all`` binds the train functions as
  module-level names at import time. We pre-seed the underlying
  ``…trainer.train_performance_*`` modules with fakes *before* importing
  ``…trainer.train_all`` so those top-level imports resolve to spies.
* ``main.main`` also calls ``load_config`` / ``setup_logging`` and dispatches
  to ``train_all`` / ``evaluate_all``; those are monkeypatched on the
  ``…trainer.main`` namespace.

A guard test asserts none of the heavy ML libraries are imported as a side
effect, so a regression that eagerly imports them (or actually trains) is
caught.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

# The trainer package now lives under recommender/predictors/performance/data_driven/.
# Its submodules resolve under this fully-qualified package path, so every
# sys.modules stub key / import target below is built from it.
_PKG = "trainer"

M = importlib.import_module(f"{_PKG}.main")


# The mapping the trainer is contractually expected to expose. Mirrors the
# 10 models documented in the project docs (registry in the trainer's main.py).
EXPECTED_REGISTRY: dict[str, tuple[str, str]] = {
    "xgboost": ("train_performance_xgboost", "train"),
    "lightgbm": ("train_performance_lightgbm", "train"),
    "catboost": ("train_performance_catboost", "train"),
    "random_forest": ("train_performance_random_forest", "train"),
    "svr": ("train_performance_svr", "train"),
    "knn": ("train_performance_knn", "train"),
    "gaussian_process": ("train_performance_gaussian_process", "train"),
    "bayesian_ridge": ("train_performance_bayesian_ridge", "train"),
    "tabpfn": ("train_performance_tabpfn", "train_tabpfn"),
    "deep_learning": ("train_performance_deep_learning", "train_deep_learning_model"),
}

# Heavy / training-only libraries that must NOT be pulled in by mere dispatch.
_HEAVY_LIBS = ("xgboost", "lightgbm", "catboost", "torch", "tabpfn")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_fake_trainer(module_name: str, attr: str, recorder: list[str]) -> types.ModuleType:
    """Install a fake ``{_PKG}.<module_name>`` whose ``attr`` is a spy.

    importlib returns the sys.modules-cached object, so the real heavy module
    is never imported/executed. The spy appends ``module_name`` to ``recorder``
    when called.
    """
    fq = f"{_PKG}.{module_name}"
    fake = types.ModuleType(fq)
    setattr(fake, attr, lambda *a, **k: recorder.append(module_name))
    sys.modules[fq] = fake
    return fake


@pytest.fixture()
def neutralize_side_effects(monkeypatch):
    """Stop ``main.main`` from configuring logging or loading real config."""
    monkeypatch.setattr(M, "setup_logging", lambda *a, **k: None)


# ===========================================================================
# Registry: name -> (module, callable)
# ===========================================================================


def test_registry_matches_the_ten_documented_models():
    # Whole-mapping equality (keys + (module, callable) values) in one shot —
    # replaces the per-model parametrized mirror of the same literal.
    assert M._MODEL_TRAINERS == EXPECTED_REGISTRY
    assert len(M._MODEL_TRAINERS) == 10


@pytest.mark.parametrize("name,expected", sorted(EXPECTED_REGISTRY.items()))
def test_registry_targets_exist_and_are_callable(name, expected):
    """Every (module, attr) the registry points at must resolve to a real
    callable in the actual trainer package (no dangling references)."""
    module_name, attr = expected
    mod = importlib.import_module(f"{_PKG}.{module_name}")
    fn = getattr(mod, attr, None)
    assert callable(fn), f"{module_name}.{attr} is not callable"


# ===========================================================================
# _run_single_model: dispatch + unknown-model error
# ===========================================================================


@pytest.mark.parametrize("name", ["xgboost", "lightgbm"])
def test_run_single_model_invokes_the_mapped_callable(name, monkeypatch):
    """_run_single_model imports the mapped module and calls the mapped attr.

    The module is stubbed via sys.modules, so the spy — not the real trainer —
    runs, and exactly once. (The tabpfn/deep_learning non-``train`` attrs are
    covered separately by test_run_single_model_does_not_call_the_wrong_attr.)
    """
    module_name, attr = EXPECTED_REGISTRY[name]
    calls: list[str] = []
    # monkeypatch.setitem auto-restores sys.modules after the test.
    fake = types.ModuleType(f"{_PKG}.{module_name}")
    setattr(fake, attr, lambda *a, **k: calls.append(name))
    monkeypatch.setitem(sys.modules, f"{_PKG}.{module_name}", fake)

    M._run_single_model(name)
    assert calls == [name]


@pytest.mark.parametrize("name", ["tabpfn", "deep_learning"])
def test_run_single_model_does_not_call_the_wrong_attr(name, monkeypatch):
    """tabpfn maps to ``train_tabpfn`` (not ``train``); deep_learning maps to
    ``train_deep_learning_model``. Verify the *named* attr is what gets called.

    Oracle: the registry's declared attr (a non-``train`` name for both of these
    models) is the one invoked; a decoy ``train`` attr on the same module must
    stay silent. A bug that hard-coded ``getattr(mod, "train")`` would fire the
    decoy and record "WRONG-train" instead of the model name.
    """
    module_name, attr = EXPECTED_REGISTRY[name]
    assert attr != "train"  # guard: these two models are the non-``train`` cases
    hits: list[str] = []
    fake = types.ModuleType(f"{_PKG}.{module_name}")
    # Wire the correct attr to a spy and a decoy 'train' that must NOT fire.
    setattr(fake, attr, lambda *a, n=name, **k: hits.append(n))
    fake.train = lambda *a, **k: hits.append("WRONG-train")
    monkeypatch.setitem(sys.modules, f"{_PKG}.{module_name}", fake)
    M._run_single_model(name)
    assert hits == [name]


def test_run_single_model_unknown_raises_systemexit_listing_valid():
    with pytest.raises(SystemExit) as ei:
        M._run_single_model("does_not_exist")
    msg = str(ei.value)
    assert "does_not_exist" in msg
    # Error lists the valid models so the user can self-correct.
    for name in EXPECTED_REGISTRY:
        assert name in msg


# ===========================================================================
# argparse: --all / --model / --evaluate routing (via main.main)
# ===========================================================================


def test_main_requires_a_mode(monkeypatch, neutralize_side_effects):
    """The mode group is required=True -> bare invocation is an argparse usage
    error. argparse.error() exits with the POSIX usage-error code 2 (not just
    any non-zero), so pin the exact code — a body that swallowed the missing
    mode and returned 0, or exited 1, would fail."""
    monkeypatch.setattr(sys, "argv", ["trainer"])
    with pytest.raises(SystemExit) as ei:
        M.main()
    assert ei.value.code == 2


@pytest.mark.parametrize(
    "args",
    [
        ["--all", "--evaluate"],
        ["--all", "--model", "xgboost"],
        ["--model", "xgboost", "--evaluate"],
    ],
)
def test_main_modes_are_mutually_exclusive(monkeypatch, neutralize_side_effects, args):
    # Two mode flags at once is an argparse usage error -> exit code 2 (the
    # POSIX usage-error code argparse.error uses), not merely non-zero.
    monkeypatch.setattr(sys, "argv", ["trainer", *args])
    with pytest.raises(SystemExit) as ei:
        M.main()
    assert ei.value.code == 2


def test_main_all_dispatches_to_train_all(monkeypatch, neutralize_side_effects):
    """``--all`` imports ``.train_all`` and calls ``train_all`` exactly once."""
    called: list[str] = []
    fake = types.ModuleType(f"{_PKG}.train_all")
    fake.train_all = lambda *a, **k: called.append("train_all")
    monkeypatch.setitem(sys.modules, f"{_PKG}.train_all", fake)

    monkeypatch.setattr(sys, "argv", ["trainer", "--all"])
    M.main()
    assert called == ["train_all"]


def test_main_evaluate_dispatches_to_evaluate_all(monkeypatch, neutralize_side_effects):
    """``--evaluate`` imports ``.evaluate_all`` and calls ``evaluate_all``."""
    called: list[str] = []
    fake = types.ModuleType(f"{_PKG}.evaluate_all")
    fake.evaluate_all = lambda *a, **k: called.append("evaluate_all")
    monkeypatch.setitem(sys.modules, f"{_PKG}.evaluate_all", fake)

    monkeypatch.setattr(sys, "argv", ["trainer", "--evaluate"])
    M.main()
    assert called == ["evaluate_all"]


def test_main_model_routes_to_single_model_only(monkeypatch, neutralize_side_effects):
    """``--model xgboost`` runs that one trainer and neither train_all nor
    evaluate_all is touched."""
    single: list[str] = []
    forbidden: list[str] = []
    # Stub the resolved trainer module for xgboost.
    fake_xgb = types.ModuleType(f"{_PKG}.train_performance_xgboost")
    fake_xgb.train = lambda *a, **k: single.append("xgboost")
    monkeypatch.setitem(sys.modules, f"{_PKG}.train_performance_xgboost", fake_xgb)
    # Tripwires on the other two dispatch targets.
    fake_all = types.ModuleType(f"{_PKG}.train_all")
    fake_all.train_all = lambda *a, **k: forbidden.append("train_all")
    fake_eval = types.ModuleType(f"{_PKG}.evaluate_all")
    fake_eval.evaluate_all = lambda *a, **k: forbidden.append("evaluate_all")
    monkeypatch.setitem(sys.modules, f"{_PKG}.train_all", fake_all)
    monkeypatch.setitem(sys.modules, f"{_PKG}.evaluate_all", fake_eval)

    monkeypatch.setattr(sys, "argv", ["trainer", "--model", "xgboost"])
    M.main()
    assert single == ["xgboost"]
    assert forbidden == []


def test_main_model_unknown_propagates_systemexit(monkeypatch, neutralize_side_effects):
    monkeypatch.setattr(sys, "argv", ["trainer", "--model", "not_a_model"])
    with pytest.raises(SystemExit) as ei:
        M.main()
    assert "not_a_model" in str(ei.value)


def test_main_sets_data_dir_env(monkeypatch, neutralize_side_effects):
    """main normalizes DATA_DIR into the environment for downstream loaders.

    The default ``./trace-archive`` is passed through ``Path``, which strips the
    leading ``./`` -> the stored value is ``trace-archive``.
    """
    fake = types.ModuleType(f"{_PKG}.evaluate_all")
    fake.evaluate_all = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, f"{_PKG}.evaluate_all", fake)
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", ["trainer", "--evaluate"])
    M.main()
    # Oracle hand-derived (not via Path() — that would re-run the impl's own
    # normalization): the default "./trace-archive" run through Path() drops the
    # redundant leading "./", so the stored string is exactly "trace-archive".
    assert M.os.environ["DATA_DIR"] == "trace-archive"


# ===========================================================================
# train_all: model-name -> module mapping it imports
# ===========================================================================


def test_train_all_invokes_all_ten_mapped_trainers(monkeypatch):
    """Pre-seed every underlying ``…trainer.train_performance_*`` module with a
    spy, (re)import ``…trainer.train_all``, run it, and confirm all ten distinct
    trainers fire. No heavy import / real training occurs.
    """
    recorder: list[str] = []
    # Seed every underlying trainer module with a spy keyed on its module name.
    for _name, (module_name, attr) in EXPECTED_REGISTRY.items():
        fake = types.ModuleType(f"{_PKG}.{module_name}")
        setattr(fake, attr, lambda *a, mn=module_name, **k: recorder.append(mn))
        monkeypatch.setitem(sys.modules, f"{_PKG}.{module_name}", fake)

    # Force a fresh import of train_all so its top-level imports bind our spies.
    monkeypatch.delitem(sys.modules, f"{_PKG}.train_all", raising=False)
    ta = importlib.import_module(f"{_PKG}.train_all")

    ta.train_all()
    # Every distinct underlying trainer module was invoked exactly once.
    assert sorted(set(recorder)) == sorted(m for m, _ in EXPECTED_REGISTRY.values())
    assert len(recorder) == 10


def test_train_all_continues_when_a_trainer_raises(monkeypatch, capsys):
    """train_all wraps each trainer in try/except: one failing model must not
    abort the rest, and the summary records the failure."""
    recorder: list[str] = []
    for _name, (module_name, attr) in EXPECTED_REGISTRY.items():
        fake = types.ModuleType(f"{_PKG}.{module_name}")
        if module_name == "train_performance_svr":
            setattr(fake, attr, lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        else:
            setattr(fake, attr, lambda *a, mn=module_name, **k: recorder.append(mn))
        monkeypatch.setitem(sys.modules, f"{_PKG}.{module_name}", fake)
    monkeypatch.delitem(sys.modules, f"{_PKG}.train_all", raising=False)
    ta = importlib.import_module(f"{_PKG}.train_all")

    ta.train_all()  # must not raise
    out = capsys.readouterr().out
    assert "Failed" in out  # the SVR failure is reported
    # The other nine still ran.
    assert len(recorder) == 9


# ===========================================================================
# Guard: dispatch must not import heavy ML libraries / train for real
# ===========================================================================


def test_dispatch_paths_do_not_import_heavy_libraries(monkeypatch, neutralize_side_effects):
    """Drive the --model and --all paths with stubs; assert none of the heavy
    training libraries got imported. Catches a regression that eagerly imports
    them (or actually trains a model)."""
    already = {lib for lib in _HEAVY_LIBS if lib in sys.modules}

    # --model path (stubbed xgboost trainer).
    fake_xgb = types.ModuleType(f"{_PKG}.train_performance_xgboost")
    fake_xgb.train = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, f"{_PKG}.train_performance_xgboost", fake_xgb)
    monkeypatch.setattr(sys, "argv", ["trainer", "--model", "xgboost"])
    M.main()

    # --all path (stub every underlying trainer + a fresh train_all import).
    for _name, (module_name, attr) in EXPECTED_REGISTRY.items():
        fake = types.ModuleType(f"{_PKG}.{module_name}")
        setattr(fake, attr, lambda *a, **k: None)
        monkeypatch.setitem(sys.modules, f"{_PKG}.{module_name}", fake)
    monkeypatch.delitem(sys.modules, f"{_PKG}.train_all", raising=False)
    monkeypatch.setattr(sys, "argv", ["trainer", "--all"])
    M.main()

    newly = {lib for lib in _HEAVY_LIBS if lib in sys.modules} - already
    assert not newly, f"dispatch unexpectedly imported heavy libs: {sorted(newly)}"
