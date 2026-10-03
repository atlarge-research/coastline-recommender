"""Tests for the trainer CLI: ``trainer.main`` (``--all`` / ``--model`` / ``--evaluate``
and the ``_MODEL_TRAINERS`` registry) and ``trainer.train_all``.

Nothing is trained, loaded or read from disk here; every dispatch path runs against stubs:

* ``main._run_single_model`` imports the trainer with ``importlib.import_module``, so
  a fake module placed in ``sys.modules`` is used in place of the real one.
* ``train_all`` calls ``_run_single_model`` for each registry entry, so the same fakes
  cover the ``--all`` path.
* ``setup_logging`` is patched out on ``trainer.main``.

One test checks that dispatch imports none of the heavy ML libraries.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

# Package name of the trainer; every sys.modules stub key and import target below uses it.
_PKG = "trainer"

M = importlib.import_module(f"{_PKG}.main")


# Expected registry: the 10 models in the project docs.
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

# Training-only libraries that dispatch must not import.
_HEAVY_LIBS = ("xgboost", "lightgbm", "catboost", "torch", "tabpfn")


def _seed_fake_trainer(module_name: str, attr: str, recorder: list[str]) -> types.ModuleType:
    """Install a fake ``{_PKG}.<module_name>`` whose ``attr`` appends ``module_name`` to ``recorder``.

    importlib returns the cached fake, so the real module never runs.
    """
    fq = f"{_PKG}.{module_name}"
    fake = types.ModuleType(fq)
    setattr(fake, attr, lambda *a, **k: recorder.append(module_name))
    sys.modules[fq] = fake
    return fake


@pytest.fixture()
def neutralize_side_effects(monkeypatch):
    """Stop ``main.main`` from configuring logging."""
    monkeypatch.setattr(M, "setup_logging", lambda *a, **k: None)


# Registry: name to (module, callable)


def test_registry_matches_the_ten_documented_models():
    # Compare the whole mapping: keys and (module, callable) values.
    assert M._MODEL_TRAINERS == EXPECTED_REGISTRY
    assert len(M._MODEL_TRAINERS) == 10


@pytest.mark.parametrize("name,expected", sorted(EXPECTED_REGISTRY.items()))
def test_registry_targets_exist_and_are_callable(name, expected):
    """Every (module, attr) in the registry resolves to a callable in the trainer package."""
    module_name, attr = expected
    mod = importlib.import_module(f"{_PKG}.{module_name}")
    fn = getattr(mod, attr, None)
    assert callable(fn), f"{module_name}.{attr} is not callable"


# _run_single_model: dispatch + unknown-model error


@pytest.mark.parametrize("name", ["xgboost", "lightgbm"])
def test_run_single_model_invokes_the_mapped_callable(name, monkeypatch):
    """_run_single_model imports the mapped module (a stub in sys.modules) and calls the mapped attribute once."""
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
    """tabpfn and deep_learning register ``train_tabpfn`` and ``train_deep_learning_model``.

    That callable runs, and a decoy ``train`` on the same module does not.
    """
    module_name, attr = EXPECTED_REGISTRY[name]
    assert attr != "train"  # both models register a callable other than ``train``
    hits: list[str] = []
    fake = types.ModuleType(f"{_PKG}.{module_name}")
    # Spy on the registered attribute; the decoy 'train' records "WRONG-train" if called.
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
    # The error lists the valid model names.
    for name in EXPECTED_REGISTRY:
        assert name in msg


# argparse: --all / --model / --evaluate routing (via main.main)


def test_main_requires_a_mode(monkeypatch, neutralize_side_effects):
    """Running without a mode flag is an argparse usage error with exit code 2."""
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
    # Two mode flags at once is an argparse usage error (exit code 2).
    monkeypatch.setattr(sys, "argv", ["trainer", *args])
    with pytest.raises(SystemExit) as ei:
        M.main()
    assert ei.value.code == 2


def test_main_all_dispatches_to_train_all(monkeypatch, neutralize_side_effects):
    """``--all`` imports ``.train_all`` and calls ``train_all`` once."""
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
    """``--model xgboost`` runs that trainer and calls neither train_all nor evaluate_all."""
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
    """main writes DATA_DIR to the environment; the default ``./trace-archive`` is stored as ``trace-archive``."""
    fake = types.ModuleType(f"{_PKG}.evaluate_all")
    fake.evaluate_all = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, f"{_PKG}.evaluate_all", fake)
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", ["trainer", "--evaluate"])
    M.main()
    # Path() drops the leading "./"; the expected value is written out literally.
    assert M.os.environ["DATA_DIR"] == "trace-archive"


# train_all: runs every registered trainer


def test_train_all_invokes_all_ten_mapped_trainers(monkeypatch):
    """train_all calls each of the ten registered trainers once (all stubbed)."""
    recorder: list[str] = []
    # Replace every trainer module with a spy that records its module name.
    for _name, (module_name, attr) in EXPECTED_REGISTRY.items():
        fake = types.ModuleType(f"{_PKG}.{module_name}")
        setattr(fake, attr, lambda *a, mn=module_name, **k: recorder.append(mn))
        monkeypatch.setitem(sys.modules, f"{_PKG}.{module_name}", fake)

    # Fresh import of train_all.
    monkeypatch.delitem(sys.modules, f"{_PKG}.train_all", raising=False)
    ta = importlib.import_module(f"{_PKG}.train_all")

    ta.train_all()
    # Each trainer module ran once.
    assert sorted(set(recorder)) == sorted(m for m, _ in EXPECTED_REGISTRY.values())
    assert len(recorder) == 10


def test_train_all_continues_when_a_trainer_raises(monkeypatch, capsys):
    """A failing trainer does not stop train_all, and the summary reports the failure."""
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


# Dispatch imports no heavy ML library


def test_dispatch_paths_do_not_import_heavy_libraries(monkeypatch, neutralize_side_effects):
    """Running the --model and --all paths with stubs imports none of the heavy training libraries."""
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
