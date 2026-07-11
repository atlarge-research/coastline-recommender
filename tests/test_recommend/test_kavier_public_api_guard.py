"""Guard: coastline must not import kavier's INTERNAL engines at module level — only the public
API (``kavier.training`` / ``kavier.inference``, top-level module ``kavier``) or the stable
``kavier.sdk.library`` spec-data package.

A module-level ``kavier.sdk.io`` / ``kavier.sdk.training`` import is exactly the kind of eager
engine dependency that once broke test collection when kavier reorganised its internals. This test
fails if one is reintroduced. Internals may still be imported LAZILY (inside a function) for any
path with no public verb yet — those are not module-scope, so exempt.
"""

from __future__ import annotations

import ast
from pathlib import Path

# Internal kavier engines coastline must not import at module level. The public API is
# ``kavier.training`` / ``kavier.inference`` (top-level module ``kavier``); ``kavier.sdk.library``
# is the stable spec-data package and is allowed at module level.
_FORBIDDEN = ("kavier.sdk.io", "kavier.sdk.training")
_REPO = Path(__file__).resolve().parents[2]  # coastline/ repo root
# coastline's first-party trees (tests are exempt — they may import internals to patch). The
# single package under src/ covers the SDK/CLI/UI; dev/benchmark is the other first-party module
# that talks to kavier.
_PACKAGE_DIRS = (Path("src") / "coastline", Path("dev") / "benchmark")


def _module_level_import_targets(path: Path) -> list[str]:
    """Top-level (module-scope) imported module names in a Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    targets: list[str] = []
    for node in tree.body:  # module scope only — function-level (lazy) imports are exempt
        if isinstance(node, ast.Import):
            targets += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            targets.append(node.module)
    return targets


def _is_forbidden(module: str) -> bool:
    return any(module == p or module.startswith(p + ".") for p in _FORBIDDEN)


def test_no_module_level_kavier_internal_imports():
    offenders: dict[str, list[str]] = {}
    scanned = 0
    for pkg in _PACKAGE_DIRS:
        assert (_REPO / pkg).is_dir(), f"guard scans a non-existent tree {pkg} — it would be vacuous"
        for py in sorted((_REPO / pkg).rglob("*.py")):
            if "tests" in py.parts:  # tests may import internals (e.g. to patch them)
                continue
            scanned += 1
            bad = [m for m in _module_level_import_targets(py) if _is_forbidden(m)]
            if bad:
                offenders[str(py.relative_to(_REPO))] = bad
    assert scanned > 50, f"guard only scanned {scanned} files — package layout moved?"
    assert not offenders, (
        "coastline must use kavier's PUBLIC API (kavier.training / kavier.sdk.library) or import "
        f"internals lazily; module-level kavier-internal imports found: {offenders}"
    )


def _targets_of_source(src: str) -> list[str]:
    """Arrange helper: write `src` to a temp .py file and return its module-level import
    targets. No oracle logic here — just exercises the real parser on controlled input."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = Path(f.name)
    try:
        return _module_level_import_targets(path)
    finally:
        path.unlink()


def test_forbidden_matches_exact_internal_package_and_submodules():
    # Rule: forbidden iff module == p OR module startswith p + ".".  Derived by hand from that
    # rule, in a different form (spelled-out cases) than the impl:
    #   "kavier.sdk.io"                 -> exact match on _FORBIDDEN[0]        -> True
    #   "kavier.sdk.training"           -> exact match on _FORBIDDEN[1]        -> True
    #   "kavier.sdk.io.adapter"         -> submodule of kavier.sdk.io          -> True
    #   "kavier.sdk.training.core.engine" -> submodule of kavier.sdk.training  -> True
    assert _is_forbidden("kavier.sdk.io")
    assert _is_forbidden("kavier.sdk.training")
    assert _is_forbidden("kavier.sdk.io.adapter")
    assert _is_forbidden("kavier.sdk.training.core.engine")


def test_forbidden_allows_public_api_and_prefix_siblings():
    # The allowed surface, derived from the rule's negation. None of these are `p` or a `p + "."`
    # submodule, so all must be permitted:
    #   "kavier"              -> top-level public module                       -> False
    #   "kavier.training"     -> public API verb                               -> False
    #   "kavier.inference"    -> public API verb                               -> False
    #   "kavier.sdk.library"  -> stable spec-data package (module-level OK)     -> False
    #   "kavier.sdk.iota"     -> shares the "kavier.sdk.io" prefix but is NOT a
    #                            submodule (no "." boundary) — the impl appends
    #                            "." precisely to reject this                   -> False
    assert not _is_forbidden("kavier")
    assert not _is_forbidden("kavier.training")
    assert not _is_forbidden("kavier.inference")
    assert not _is_forbidden("kavier.sdk.library")
    assert not _is_forbidden("kavier.sdk.iota")


def test_module_level_imports_are_collected_across_both_import_forms():
    # Body order, both `import` and `from ... import` forms, and a bare relative import (whose ast
    # module is None) which must be skipped without crashing. Hand-enumerated module-scope targets:
    #   import kavier.sdk.training               -> "kavier.sdk.training"
    #   from kavier.sdk.io.adapter import export -> "kavier.sdk.io.adapter"
    #   from . import sibling                    -> module is None -> dropped
    #   import os                                -> "os"
    src = "import kavier.sdk.training\nfrom kavier.sdk.io.adapter import export\nfrom . import sibling\nimport os\n"
    assert _targets_of_source(src) == ["kavier.sdk.training", "kavier.sdk.io.adapter", "os"]


def test_scan_pipeline_flags_module_level_offender_but_not_allowed_or_lazy():
    # Positive control for the composed guard: the real check is
    #   bad = [m for m in _module_level_import_targets(py) if _is_forbidden(m)].
    # Feed a file mixing forbidden, allowed, and lazy imports and hand-derive which survive both
    # stages, in source order:
    #   import kavier.sdk.io                     -> collected, forbidden (exact)   -> KEEP
    #   from kavier.training import fit          -> collected, allowed (verb)      -> drop
    #   import kavier.sdk.library                -> collected, allowed (spec data) -> drop
    #   def f(): import kavier.sdk.training      -> NOT module-scope               -> never collected
    #   from kavier.sdk.io.adapter import writer -> collected, forbidden (submod)  -> KEEP
    src = (
        "import kavier.sdk.io\n"
        "from kavier.training import fit\n"
        "import kavier.sdk.library\n"
        "def f():\n"
        "    import kavier.sdk.training\n"
        "from kavier.sdk.io.adapter import writer\n"
    )
    flagged = [m for m in _targets_of_source(src) if _is_forbidden(m)]
    assert flagged == ["kavier.sdk.io", "kavier.sdk.io.adapter"]


def test_lazy_function_level_internal_imports_are_exempt():
    # The guard's central contract: internals may be imported LAZILY (inside a function). Only
    # `tree.body` (module scope) is inspected, so a forbidden import nested in a def contributes
    # NOTHING. Falsification: switching to ast.walk() would collect these and wrongly flag every
    # lazy internal import.
    src = (
        "def export():\n"
        "    import kavier.sdk.training\n"
        "    from kavier.sdk.io.adapter import writer\n"
        "    return writer\n"
    )
    assert _targets_of_source(src) == []
