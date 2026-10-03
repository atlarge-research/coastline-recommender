"""Coastline imports kavier at module level only through its public API.

Allowed at module level: ``kavier``, ``kavier.training``, ``kavier.inference`` and the
``kavier.sdk.library`` spec data. ``kavier.sdk.io`` and ``kavier.sdk.training`` may be imported
only inside functions, so a reorganisation of kavier's internals cannot break test collection.
"""

from __future__ import annotations

import ast
from pathlib import Path

# kavier internals that coastline may import only inside functions.
_FORBIDDEN = ("kavier.sdk.io", "kavier.sdk.training")
_REPO = Path(__file__).resolve().parents[2]  # repo root
# The trees to scan: the package under src/ and dev/benchmark. Tests are exempt so they can patch
# internals.
_PACKAGE_DIRS = (Path("src") / "coastline", Path("dev") / "benchmark")


def _module_level_import_targets(path: Path) -> list[str]:
    """Top-level (module-scope) imported module names in a Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    targets: list[str] = []
    for node in tree.body:  # module scope only; imports inside functions are exempt
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
        assert (_REPO / pkg).is_dir(), f"guard scans a non-existent tree {pkg} - it would be vacuous"
        for py in sorted((_REPO / pkg).rglob("*.py")):
            if "tests" in py.parts:  # tests may import internals (e.g. to patch them)
                continue
            scanned += 1
            bad = [m for m in _module_level_import_targets(py) if _is_forbidden(m)]
            if bad:
                offenders[str(py.relative_to(_REPO))] = bad
    assert scanned > 50, f"guard only scanned {scanned} files - package layout moved?"
    assert not offenders, (
        "coastline must use kavier's public API (kavier.training / kavier.sdk.library) or import "
        f"internals lazily; module-level kavier-internal imports found: {offenders}"
    )


def _targets_of_source(src: str) -> list[str]:
    """Write ``src`` to a temporary .py file and return its module-level import targets."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = Path(f.name)
    try:
        return _module_level_import_targets(path)
    finally:
        path.unlink()


def test_forbidden_matches_exact_internal_package_and_submodules():
    # A module is forbidden if it equals an entry of _FORBIDDEN or is a submodule of one.
    assert _is_forbidden("kavier.sdk.io")
    assert _is_forbidden("kavier.sdk.training")
    assert _is_forbidden("kavier.sdk.io.adapter")
    assert _is_forbidden("kavier.sdk.training.core.engine")


def test_forbidden_allows_public_api_and_prefix_siblings():
    # The public modules are allowed. kavier.sdk.iota starts with "kavier.sdk.io" but is another
    # module, so the check compares against the prefix plus ".".
    assert not _is_forbidden("kavier")
    assert not _is_forbidden("kavier.training")
    assert not _is_forbidden("kavier.inference")
    assert not _is_forbidden("kavier.sdk.library")
    assert not _is_forbidden("kavier.sdk.iota")


def test_module_level_imports_are_collected_across_both_import_forms():
    # Both import forms are collected in source order. "from . import sibling" has no module
    # name (ast gives None) and is skipped.
    src = "import kavier.sdk.training\nfrom kavier.sdk.io.adapter import export\nfrom . import sibling\nimport os\n"
    assert _targets_of_source(src) == ["kavier.sdk.training", "kavier.sdk.io.adapter", "os"]


def test_scan_pipeline_flags_module_level_offender_but_not_allowed_or_lazy():
    # The same two steps as the real scan, on a file that mixes forbidden, allowed and
    # function-level imports. Only the two module-level kavier.sdk.io imports are flagged.
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
    # Only tree.body (module scope) is read, so imports inside a function are ignored.
    # Walking the whole tree with ast.walk() would collect them.
    src = (
        "def export():\n"
        "    import kavier.sdk.training\n"
        "    from kavier.sdk.io.adapter import writer\n"
        "    return writer\n"
    )
    assert _targets_of_source(src) == []
