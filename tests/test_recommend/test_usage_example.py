"""The embedded docs example (docs/usage.py) runs end-to-end and prints the documented columns.

Two guarantees in one: the example reproduced in the docs is correct, and the whole public API path
— the callable facade, the batch DataFrame verb, and recommend_csv — works via a real subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_USAGE = Path(__file__).resolve().parents[2] / "docs" / "usage.py"


@pytest.mark.skipif(not _USAGE.exists(), reason="docs/usage.py not present")
def test_usage_example_runs_and_prints_documented_columns():
    env = {**os.environ, "COASTLINE_ALLOW_RULES_FALLBACK": "1"}
    proc = subprocess.run([sys.executable, str(_USAGE)], capture_output=True, text=True, env=env, timeout=300)
    assert proc.returncode == 0, proc.stderr

    out = proc.stdout
    # Oracle: the public output contract — the batch DataFrame's prediction column names. These are
    # the API's promise, not a snapshot of any computed number, so a rename/removal here breaks the
    # docs and this test, while a change to the predicted values leaves it green.
    for column in ("total_gpus", "throughput_tok_s", "energy_wh"):
        assert column in out, f"{column} missing from usage.py output"
    # The final (CSV) section reached its print, so all four sections ran without raising.
    assert "recommend_csv wrote" in out
