"""docs/usage.py runs end to end in a subprocess and prints the documented columns.

The example uses the callable facade, the batch DataFrame API and recommend_csv.
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
    # Check the column names of the batch output only, so a change in the predicted values
    # does not fail the test.
    for column in ("total_gpus", "throughput_tok_s", "energy_wh"):
        assert column in out, f"{column} missing from usage.py output"
    # The last section (CSV) printed, so all four sections ran.
    assert "recommend_csv wrote" in out
