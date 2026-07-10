"""Bundled data files shipped with the package.

* ``sample_raw_trace.csv`` — 5 synthetic jobs (cache warm-up / hermetic test fixture).
* ``default_lookup.csv`` — a small measured-runs lookup DB sampled from the ado
  sfttrainer profiling dataset, perf values jittered by ±0–3% so no exact benchmark
  numbers ship. Selected by ``lookup: default`` (config) or ``--lookup default`` (CLI).
"""

from pathlib import Path

_DATA = Path(__file__).resolve().parent / "data"

SAMPLE_RAW_TRACE = _DATA / "sample_raw_trace.csv"
DEFAULT_LOOKUP = _DATA / "default_lookup.csv"


def sample_raw_trace_path() -> Path:
    """Absolute path to the bundled 5-job raw-trace sample."""
    return SAMPLE_RAW_TRACE


def default_lookup_path() -> Path:
    """Absolute path to the bundled (jittered) sfttrainer lookup database."""
    return DEFAULT_LOOKUP
