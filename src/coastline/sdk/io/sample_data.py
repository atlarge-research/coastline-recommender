"""Bundled 5-job trace sample shipped with the package (cache warm-up; set DATA_DIR for full trace)."""

from pathlib import Path

SAMPLE_RAW_TRACE = Path(__file__).resolve().parent / "data" / "sample_raw_trace.csv"


def sample_raw_trace_path() -> Path:
    """Absolute path to the bundled 5-job raw-trace sample."""
    return SAMPLE_RAW_TRACE
