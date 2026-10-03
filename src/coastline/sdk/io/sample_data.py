"""Data files shipped inside the package, so an installed wheel has them without a repository
checkout: the 5-job raw-trace sample (cache warm-up; set DATA_DIR for the full trace) and the
small lookup database that ``lookup: default`` selects."""

from pathlib import Path

SAMPLE_RAW_TRACE = Path(__file__).resolve().parent / "data" / "sample_raw_trace.csv"
DEFAULT_RUN_DATABASE = Path(__file__).resolve().parent / "data" / "run_database.csv"


def sample_raw_trace_path() -> Path:
    """Absolute path to the bundled 5-job raw-trace sample."""
    return SAMPLE_RAW_TRACE


def default_run_database_path() -> Path:
    """Absolute path to the bundled lookup database that ``lookup: default`` selects."""
    return DEFAULT_RUN_DATABASE
