"""Centralized logging setup. Call ``setup_logging()`` once at application start."""

import logging
import sys

_NOISY_LOGGERS = ["xgboost", "lightgbm", "sklearn", "torch", "tabpfn", "urllib3", "werkzeug"]


def setup_logging(
    level: str = "INFO",
    format_string: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    date_format: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """Configure the root logger and quiet noisy third-party loggers."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        datefmt=date_format,
        stream=sys.stdout,
        force=True,
    )

    for noisy_logger in _NOISY_LOGGERS:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    logging.getLogger(__name__).debug("Logging configured at %s level", level)
