"""Centralized logging configuration.

Every entry point (``main.py``, dashboard, notebooks) calls
``configure_logging()`` exactly once. Modules then do
``logger = logging.getLogger(__name__)`` — no per-module setup.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.config.constants import LOGS_DIR

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False  # guard against repeat configuration in notebooks / Streamlit


def configure_logging(
    level: str | int = "INFO",
    log_file: Path | None = None,
    fmt: str = _DEFAULT_FORMAT,
) -> None:
    """Configure the root logger with a console handler and an optional file handler.

    Idempotent — calling twice in the same process is a no-op. This matters
    because Streamlit re-imports modules on every interaction.

    Parameters
    ----------
    level
        Log level name (``DEBUG``/``INFO``/``WARNING``/``ERROR``) or numeric.
    log_file
        Optional path. If given, logs are also written there. Parent dir is
        created if missing.
    fmt
        Format string for the root handler.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    # On Windows with a non-UTF-8 console codepage (e.g. Thai cp874),
    # any non-ASCII character in a log message raises UnicodeEncodeError.
    # Force stdout/stderr to UTF-8 with replacement on Python 3.7+ where
    # ``reconfigure`` exists.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(fmt, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # encoding="utf-8" ensures the log file is portable across OS / locales.
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Quiet down third-party noise.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("numexpr").setLevel(logging.WARNING)

    _CONFIGURED = True


def default_log_file() -> Path:
    """Standard log location: ``<project>/logs/pipeline.log``."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return LOGS_DIR / "pipeline.log"
