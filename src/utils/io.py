"""Small I/O helpers shared by the pipelines.

Keep this module narrow. Heavy persistence (parquet, model files) goes here
so callers don't import joblib/pandas just for file plumbing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if missing; return it for chaining."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_joblib(obj: Any, path: Path) -> Path:
    """Persist ``obj`` via joblib. Parent directory is created if missing."""
    ensure_dir(path.parent)
    joblib.dump(obj, path)
    logger.info("Saved joblib artifact -> %s", path)
    return path


def load_joblib(path: Path) -> Any:
    """Load a joblib artifact. Raises ``FileNotFoundError`` with a clear msg."""
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}. "
            "Run the training pipeline first (`python main.py --stage train`)."
        )
    logger.debug("Loading joblib artifact <- %s", path)
    return joblib.load(path)
