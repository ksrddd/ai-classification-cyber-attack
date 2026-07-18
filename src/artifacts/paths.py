"""Canonical, traversal-safe paths for versioned training runs."""

from __future__ import annotations

import re
from pathlib import Path

from src.config.constants import RESULTS_DIR

RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")


def result_run_dir(run_id: str, *, results_root: Path = RESULTS_DIR) -> Path:
    """Return ``results/<run-id>`` while rejecting path-like run IDs."""
    if not run_id or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(
            "Run ID may contain only letters, numbers, dots, underscores, and hyphens"
        )
    root = results_root.resolve()
    run_dir = (root / run_id).resolve()
    run_dir.relative_to(root)
    return run_dir
