"""CICIDS2017 CSV loader.

Responsibilities (Phase 4):
- Read one or many CSV files from ``data/raw/``.
- Strip the leading whitespace bug from column names (R-01).
- Validate schema via ``src.data.schema``.
- Return a single concatenated DataFrame, optionally row-subsampled.

Logging: every load logs row + column counts and the file list.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config.constants import RAW_DIR

logger = logging.getLogger(__name__)


def load_raw(
    files: list[str] | None = None,
    raw_dir: Path = RAW_DIR,
    subsample_n: int | None = None,
) -> pd.DataFrame:
    """Load CICIDS2017 CSVs into a single DataFrame.

    Parameters
    ----------
    files
        Optional explicit file list (relative to ``raw_dir``). If ``None``,
        loads every ``*.csv`` in ``raw_dir``.
    raw_dir
        Directory containing the CSVs.
    subsample_n
        If given, stratified row-subsample to this many rows on ``Label``.

    Returns
    -------
    pd.DataFrame
    """
    # TODO(phase 4): implement
    raise NotImplementedError("Phase 4 implementation pending.")
