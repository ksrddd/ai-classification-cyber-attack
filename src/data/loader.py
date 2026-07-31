"""CICIDS2017 CSV loader.

Reads one-or-many CSV files from ``data/raw/``, normalizes column names,
validates schema, optionally row-subsamples (stratified on label), and
returns a single concatenated DataFrame.

Memory note: ``Wednesday-workingHours.pcap_ISCX.csv`` is ~215 MB on disk
and expands to ~1.5 GB in pandas because every numeric column defaults
to float64. The full 8-file corpus is ~2.8M rows. On a 16 GB laptop
prefer ``subsample_n`` for development; the production training run can
use the full corpus.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.constants import LABEL_COLUMN, RANDOM_STATE, RAW_DIR
from src.data.schema import clean_column_names, validate_schema

logger = logging.getLogger(__name__)


def load_raw(
    files: list[str] | None = None,
    raw_dir: Path = RAW_DIR,
    subsample_n: int | None = None,
    validate_strict: bool = False,
) -> pd.DataFrame:
    """Load CICIDS2017 CSVs into a single DataFrame.

    Parameters
    ----------
    files
        Explicit file list (names relative to ``raw_dir``). If ``None`` or
        empty, loads every ``*.csv`` under ``raw_dir`` -- the default for
        the new pipeline so all 15 attack labels are represented.
    raw_dir
        Directory containing the CSVs.
    subsample_n
        If given, return a stratified row sample of this size on
        ``LABEL_COLUMN``. Strongly recommended for development.
    validate_strict
        Forward to ``schema.validate_schema``. Strict mode treats
        schema drift as an error rather than a warning.

    Returns
    -------
    pd.DataFrame
        Concatenated, column-stripped, schema-validated CICIDS frame.

    Raises
    ------
    FileNotFoundError
        If no CSVs are found in ``raw_dir`` (dataset not extracted).
    """
    raw_dir = Path(raw_dir)
    paths = _resolve_paths(files, raw_dir)
    if not paths:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. "
            "Extract MachineLearningCSV.zip from "
            "https://www.unb.ca/cic/datasets/ids-2017.html into data/raw/."
        )

    logger.info("Loading %d CSV file(s) from %s", len(paths), raw_dir)
    frames = [_read_one(p) for p in paths]
    df = pd.concat(frames, axis=0, ignore_index=True)
    logger.info("Concatenated frame shape: %s", df.shape)

    df = clean_column_names(df)
    validate_schema(df, strict=validate_strict)

    if subsample_n is not None and subsample_n < len(df):
        df = stratified_subsample(df, n=subsample_n)
        logger.info("Stratified subsample applied: %s", df.shape)

    return df


def list_raw_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    """Return all CSV paths under ``raw_dir`` in sorted order."""
    return sorted(Path(raw_dir).glob("*.csv"))


def _resolve_paths(files: list[str] | None, raw_dir: Path) -> list[Path]:
    if not files:
        return list_raw_files(raw_dir)
    return [raw_dir / f for f in files]


def _read_one(path: Path) -> pd.DataFrame:
    """Read a single CICIDS CSV with format-aware options."""
    if not path.exists():
        raise FileNotFoundError(f"CICIDS CSV not found: {path}")

    logger.debug("Reading %s", path.name)
    # low_memory=False: forces pandas to read the whole column before
    # inferring dtype, avoiding mixed-type warnings CICIDS triggers.
    # latin-1: tolerates the 0x96 byte inside Web Attack labels.
    df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    logger.debug("  -> %s rows x %s cols", *df.shape)
    return df


def stratified_subsample(
    df: pd.DataFrame,
    n: int,
    label_col: str = LABEL_COLUMN,
    random_state: int = RANDOM_STATE,
    min_per_class: int = 1,
) -> pd.DataFrame:
    """Return ``n`` rows stratified on ``label_col``.

    Each class contributes ``n * class_freq`` rows (rounded), so the class
    distribution of the subsample matches the original. Classes with fewer
    rows than their quota contribute all their rows.

    ``min_per_class`` ensures rare classes (Heartbleed = 11 rows) never
    drop to zero when n is very small.
    """
    if n >= len(df):
        return df.copy()

    rng = np.random.default_rng(random_state)
    pieces: list[pd.DataFrame] = []
    for _label, group in df.groupby(label_col, sort=False):
        quota = max(min_per_class, round(n * len(group) / len(df)))
        take = min(quota, len(group))
        idx = rng.choice(group.index.to_numpy(), size=take, replace=False)
        pieces.append(df.loc[idx])

    out = pd.concat(pieces, axis=0).sample(frac=1, random_state=random_state)
    out = out.reset_index(drop=True)
    return out
