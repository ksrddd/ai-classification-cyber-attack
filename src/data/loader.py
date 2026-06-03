"""CICIDS2017 CSV loader.

Reads one-or-many CSV files from ``data/raw/``, normalizes column names,
validates schema, optionally row-subsamples (stratified on label), and
returns a single concatenated DataFrame.

Memory note: CICIDS Friday-Afternoon-DDos is ~225 MB on disk and grows
to ~1.5 GB in pandas because every numeric column defaults to float64.
We don't downcast here (would mask Inf bugs); cleaning + parquet
serialisation cut the on-disk size later.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.constants import LABEL_COLUMN, RAW_DIR, RANDOM_STATE
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
        Explicit file list (names relative to ``raw_dir``). If ``None``,
        loads every ``*.csv`` under ``raw_dir``.
    raw_dir
        Directory containing the CSVs.
    subsample_n
        If given, return a stratified row sample of this size on
        ``LABEL_COLUMN``. Useful for development / MLP training.
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
        If no CSVs are found in ``raw_dir`` (likely cause: dataset not
        downloaded yet).
    """
    raw_dir = Path(raw_dir)
    paths = _resolve_paths(files, raw_dir)
    if not paths:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. "
            "Download CICIDS2017 from https://www.unb.ca/cic/datasets/ids-2017.html "
            "and place the CSVs under data/raw/."
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


def _resolve_paths(files: list[str] | None, raw_dir: Path) -> list[Path]:
    if files is None:
        return sorted(raw_dir.glob("*.csv"))
    return [raw_dir / f for f in files]


def _read_one(path: Path) -> pd.DataFrame:
    """Read a single CSV with CICIDS-aware options."""
    if not path.exists():
        raise FileNotFoundError(f"CICIDS CSV not found: {path}")

    logger.debug("Reading %s", path.name)
    # low_memory=False: forces pandas to read the whole column before
    # inferring dtype, avoiding the mixed-type warnings CICIDS triggers.
    df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    logger.debug("  -> %s rows x %s cols", *df.shape)
    return df


def stratified_subsample(
    df: pd.DataFrame,
    n: int,
    label_col: str = LABEL_COLUMN,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Return ``n`` rows stratified on ``label_col``.

    Each class contributes ``n * class_freq`` rows (rounded), so the class
    distribution of the subsample matches the original. If a class has
    fewer rows than its quota, all of its rows are kept.
    """
    if n >= len(df):
        return df.copy()

    rng = np.random.default_rng(random_state)
    pieces: list[pd.DataFrame] = []
    for label, group in df.groupby(label_col, sort=False):
        quota = max(1, round(n * len(group) / len(df)))
        take = min(quota, len(group))
        idx = rng.choice(group.index.to_numpy(), size=take, replace=False)
        pieces.append(df.loc[idx])

    out = pd.concat(pieces, axis=0).sample(frac=1, random_state=random_state)
    out = out.reset_index(drop=True)
    return out
