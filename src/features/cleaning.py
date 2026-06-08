"""Data cleaning -- row-count-changing steps.

Why this lives OUTSIDE the sklearn Pipeline: these operations remove
rows (drop NaN/Inf, drop duplicates), and sklearn transformers can't
return fewer rows than they were given. Row-preserving steps (scaling,
encoding) go inside the Pipeline instead.

Order of operations matters and is documented in ``clean``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.config.constants import LABEL_COLUMN, MAPPED_LABEL_COLUMN
from src.data.schema import DUPLICATE_COLUMNS, clean_column_names

logger = logging.getLogger(__name__)


def clean(
    df: pd.DataFrame,
    drop_dup: bool = True,
    drop_inf: bool = True,
    drop_na: bool = True,
    drop_duplicate_columns: bool = True,
) -> pd.DataFrame:
    """Apply all row-changing cleaning steps in canonical order.

    Steps
    -----
    1. Strip column-name whitespace (idempotent).
    2. Drop CICIDS's literal duplicate column ``Fwd Header Length.1``.
    3. Replace ``+/-inf`` with ``NaN`` so downstream code only sees one
       sentinel for "broken value".
    4. Drop rows with any ``NaN`` in feature columns.
    5. Drop exact duplicate rows.

    Returns a new DataFrame; never mutates the input.
    """
    n_in = len(df)
    df = clean_column_names(df)

    if drop_duplicate_columns:
        df = _drop_duplicate_columns(df)

    if drop_inf:
        df = _replace_inf_with_nan(df)

    if drop_na:
        df = _drop_rows_with_na(df)

    if drop_dup:
        df = _drop_duplicate_rows(df)

    logger.info("Cleaning summary: %d -> %d rows (%.2f%% kept)",
                n_in, len(df), 100 * len(df) / max(n_in, 1))
    return df


def filter_target_classes(
    df: pd.DataFrame,
    labels: Iterable[str],
    label_col: str = MAPPED_LABEL_COLUMN,
) -> pd.DataFrame:
    """Keep only rows whose label is in ``labels``.

    Default reads from the mapped-label column produced by
    ``label_mapping.add_mapped_column``. Useful when you want to drop
    the ``"Other"`` bucket or restrict to a specific subset.
    """
    if label_col not in df.columns:
        raise ValueError(
            f"Column {label_col!r} not present. Did you call "
            f"add_mapped_column() before filter_target_classes()?"
        )
    allowed = set(labels)
    before = df[label_col].value_counts(dropna=False)
    mask = df[label_col].isin(allowed)
    out = df.loc[mask].reset_index(drop=True)
    after = out[label_col].value_counts(dropna=False)

    dropped = before.sum() - after.sum()
    logger.info(
        "Filtered to target classes %s: dropped %d rows (%d -> %d)",
        sorted(allowed), dropped, before.sum(), after.sum(),
    )
    return out


def drop_other_class(
    df: pd.DataFrame,
    label_col: str = MAPPED_LABEL_COLUMN,
    other_value: str = "Other",
) -> pd.DataFrame:
    """Drop rows whose mapped label is ``"Other"``."""
    if label_col not in df.columns:
        return df
    before = len(df)
    out = df.loc[df[label_col].astype(object) != other_value].reset_index(drop=True)
    if len(out) != before:
        logger.info(
            "Dropped %d rows mapped to %r (kept %d)",
            before - len(out), other_value, len(out),
        )
    return out


def split_features_and_label(
    df: pd.DataFrame,
    label_col: str = MAPPED_LABEL_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` ready for sklearn.

    The raw ``LABEL_COLUMN`` is also dropped from X so the label never
    leaks back into features.
    """
    drop_cols = [c for c in (LABEL_COLUMN, label_col) if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[label_col].astype(object) if label_col in df.columns else df[LABEL_COLUMN]
    return X, pd.Series(y, index=df.index, name=label_col)


# ---------------------------------------------------------------------------
# Internal step helpers
# ---------------------------------------------------------------------------


def _drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in DUPLICATE_COLUMNS if c in df.columns]
    if present:
        logger.info("Dropping CICIDS duplicate columns: %s", present)
        df = df.drop(columns=present)
    return df


def _replace_inf_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=np.number).columns
    if len(numeric) == 0:
        return df
    n_inf = int(np.isinf(df[numeric].to_numpy()).sum())
    if n_inf:
        logger.info("Replacing %d +/-Inf values with NaN", n_inf)
        df = df.copy()
        df[numeric] = df[numeric].replace([np.inf, -np.inf], np.nan)
    return df


def _drop_rows_with_na(df: pd.DataFrame) -> pd.DataFrame:
    n_in = len(df)
    out = df.dropna()
    dropped = n_in - len(out)
    if dropped:
        logger.info("Dropped %d rows containing NaN", dropped)
    return out.reset_index(drop=True) if dropped else df


def _drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    n_in = len(df)
    out = df.drop_duplicates()
    dropped = n_in - len(out)
    if dropped:
        logger.info("Dropped %d exact-duplicate rows", dropped)
    return out.reset_index(drop=True) if dropped else df
