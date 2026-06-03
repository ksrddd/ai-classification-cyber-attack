"""Tests for src.features.cleaning."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.constants import LABEL_COLUMN, TARGET_LABELS
from src.features.cleaning import clean, filter_target_classes


def test_clean_strips_columns_and_drops_inf_na(synthetic_cicids_df) -> None:
    raw = synthetic_cicids_df
    assert any(col.startswith(" ") for col in raw.columns)  # fixture has the bug

    cleaned = clean(raw)

    # 1. Column names stripped.
    assert not any(col.startswith(" ") for col in cleaned.columns)
    # 2. No inf left.
    numeric = cleaned.select_dtypes(include=np.number)
    assert not np.isinf(numeric.to_numpy()).any()
    # 3. No NaN left.
    assert cleaned.isna().sum().sum() == 0
    # 4. Fewer rows than input (we injected NaN/Inf/duplicates).
    assert len(cleaned) < len(raw)


def test_clean_drops_known_duplicate_column(synthetic_cicids_df) -> None:
    cleaned = clean(synthetic_cicids_df)
    assert "Fwd Header Length.1" not in cleaned.columns


def test_clean_does_not_mutate_input(synthetic_cicids_df) -> None:
    before_shape = synthetic_cicids_df.shape
    before_cols = list(synthetic_cicids_df.columns)
    clean(synthetic_cicids_df)
    assert synthetic_cicids_df.shape == before_shape
    assert list(synthetic_cicids_df.columns) == before_cols


def test_filter_target_classes_keeps_only_allowed(cleaned_df) -> None:
    out = filter_target_classes(cleaned_df, TARGET_LABELS)
    assert set(out[LABEL_COLUMN].unique()) <= set(TARGET_LABELS)


def test_filter_target_classes_drops_other(synthetic_cicids_df) -> None:
    df = synthetic_cicids_df.copy()
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df.loc[df.index[:5], LABEL_COLUMN] = "Heartbleed"  # not in TARGET_LABELS
    out = filter_target_classes(df, TARGET_LABELS)
    assert "Heartbleed" not in out[LABEL_COLUMN].unique()
