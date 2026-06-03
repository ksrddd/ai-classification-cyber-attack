"""Tests for src.data.loader and src.data.schema."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config.constants import LABEL_COLUMN, TARGET_LABELS
from src.data import loader
from src.data.schema import (
    EXPECTED_FEATURES,
    assert_labels_subset,
    clean_column_names,
    validate_schema,
)


def test_loader_module_exposes_load_raw() -> None:
    assert callable(loader.load_raw)


def test_schema_has_78_features_plus_label() -> None:
    assert len(EXPECTED_FEATURES) == 78


def test_clean_column_names_strips_whitespace() -> None:
    """R-01: real CICIDS columns ship with a leading space."""
    df = pd.DataFrame(columns=[" Destination Port", " Flow Duration", "Label"])
    stripped = clean_column_names(df)
    assert list(stripped.columns) == ["Destination Port", "Flow Duration", "Label"]


def test_validate_schema_raises_when_label_missing() -> None:
    df = pd.DataFrame({"Flow Duration": [1.0]})
    with pytest.raises(ValueError, match="Label"):
        validate_schema(df)


def test_validate_schema_warns_on_drift_when_not_strict(caplog) -> None:
    df = pd.DataFrame({"Random Feature": [1.0], LABEL_COLUMN: ["BENIGN"]})
    with caplog.at_level("WARNING"):
        validate_schema(df, strict=False)
    assert any("missing" in r.message or "unexpected" in r.message for r in caplog.records)


def test_validate_schema_raises_on_drift_when_strict() -> None:
    df = pd.DataFrame({"Random Feature": [1.0], LABEL_COLUMN: ["BENIGN"]})
    with pytest.raises(ValueError):
        validate_schema(df, strict=True)


def test_load_raw_raises_on_empty_dir(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="No CSV files"):
        loader.load_raw(raw_dir=tmp_path)


def test_load_raw_reads_synthetic_csv(tmp_path, synthetic_cicids_df) -> None:
    csv = tmp_path / "synth.csv"
    synthetic_cicids_df.to_csv(csv, index=False)
    df = loader.load_raw(raw_dir=tmp_path)
    assert LABEL_COLUMN in df.columns
    assert not any(col.startswith(" ") for col in df.columns), \
        "columns must be stripped after load"
    assert len(df) == len(synthetic_cicids_df)


def test_stratified_subsample_preserves_class_distribution(synthetic_cicids_df) -> None:
    df = clean_column_names(synthetic_cicids_df)
    n = 100
    sub = loader.stratified_subsample(df, n=n)
    assert abs(len(sub) - n) <= len(df[LABEL_COLUMN].unique())  # rounding slack
    # Every class in the original should still be represented.
    assert set(sub[LABEL_COLUMN].unique()) == set(df[LABEL_COLUMN].unique())


def test_assert_labels_subset_catches_unknown_label() -> None:
    df = pd.DataFrame({LABEL_COLUMN: ["BENIGN", "MysteryAttack"]})
    with pytest.raises(ValueError, match="MysteryAttack"):
        assert_labels_subset(df, TARGET_LABELS)
