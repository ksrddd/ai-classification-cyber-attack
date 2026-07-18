"""Tests for src.features.cleaning + src.data.label_mapping."""

from __future__ import annotations

import numpy as np

from src.config.constants import MAPPED_LABEL_COLUMN
from src.data.label_mapping import (
    add_mapped_column,
    map_to_binary,
    map_to_multiclass,
    normalize_label,
)
from src.features.cleaning import clean, drop_other_class, filter_target_classes


def test_clean_strips_columns_and_drops_inf_na(synthetic_cicids_df) -> None:
    raw = synthetic_cicids_df
    assert any(col.startswith(" ") for col in raw.columns)  # fixture has the bug

    cleaned = clean(raw)

    assert not any(col.startswith(" ") for col in cleaned.columns)
    numeric = cleaned.select_dtypes(include=np.number)
    assert not np.isinf(numeric.to_numpy()).any()
    assert cleaned.isna().sum().sum() == 0
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


def test_normalize_label_strips_control_chars() -> None:
    raw = "Web Attack \x96 Brute Force"
    assert normalize_label(raw) == "web attack brute force"


def test_map_to_multiclass_maps_all_real_labels() -> None:
    cases = {
        "BENIGN":                              "BENIGN",
        "DoS Hulk":                            "DoS",
        "DoS GoldenEye":                       "DoS",
        "DoS slowloris":                       "DoS",
        "DoS Slowhttptest":                    "DoS",
        "DDoS":                                "DDoS",
        "PortScan":                            "PortScan",
        "Bot":                                 "Bot",
        "FTP-Patator":                         "Brute Force",
        "SSH-Patator":                         "Brute Force",
        "Infiltration":                        "Infiltration",
        "Heartbleed":                          "Heartbleed",
        "Web Attack \x96 Brute Force":         "Web Attack",
        "Web Attack \x96 XSS":                 "Web Attack",
        "Web Attack \x96 Sql Injection":       "Web Attack",
    }
    for raw, expected in cases.items():
        assert map_to_multiclass(raw) == expected, raw


def test_map_to_multiclass_unknown_goes_to_other() -> None:
    assert map_to_multiclass("CompletelyNewAttack") == "Other"
    assert map_to_multiclass(None) == "Other"


def test_map_to_binary_distinguishes_benign() -> None:
    assert map_to_binary("BENIGN") == "Normal"
    assert map_to_binary("DoS Hulk") == "Attack"
    assert map_to_binary("Heartbleed") == "Attack"


def test_add_mapped_column_creates_categorical(synthetic_cicids_df) -> None:
    df = clean(synthetic_cicids_df)
    out = add_mapped_column(df, mode="multiclass")
    assert MAPPED_LABEL_COLUMN in out.columns
    assert out[MAPPED_LABEL_COLUMN].dtype.name == "category"


def test_filter_target_classes_drops_unwanted(cleaned_df) -> None:
    keep = {"BENIGN", "DoS"}
    out = filter_target_classes(cleaned_df, keep, label_col=MAPPED_LABEL_COLUMN)
    assert set(out[MAPPED_LABEL_COLUMN].dropna().astype(object).unique()) <= keep


def test_drop_other_class_removes_other_rows(cleaned_df) -> None:
    out = drop_other_class(cleaned_df, label_col=MAPPED_LABEL_COLUMN)
    assert "Other" not in out[MAPPED_LABEL_COLUMN].astype(object).unique()
