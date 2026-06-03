"""Tests for src.features.encoder. (feature_selector lands in Phase 5.)"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.constants import LABEL_COLUMN, TARGET_LABELS
from src.features import encoder, feature_selector


def test_feature_selector_module_imports() -> None:
    assert feature_selector is not None  # populated in Phase 5


def test_fit_label_encoder_returns_target_labels(cleaned_df) -> None:
    le = encoder.fit_label_encoder(cleaned_df[LABEL_COLUMN])
    assert set(le.classes_.tolist()) == set(TARGET_LABELS)


def test_fit_label_encoder_rejects_unexpected_labels(cleaned_df) -> None:
    bad = cleaned_df.copy()
    bad.loc[bad.index[:3], LABEL_COLUMN] = "UnknownAttack"
    with pytest.raises(ValueError, match="filter_target_classes"):
        encoder.fit_label_encoder(bad[LABEL_COLUMN])


def test_save_and_load_label_encoder_roundtrip(tmp_path: Path, cleaned_df) -> None:
    le = encoder.fit_label_encoder(cleaned_df[LABEL_COLUMN])
    path = tmp_path / "le.joblib"
    encoder.save_label_encoder(le, path)
    loaded = encoder.load_label_encoder(path)
    assert loaded.classes_.tolist() == le.classes_.tolist()
