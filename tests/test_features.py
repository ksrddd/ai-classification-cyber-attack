"""Tests for src.features.encoder + src.features.pipeline + selector."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config.constants import MAPPED_LABEL_COLUMN, MULTICLASS_LABELS
from src.features import encoder, feature_selector, pipeline, validator


def test_feature_modules_import() -> None:
    assert encoder is not None
    assert feature_selector is not None
    assert pipeline is not None
    assert validator is not None


def test_fit_label_encoder_in_multiclass_mode(cleaned_df) -> None:
    le = encoder.fit_label_encoder(
        cleaned_df[MAPPED_LABEL_COLUMN].astype(object),
        mode="multiclass",
    )
    assert set(le.classes_.tolist()) <= set(MULTICLASS_LABELS)


def test_fit_label_encoder_rejects_unexpected_labels(cleaned_df) -> None:
    bad = cleaned_df.copy()
    bad[MAPPED_LABEL_COLUMN] = bad[MAPPED_LABEL_COLUMN].astype(object)
    bad.loc[bad.index[:3], MAPPED_LABEL_COLUMN] = "NotInScheme"
    with pytest.raises(ValueError, match="outside the expected"):
        encoder.fit_label_encoder(bad[MAPPED_LABEL_COLUMN], mode="multiclass")


def test_save_and_load_label_encoder_roundtrip(tmp_path: Path, cleaned_df) -> None:
    le = encoder.fit_label_encoder(
        cleaned_df[MAPPED_LABEL_COLUMN].astype(object),
        mode="multiclass",
    )
    path = tmp_path / "le.joblib"
    encoder.save_label_encoder(le, path)
    loaded = encoder.load_label_encoder(path)
    assert loaded.classes_.tolist() == le.classes_.tolist()


def test_build_model_pipeline_is_pipeline_with_scaler_and_clf() -> None:
    from sklearn.ensemble import RandomForestClassifier
    p = pipeline.build_model_pipeline(RandomForestClassifier(), scaler_kind="standard")
    assert isinstance(p, Pipeline)
    assert [name for name, _ in p.steps] == ["scaler", "clf"]


def test_build_scaler_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unknown scaler kind"):
        pipeline.build_scaler("magic")


def test_select_by_correlation_drops_redundant() -> None:
    """`a` and `b` are identical; `c` is unrelated noise. b should be dropped."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=200)
    df = pd.DataFrame({
        "a": base,
        "b": base + 1e-6 * rng.normal(size=200),  # near-identical to a
        "c": rng.normal(size=200),                 # independent
    })
    result = feature_selector.select_by_correlation(df, threshold=0.99)
    assert "a" in result.selected
    assert "b" not in result.selected
    assert "c" in result.selected


def test_validator_load_expected_features_missing(tmp_path: Path) -> None:
    bogus = tmp_path / "feature_names.json"
    with pytest.raises(FileNotFoundError):
        validator.load_expected_features(bogus)


def test_validator_validate_ok_with_full_schema(tmp_path: Path) -> None:
    fnames = ["Destination Port", "Flow Duration"]
    path = tmp_path / "feature_names.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(fnames, f)
    df = pd.DataFrame({"Destination Port": [1], "Flow Duration": [2]})
    rep = validator.validate_inference_csv(df, expected_features=fnames)
    assert rep.ok
