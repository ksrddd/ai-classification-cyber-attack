"""Tests for src.features.cleaning + features.pipeline. Phase 4/5."""

from __future__ import annotations

import pytest

from src.features import cleaning, pipeline


def test_cleaning_module_imports() -> None:
    assert hasattr(cleaning, "clean")


def test_pipeline_module_imports() -> None:
    assert pipeline is not None


@pytest.mark.skip(reason="Phase 4: implement clean()")
def test_clean_drops_inf_and_na() -> None:
    pass


@pytest.mark.skip(reason="Phase 5: implement build_model_pipeline()")
def test_pipeline_refits_scaler_inside_cv() -> None:
    """The scaler must NOT see test data — leak guard."""
