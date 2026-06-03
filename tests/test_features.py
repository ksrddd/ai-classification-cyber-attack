"""Tests for src.features.feature_selector. Phase 5."""

from __future__ import annotations

import pytest

from src.features import feature_selector


def test_feature_selector_imports() -> None:
    assert feature_selector is not None


@pytest.mark.skip(reason="Phase 5: implement select_by_correlation()")
def test_correlation_drops_one_of_each_correlated_pair() -> None:
    pass


@pytest.mark.skip(reason="Phase 5: implement select_by_rf_importance()")
def test_rf_importance_returns_top_k() -> None:
    pass
