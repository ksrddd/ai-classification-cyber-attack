"""Tests for src.evaluation. Phase 8."""

from __future__ import annotations

import pytest

from src.evaluation import comparison, confusion_matrix, metrics


def test_evaluation_modules_import() -> None:
    assert metrics is not None
    assert confusion_matrix is not None
    assert comparison is not None


@pytest.mark.skip(reason="Phase 8: implement compute_metrics()")
def test_compute_metrics_returns_expected_keys() -> None:
    pass
