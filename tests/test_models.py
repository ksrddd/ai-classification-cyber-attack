"""Tests for src.models. Phase 6."""

from __future__ import annotations

import pytest

from src.models import base, logistic_regression, mlp, random_forest


def test_model_modules_import() -> None:
    assert base is not None
    assert logistic_regression is not None
    assert random_forest is not None
    assert mlp is not None


def test_base_model_is_abstract() -> None:
    with pytest.raises(TypeError):
        base.BaseModel(config={})  # type: ignore[abstract]


@pytest.mark.skip(reason="Phase 6: implement model wrappers")
def test_random_forest_uses_random_state() -> None:
    pass
