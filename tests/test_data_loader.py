"""Tests for src.data.loader. Real implementation lands in Phase 4."""

from __future__ import annotations

import pytest

from src.data import loader


def test_loader_module_imports() -> None:
    assert hasattr(loader, "load_raw")


@pytest.mark.skip(reason="Phase 4: implement load_raw + add synthetic fixture")
def test_load_raw_strips_column_whitespace() -> None:
    """CICIDS2017 columns have leading whitespace (R-01). Loader must strip."""
