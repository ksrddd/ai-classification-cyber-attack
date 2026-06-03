"""Smoke tests for the utils layer."""

from __future__ import annotations

from pathlib import Path

from src.utils.io import ensure_dir, save_joblib, load_joblib
from src.utils.logging import configure_logging
from src.utils.seeds import seed_everything


def test_seed_everything_is_idempotent() -> None:
    seed_everything()
    seed_everything()


def test_configure_logging_is_idempotent() -> None:
    configure_logging()
    configure_logging()


def test_ensure_dir_creates_and_returns(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "subdir"
    result = ensure_dir(target)
    assert result == target
    assert target.is_dir()


def test_save_and_load_joblib_roundtrip(tmp_path: Path) -> None:
    obj = {"alpha": 1, "beta": [1, 2, 3]}
    path = tmp_path / "art.joblib"
    save_joblib(obj, path)
    assert path.exists()
    loaded = load_joblib(path)
    assert loaded == obj
