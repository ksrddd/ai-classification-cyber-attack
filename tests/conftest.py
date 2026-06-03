"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Force non-interactive matplotlib backend. Windows Python's bundled tcl/tk
# is broken on some installs, and tests never need a display anyway.
import matplotlib  # noqa: E402
matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from scripts.generate_sample import generate  # noqa: E402
from src.features.cleaning import clean, filter_target_classes  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def synthetic_cicids_df() -> pd.DataFrame:
    """Tiny synthetic CICIDS-shaped DataFrame.

    Column names INCLUDE the leading-space quirk so tests that exercise
    ``clean_column_names`` have something to strip.
    """
    return generate(n_rows=400, seed=42)


@pytest.fixture(scope="session")
def cleaned_df(synthetic_cicids_df) -> pd.DataFrame:
    """Synthetic frame after cleaning. Used by encoder / pipeline tests."""
    from src.config.constants import TARGET_LABELS
    df = clean(synthetic_cicids_df)
    df = filter_target_classes(df, TARGET_LABELS)
    return df
