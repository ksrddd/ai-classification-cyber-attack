"""Shared pytest fixtures.

Phase 11 will expand this with a synthetic CICIDS-shaped DataFrame fixture
so tests don't need the real (multi-GB) CSVs.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src importable in tests when running ``pytest`` from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


# TODO(phase 11):
#   @pytest.fixture
#   def synthetic_cicids_df() -> pd.DataFrame: ...
#   @pytest.fixture
#   def cleaned_df(synthetic_cicids_df) -> pd.DataFrame: ...
