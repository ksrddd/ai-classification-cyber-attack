"""Schema validation for user-uploaded CSVs (Dashboard page 6).

When a user uploads a CSV via the Streamlit dashboard, we must check
that it has the same feature columns the training Pipeline expects.
Mismatch -> friendly error, no inference attempt.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config.constants import PROCESSED_DIR
from src.data.schema import clean_column_names

logger = logging.getLogger(__name__)

FEATURE_NAMES_PATH: Path = PROCESSED_DIR / "feature_names.json"


@dataclass
class ValidationReport:
    """Result of validating an uploaded CSV against the training schema."""

    ok: bool
    n_rows: int
    n_cols: int
    missing: list[str]
    extra: list[str]
    message: str

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "missing": self.missing,
            "extra": self.extra,
            "message": self.message,
        }


def load_expected_features(path: Path = FEATURE_NAMES_PATH) -> list[str]:
    """Return the feature column list produced by the preprocessing stage."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run preprocessing first: "
            "`python main.py --stage preprocess`."
        )
    with path.open("r", encoding="utf-8") as f:
        return list(json.load(f))


def validate_inference_csv(
    df: pd.DataFrame,
    expected_features: list[str] | None = None,
) -> ValidationReport:
    """Check that ``df`` has every column listed in ``expected_features``.

    Column names are whitespace-stripped before comparison (CICIDS bug).
    Extra columns are allowed but reported. Missing columns are fatal.
    """
    expected = expected_features if expected_features is not None else load_expected_features()
    df = clean_column_names(df)

    present = set(df.columns)
    needed = set(expected)
    missing = sorted(needed - present)
    extra = sorted(present - needed)

    if missing:
        msg = (
            f"Upload is missing {len(missing)} required column(s). "
            f"Example: {missing[:3]}."
        )
        return ValidationReport(False, len(df), df.shape[1], missing, extra, msg)

    return ValidationReport(
        ok=True,
        n_rows=len(df),
        n_cols=df.shape[1],
        missing=[],
        extra=extra,
        message=(f"OK -- {len(df)} rows, {df.shape[1]} columns. "
                 f"{len(extra)} extra column(s) ignored." if extra else
                 f"OK -- {len(df)} rows, {df.shape[1]} columns."),
    )


def align_inference_csv(df: pd.DataFrame, expected_features: list[str]) -> pd.DataFrame:
    """Reorder + subset ``df`` to match the training feature schema."""
    df = clean_column_names(df)
    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot align -- missing columns: {missing}")
    return df[expected_features].copy()
