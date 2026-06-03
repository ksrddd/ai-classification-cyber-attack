"""Label encoding helpers.

Wraps ``sklearn.preprocessing.LabelEncoder`` with:
- joblib persistence helpers.
- An assertion that the fitted class set matches ``TARGET_LABELS``
  (catches the case where ``filter_target_classes`` was forgotten and
  unexpected labels sneak into training).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config.constants import MODELS_DIR, TARGET_LABELS
from src.utils.io import load_joblib, save_joblib

logger = logging.getLogger(__name__)

DEFAULT_PATH: Path = MODELS_DIR / "label_encoder.joblib"


def fit_label_encoder(
    y: pd.Series | np.ndarray,
    expected_labels: tuple[str, ...] = TARGET_LABELS,
) -> LabelEncoder:
    """Fit a LabelEncoder on ``y`` and verify the class set."""
    le = LabelEncoder()
    le.fit(y)
    fitted = tuple(le.classes_.tolist())
    if set(fitted) != set(expected_labels):
        raise ValueError(
            f"LabelEncoder class set does not match TARGET_LABELS. "
            f"Encoder saw {fitted}, expected {expected_labels}. "
            "Did you forget filter_target_classes(df, TARGET_LABELS)?"
        )
    logger.info("Fitted LabelEncoder: %s -> %s",
                fitted, list(le.transform(le.classes_)))
    return le


def save_label_encoder(le: LabelEncoder, path: Path = DEFAULT_PATH) -> Path:
    return save_joblib(le, path)


def load_label_encoder(path: Path = DEFAULT_PATH) -> LabelEncoder:
    le = load_joblib(path)
    if not isinstance(le, LabelEncoder):
        raise TypeError(f"Expected sklearn LabelEncoder at {path}, got {type(le)}")
    return le
