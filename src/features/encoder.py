"""Label encoding helpers.

Wraps ``sklearn.preprocessing.LabelEncoder`` with:
- joblib persistence helpers.
- An assertion that the fitted class set matches the expected label tuple
  for the active classification mode (binary or multiclass). Mismatch is
  usually a sign that label normalization was skipped.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config.constants import (
    BINARY_LABELS,
    MODELS_DIR,
    MULTICLASS_LABELS,
    get_target_labels,
)
from src.utils.io import load_joblib, save_joblib

logger = logging.getLogger(__name__)

DEFAULT_PATH: Path = MODELS_DIR / "label_encoder.joblib"


def fit_label_encoder(
    y: pd.Series | np.ndarray,
    mode: str = "multiclass",
    strict: bool = False,
) -> LabelEncoder:
    """Fit a LabelEncoder on ``y`` and verify the class set.

    Parameters
    ----------
    y
        Series or array of post-normalization labels (i.e. already mapped
        to BINARY_LABELS or MULTICLASS_LABELS).
    mode
        ``"binary"`` or ``"multiclass"`` -- determines the expected class set.
    strict
        If True, raise when ``y`` is missing one or more expected classes.
        If False (default), warn and continue -- useful on small subsamples
        where rare classes (e.g. Heartbleed) may not show up.
    """
    expected = get_target_labels(mode)
    le = LabelEncoder()
    le.fit(y)
    fitted = tuple(le.classes_.tolist())

    rogue = set(fitted) - set(expected)
    if rogue:
        raise ValueError(
            f"LabelEncoder saw labels outside the expected {mode} set: {rogue}. "
            "Did you forget add_mapped_column(df, mode)?"
        )

    missing = set(expected) - set(fitted)
    if missing:
        msg = (
            f"LabelEncoder is missing {len(missing)} expected classes: {sorted(missing)}. "
            "This is normal if your subsample is small or you set drop_other_class=true."
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    logger.info("Fitted LabelEncoder (%s): %s -> %s",
                mode, fitted, list(le.transform(le.classes_)))
    return le


def save_label_encoder(le: LabelEncoder, path: Path = DEFAULT_PATH) -> Path:
    return save_joblib(le, path)


def load_label_encoder(path: Path = DEFAULT_PATH) -> LabelEncoder:
    le = load_joblib(path)
    if not isinstance(le, LabelEncoder):
        raise TypeError(f"Expected sklearn LabelEncoder at {path}, got {type(le)}")
    return le


def encoder_classes(le: LabelEncoder) -> tuple[str, ...]:
    """Return the encoder's class list as a plain tuple of str."""
    return tuple(str(c) for c in le.classes_)


def is_binary_encoder(le: LabelEncoder) -> bool:
    """True if ``le`` was fit on the binary label scheme."""
    return set(encoder_classes(le)) <= set(BINARY_LABELS)


def is_multiclass_encoder(le: LabelEncoder) -> bool:
    """True if ``le`` was fit on the multiclass label scheme."""
    return set(encoder_classes(le)) <= set(MULTICLASS_LABELS) and not is_binary_encoder(le)
