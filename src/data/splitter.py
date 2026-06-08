"""Train / validation / test splitting.

A two-step stratified split that:

1. Carves out ``test`` from the full dataset.
2. Carves out ``val`` from the remaining (train + val) pool.

Both splits stratify on the (encoded) label so class proportions match
across the three sets -- important for the rarer attack classes
(Heartbleed, Infiltration) that can vanish under a naive random split.

Single shuffle + fixed seed -> reproducible across runs (R-03).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.constants import RANDOM_STATE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitResult:
    """Container for the six arrays of a three-way split."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

    def shapes(self) -> dict[str, Any]:
        return {
            "X_train": self.X_train.shape,
            "X_val":   self.X_val.shape,
            "X_test":  self.X_test.shape,
            "y_train": self.y_train.shape,
            "y_val":   self.y_val.shape,
            "y_test":  self.y_test.shape,
        }


def train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    stratify: bool = True,
    random_state: int = RANDOM_STATE,
) -> SplitResult:
    """Stratified three-way split.

    ``val_size`` is interpreted as a fraction of the *post-test* pool, so
    with ``test_size=0.2`` and ``val_size=0.2`` you get 0.64 / 0.16 / 0.20
    train / val / test.

    Stratification is dropped automatically for any class that has fewer
    than two samples in the pool being split (sklearn would raise);
    these rare classes (Heartbleed, Infiltration with subsampling) still
    end up represented because the *first* split runs on the full data.
    """
    strat = y if stratify else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=strat,
        random_state=random_state,
    )

    strat = y_temp if stratify and _can_stratify(y_temp) else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        stratify=strat,
        random_state=random_state,
    )

    result = SplitResult(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=pd.Series(y_train).reset_index(drop=True),
        y_val=pd.Series(y_val).reset_index(drop=True),
        y_test=pd.Series(y_test).reset_index(drop=True),
    )
    logger.info("Split shapes: %s", result.shapes())
    return result


def _can_stratify(y: pd.Series | np.ndarray) -> bool:
    """sklearn refuses stratify when any class has < 2 members."""
    s = pd.Series(y)
    return int(s.value_counts().min()) >= 2
