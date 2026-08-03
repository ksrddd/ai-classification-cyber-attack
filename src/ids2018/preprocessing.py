"""Cleaning, encoding and scaling for CSE-CIC-IDS2018.

Written from scratch for the 2018 schema ("Tot Fwd Pkts", "Init Fwd Win
Byts", ...). None of the 2017 feature names or cleaning rules apply here,
so nothing is imported from ``src.features``.

The class-based design exists to keep the train/test boundary honest:
every statistic that could leak (imputation medians, scaler mean/std,
which columns are constant) is learned on the training split alone and
then replayed onto the test split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler

from src.ids2018.config import (
    DST_PORT_COL,
    IDENTITY_COLS,
    LABEL_COL,
    RANDOM_STATE,
    TEST_SIZE,
    TIMESTAMP_COL,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Column surgery
# ----------------------------------------------------------------------
def split_features_labels(
    df: pd.DataFrame, keep_dst_port: bool = False, keep_timestamp: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    """Drop junk columns and separate the target.

    Removed unconditionally:

    * ``Flow ID / Src IP / Src Port / Dst IP`` -- present only in
      02-20-2018.csv. Pure identifiers: a model that learns "traffic from
      18.219.211.138 is an attack" has memorised the lab, not the attack.
    * ``Timestamp`` -- each attack ran in its own capture window, so the
      timestamp is a near-perfect label proxy (target leakage).
    * ``Dst Port`` -- unless ``keep_dst_port``. High-cardinality and tied
      to the lab setup (HOIC always targeted port 80).

    ``keep_timestamp`` retains ``Timestamp`` *as an ordering key only*, for the
    chronological split in :mod:`src.ids2018.temporal_split`. The caller must
    drop it before the preprocessor sees it; the leakage argument above is
    unchanged, and ``temporal_split.assert_timestamp_absent`` enforces it.
    """
    drop = [c for c in IDENTITY_COLS if c in df.columns]
    if not keep_timestamp and TIMESTAMP_COL in df.columns:
        drop.append(TIMESTAMP_COL)
    if not keep_dst_port and DST_PORT_COL in df.columns:
        drop.append(DST_PORT_COL)

    if drop:
        logger.info("Dropping non-generalisable columns: %s", ", ".join(drop))

    y = df[LABEL_COL].astype(str)
    X = df.drop(columns=[*drop, LABEL_COL], errors="ignore")
    return X, y


def stratified_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """70/30 split that preserves the class proportions of the sample.

    Stratifying here matters as much as it did during sampling: without it
    the rarest classes (SQL Injection, Brute Force -XSS) can land entirely
    in one side of the split, making their test metrics undefined.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y, shuffle=True
    )
    logger.info(
        "Stratified split -> train %s rows / test %s rows (%.0f/%.0f)",
        f"{len(X_train):,}",
        f"{len(X_test):,}",
        100 * (1 - test_size),
        100 * test_size,
    )
    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------------------
# Feature preprocessor
# ----------------------------------------------------------------------
@dataclass
class Ids2018Preprocessor:
    """Fit-on-train / apply-to-test feature preparation.

    Pipeline order:

    1. Coerce every column to numeric (bad tokens become NaN).
    2. Replace +/-inf with NaN. ``Flow Byts/s`` and ``Flow Pkts/s`` are
       byte/packet counts divided by flow duration, and single-packet
       flows have duration 0, so infinities are common and legitimate.
    3. Drop columns that are entirely NaN or constant **on the training
       split** -- the 2018 export ships several always-zero columns
       (``Fwd Byts/b Avg``, ``Bwd Blk Rate Avg``, ...) that carry no signal.
    4. Impute remaining NaNs with the training median (robust to the heavy
       right skew of flow features).
    5. Downcast to float32 -- halves memory with no measurable accuracy cost.
    6. Scale. StandardScaler by default; RobustScaler is the better choice
       if the tail-heavy IAT features dominate, hence the option.
    """

    scaler_kind: str = "standard"
    feature_names: list[str] = field(default_factory=list)
    medians: pd.Series | None = None
    scaler: StandardScaler | RobustScaler | None = None
    dropped_columns: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------
    @staticmethod
    def _to_numeric(X: pd.DataFrame) -> pd.DataFrame:
        """Force numeric dtypes and neutralise infinities."""
        out = X.apply(pd.to_numeric, errors="coerce")
        return out.replace([np.inf, -np.inf], np.nan)

    def _make_scaler(self) -> StandardScaler | RobustScaler:
        if self.scaler_kind == "robust":
            return RobustScaler()
        if self.scaler_kind == "standard":
            return StandardScaler()
        raise ValueError(f"Unknown scaler_kind {self.scaler_kind!r}; use 'standard' or 'robust'")

    # ---------------------------------------------------------------
    def fit(self, X: pd.DataFrame) -> Ids2018Preprocessor:
        """Learn every statistic from the training split only."""
        num = self._to_numeric(X)

        all_nan = num.columns[num.isna().all()].tolist()
        # nunique(dropna=True) == 1 catches constant columns; == 0 is
        # already covered by all_nan.
        constant = num.columns[num.nunique(dropna=True) <= 1].tolist()
        self.dropped_columns = sorted(set(all_nan) | set(constant))
        if self.dropped_columns:
            logger.info(
                "Dropping %d uninformative column(s) (all-NaN or constant on train): %s",
                len(self.dropped_columns),
                ", ".join(self.dropped_columns),
            )

        num = num.drop(columns=self.dropped_columns)
        self.feature_names = num.columns.tolist()
        self.medians = num.median()

        filled = num.fillna(self.medians).astype(np.float32)
        self.scaler = self._make_scaler().fit(filled)

        logger.info(
            "Preprocessor fitted: %d features, scaler=%s",
            len(self.feature_names),
            self.scaler_kind,
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply the fitted cleaning + scaling to any split."""
        if self.scaler is None or self.medians is None:
            raise RuntimeError("Ids2018Preprocessor.transform() called before fit()")

        num = self._to_numeric(X)
        # reindex enforces the training column order and silently supplies
        # a NaN column if an input is missing one, which the median fill
        # then handles.
        num = num.reindex(columns=self.feature_names)
        filled = num.fillna(self.medians).astype(np.float32)
        return self.scaler.transform(filled).astype(np.float32)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)


# ----------------------------------------------------------------------
# Label encoding
# ----------------------------------------------------------------------
def encode_labels(
    y_train: pd.Series, y_test: pd.Series
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Map class names to the contiguous 0..n-1 integers XGBoost requires.

    The encoder is fitted on the union of both splits so a class that the
    stratified split happened to route entirely into test still gets a
    stable code. With stratification that should not happen, but the
    encoder is cheap insurance and keeps the code order deterministic.
    """
    encoder = LabelEncoder().fit(pd.concat([y_train, y_test], ignore_index=True))
    logger.info("Encoded %d classes: %s", len(encoder.classes_), list(encoder.classes_))
    return encoder.transform(y_train), encoder.transform(y_test), encoder
