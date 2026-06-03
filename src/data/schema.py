"""CICIDS2017 schema constants + validation helpers.

The Canadian Institute for Cybersecurity ships CICIDS2017 with a known
data-quality bug: every column name has a single leading space (e.g.
``" Destination Port"``). ``clean_column_names`` strips that.

``EXPECTED_FEATURES`` is the canonical 78-feature list after stripping.
``validate_schema`` raises if the label column is missing, warns on any
unexpected feature drift (so a slightly different CICIDS variant still
loads but doesn't pass silently).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd

from src.config.constants import LABEL_COLUMN

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected feature columns (in canonical order, post-whitespace-strip)
# ---------------------------------------------------------------------------
# 78 flow-level features as documented by Sharafaldin et al. (2018).
# Column 56 ("Fwd Header Length.1") is a known duplicate of column 35
# in the original CSVs — kept here for fidelity; dropped in cleaning.
EXPECTED_FEATURES: tuple[str, ...] = (
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
)

EXPECTED_COLUMNS: tuple[str, ...] = (*EXPECTED_FEATURES, LABEL_COLUMN)

# Column known to be a literal duplicate of "Fwd Header Length" — dropped in
# cleaning.drop_redundant_columns.
DUPLICATE_COLUMNS: tuple[str, ...] = ("Fwd Header Length.1",)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from every column name.

    Returns a new DataFrame with renamed columns; does not mutate input.
    Idempotent: calling twice has no further effect.
    """
    return df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)


def validate_schema(df: pd.DataFrame, strict: bool = False) -> None:
    """Validate a CICIDS-shaped DataFrame.

    Always raises if ``LABEL_COLUMN`` is missing — without it nothing
    downstream works. Missing or unexpected feature columns produce a
    warning by default; pass ``strict=True`` to make them errors too.

    Parameters
    ----------
    df
        DataFrame to validate. Column names should already be stripped.
    strict
        If True, missing expected features or unexpected extra columns
        raise ``ValueError``. If False (default), they only warn.
    """
    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Required column '{LABEL_COLUMN}' is missing from the DataFrame. "
            f"Got columns: {list(df.columns)[:8]}..."
        )

    present = set(df.columns)
    expected = set(EXPECTED_COLUMNS)
    missing = expected - present
    extra = present - expected

    if missing:
        msg = f"Schema missing {len(missing)} expected columns: {sorted(missing)[:5]}..."
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    if extra:
        msg = f"Schema has {len(extra)} unexpected columns: {sorted(extra)[:5]}..."
        if strict:
            raise ValueError(msg)
        logger.warning(msg)


def assert_labels_subset(df: pd.DataFrame, allowed: Iterable[str]) -> None:
    """Assert every value in the label column is in ``allowed``.

    Used after ``filter_target_classes`` to verify the filter did its job.
    """
    actual = set(df[LABEL_COLUMN].unique())
    bad = actual - set(allowed)
    if bad:
        raise ValueError(
            f"DataFrame contains label values not in allowed set: {bad}. "
            f"Allowed: {set(allowed)}."
        )
