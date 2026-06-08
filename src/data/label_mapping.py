"""CICIDS2017 label normalization + grouping.

The raw CSVs contain 15 distinct label strings. We collapse them into two
schemes:

- ``binary``:     BENIGN -> ``"Normal"``, all attacks -> ``"Attack"``.
- ``multiclass``: 10 attack families per the project spec --
  BENIGN, DoS, DDoS, PortScan, Bot, Web Attack, Brute Force,
  Infiltration, Heartbleed, Other.

Two CICIDS quirks worth knowing about:

1. The three ``Web Attack ...`` labels embed a Windows-1252 byte ``0x96``
   (en-dash). When the CSV is loaded with ``encoding="latin-1"`` that
   shows up as U+0096 (a C1 control char). We normalize it away.
2. The DoS sub-types use inconsistent capitalisation (``DoS slowloris``,
   ``DoS Slowhttptest``). Matching is case-insensitive after normalization.
"""

from __future__ import annotations

import logging
import re
import unicodedata

import numpy as np
import pandas as pd

from src.config.constants import (
    BINARY_LABELS,
    LABEL_COLUMN,
    MAPPED_LABEL_COLUMN,
    MULTICLASS_LABELS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical raw -> multiclass mapping
# ---------------------------------------------------------------------------
# Keys are the *normalized* (collapsed-whitespace, ASCII-only) form of the
# raw CICIDS label. ``normalize_label`` produces this form; ``map_to_multiclass``
# does the lookup.
RAW_TO_MULTICLASS: dict[str, str] = {
    "benign":                       "BENIGN",
    "dos hulk":                     "DoS",
    "dos goldeneye":                "DoS",
    "dos slowloris":                "DoS",
    "dos slowhttptest":             "DoS",
    "ddos":                         "DDoS",
    "portscan":                     "PortScan",
    "bot":                          "Bot",
    "web attack brute force":       "Web Attack",
    "web attack xss":               "Web Attack",
    "web attack sql injection":     "Web Attack",
    "ftp-patator":                  "Brute Force",
    "ssh-patator":                  "Brute Force",
    "infiltration":                 "Infiltration",
    "heartbleed":                   "Heartbleed",
}

# Fallback prefix rules -- catch new sub-variants without explicit listing.
_PREFIX_FALLBACKS: tuple[tuple[str, str], ...] = (
    ("dos ",          "DoS"),
    ("ddos",          "DDoS"),
    ("web attack",    "Web Attack"),
    ("portscan",      "PortScan"),
    ("infiltration",  "Infiltration"),
    ("heartbleed",    "Heartbleed"),
    ("bot",           "Bot"),
    ("ftp-patator",   "Brute Force"),
    ("ssh-patator",   "Brute Force"),
    ("brute force",   "Brute Force"),
)

_BENIGN_NAME = "BENIGN"
_OTHER_NAME = "Other"

# Matches any C0/C1 control chars and the Unicode replacement char (U+FFFD).
# The Web Attack labels embed U+0096 from the Windows-1252 en-dash; latin-1
# decoding preserves the byte verbatim, so we strip it here.
_BAD_CHAR_RE = re.compile("[\x00-\x1f\x7f-�]+")
# Collapse runs of whitespace.
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_label(raw: object) -> str:
    """Normalize a single raw label for lookup.

    Steps: NFKD-normalize, drop combining marks, replace control/replacement
    chars with a space, collapse whitespace, lowercase, strip. The result is
    a stable ASCII-only key for the lookup tables.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    s = str(raw)
    s = unicodedata.normalize("NFKD", s)
    s = _BAD_CHAR_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s)
    return s.strip().lower()


def map_to_multiclass(raw: object) -> str:
    """Map a raw label to one of ``MULTICLASS_LABELS``.

    Lookup order: exact normalized match -> prefix fallback -> ``"Other"``.
    """
    key = normalize_label(raw)
    if not key:
        return _OTHER_NAME
    if key == "benign":
        return _BENIGN_NAME
    if key in RAW_TO_MULTICLASS:
        return RAW_TO_MULTICLASS[key]
    for prefix, target in _PREFIX_FALLBACKS:
        if key.startswith(prefix):
            return target
    return _OTHER_NAME


def map_to_binary(raw: object) -> str:
    """Map a raw label to binary: BENIGN -> ``"Normal"``, else -> ``"Attack"``."""
    key = normalize_label(raw)
    return "Normal" if key == "benign" else "Attack"


def add_mapped_column(
    df: pd.DataFrame,
    mode: str = "multiclass",
    label_col: str = LABEL_COLUMN,
    out_col: str = MAPPED_LABEL_COLUMN,
) -> pd.DataFrame:
    """Append a normalized-label column to ``df`` based on ``mode``.

    Parameters
    ----------
    df
        DataFrame containing the raw ``label_col``.
    mode
        ``"binary"`` or ``"multiclass"``.
    label_col, out_col
        Source and destination column names.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with ``out_col`` set to a Categorical with the
        canonical category order from constants.
    """
    if label_col not in df.columns:
        raise ValueError(f"DataFrame has no column {label_col!r}.")

    if mode == "binary":
        mapped = df[label_col].map(map_to_binary)
        categories = BINARY_LABELS
    elif mode == "multiclass":
        mapped = df[label_col].map(map_to_multiclass)
        categories = MULTICLASS_LABELS
    else:
        raise ValueError(f"Unknown mode {mode!r}. Use 'binary' or 'multiclass'.")

    out = df.copy()
    out[out_col] = pd.Categorical(mapped, categories=list(categories))
    _audit(df[label_col], out[out_col], mode)
    return out


def label_distribution(df: pd.DataFrame, col: str) -> pd.Series:
    """Return value_counts of ``col`` including zero-count categorical levels."""
    if col not in df.columns:
        raise ValueError(f"Column {col!r} not in DataFrame.")
    if isinstance(df[col].dtype, pd.CategoricalDtype):
        return df[col].value_counts().reindex(df[col].cat.categories, fill_value=0)
    return df[col].value_counts()


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _audit(raw_series: pd.Series, mapped_series: pd.Series, mode: str) -> None:
    """Log per-class counts after mapping; warn on any 'Other' overflow."""
    after = mapped_series.value_counts(dropna=False).to_dict()
    logger.info("Label mapping (%s) result: %s", mode, after)
    if mode == "multiclass" and after.get(_OTHER_NAME, 0) > 0:
        # Surface which raw labels fell into Other so a future contributor
        # can extend RAW_TO_MULTICLASS rather than silently swallow them.
        other_mask = mapped_series.astype(object) == _OTHER_NAME
        rogue = raw_series[other_mask].value_counts().head(5).to_dict()
        logger.warning(
            "Mapped %d rows to 'Other'. Top raw labels in this bucket: %s",
            int(other_mask.sum()),
            rogue,
        )
