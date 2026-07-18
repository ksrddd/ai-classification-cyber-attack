"""Lightweight IQR-based OOD checks for local inference validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OODProfile:
    columns: tuple[str, ...]
    lower: dict[str, float]
    upper: dict[str, float]


def fit_iqr_profile(
    reference: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    whisker: float = 3.0,
) -> OODProfile:
    if whisker <= 0:
        raise ValueError("whisker must be positive")
    selected = columns or [c for c in reference.columns if pd.api.types.is_numeric_dtype(reference[c])]
    missing = [c for c in selected if c not in reference.columns]
    if missing:
        raise ValueError(f"Missing OOD columns: {missing}")
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    for column in selected:
        values = pd.to_numeric(reference[column], errors="coerce")
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = float(q3 - q1)
        lower[column] = float(q1 - whisker * iqr)
        upper[column] = float(q3 + whisker * iqr)
    return OODProfile(tuple(selected), lower, upper)


@dataclass(frozen=True)
class OODReport:
    row_flags: tuple[bool, ...]
    feature_fraction: tuple[float, ...]
    rejected_rows: int


def detect_ood(
    candidate: pd.DataFrame,
    profile: OODProfile,
    *,
    max_feature_fraction: float = 0.1,
) -> OODReport:
    if not 0.0 <= max_feature_fraction <= 1.0:
        raise ValueError("max_feature_fraction must be in [0, 1]")
    missing = [c for c in profile.columns if c not in candidate.columns]
    if missing:
        raise ValueError(f"Candidate is missing OOD columns: {missing}")
    flags: list[bool] = []
    fractions: list[float] = []
    for _, row in candidate.iterrows():
        bad = 0
        for column in profile.columns:
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if not np.isfinite(value) or value < profile.lower[column] or value > profile.upper[column]:
                bad += 1
        fraction = bad / max(len(profile.columns), 1)
        fractions.append(fraction)
        flags.append(fraction > max_feature_fraction)
    return OODReport(tuple(flags), tuple(fractions), sum(flags))
