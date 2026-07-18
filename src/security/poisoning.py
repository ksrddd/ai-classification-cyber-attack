"""Data-poisoning indicators that operate only on local DataFrames."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class LabelConflictReport:
    groups_checked: int
    conflicting_groups: int
    conflicting_rows: int
    examples: tuple[dict[str, object], ...]


def find_conflicting_labels(
    frame: pd.DataFrame,
    *,
    label_column: str,
    feature_columns: list[str] | None = None,
    max_examples: int = 10,
) -> LabelConflictReport:
    """Find identical feature vectors assigned to multiple labels."""
    if label_column not in frame.columns:
        raise ValueError(f"Missing label column: {label_column}")
    features = feature_columns or [c for c in frame.columns if c != label_column]
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    groups = frame.groupby(features, dropna=False, sort=False, observed=True)
    conflicts: list[dict[str, object]] = []
    conflicting_groups = 0
    conflicting_rows = 0
    for key, group in groups:
        labels = group[label_column].drop_duplicates().tolist()
        if len(labels) > 1:
            conflicting_groups += 1
            conflicting_rows += len(group)
            if len(conflicts) < max_examples:
                values = key if isinstance(key, tuple) else (key,)
                conflicts.append({"features": dict(zip(features, values, strict=True)), "labels": labels})
    return LabelConflictReport(
        groups_checked=int(frame.groupby(features, dropna=False, observed=True).ngroups),
        conflicting_groups=conflicting_groups,
        conflicting_rows=conflicting_rows,
        examples=tuple(conflicts),
    )


@dataclass(frozen=True)
class DistributionShiftReport:
    total_variation: float
    reference_counts: dict[str, int]
    candidate_counts: dict[str, int]


def label_distribution_shift(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    label_column: str,
) -> DistributionShiftReport:
    """Return total-variation distance between two label distributions."""
    if label_column not in reference.columns or label_column not in candidate.columns:
        raise ValueError(f"Missing label column: {label_column}")
    ref = reference[label_column].astype(str).value_counts().to_dict()
    cur = candidate[label_column].astype(str).value_counts().to_dict()
    labels = set(ref) | set(cur)
    ref_total = max(sum(ref.values()), 1)
    cur_total = max(sum(cur.values()), 1)
    tv = 0.5 * sum(abs(ref.get(k, 0) / ref_total - cur.get(k, 0) / cur_total) for k in labels)
    return DistributionShiftReport(float(tv), dict(ref), dict(cur))
