"""Deterministic, source-held-out split helpers for CICIDS experiments.

The canonical trainer historically sampled individual rows from one mixed
corpus.  This module keeps capture/source metadata outside the feature matrix
and makes the experimental split auditable and reproducible without an RNG.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

V2_SPLIT_MANIFEST_VERSION = "source_holdout_v2_70_30"
V3_SPLIT_MANIFEST_VERSION = "source_holdout_v3_full_70_30"
SPLIT_MANIFEST_VERSION = V3_SPLIT_MANIFEST_VERSION

SOURCE_ROLES: dict[str, tuple[str, ...]] = {
    "train": (
        "02-14-2018.csv",
        "02-20-2018.csv",
        "02-22-2018.csv",
        "02-28-2018.csv",
        "03-02-2018.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
    ),
    "test": (
        "02-15-2018.csv",
        "02-16-2018.csv",
        "02-21-2018.csv",
        "02-23-2018.csv",
        "03-01-2018.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Monday-WorkingHours.pcap_ISCX.csv",
    ),
}

V2_TRAIN_QUOTAS: dict[str, int] = {
    "BENIGN": 100_000,
    "Bot": 80_000,
    "Brute Force": 80_000,
    "DDoS": 80_000,
    "DoS": 80_000,
    "Infiltration": 68_871,
    "PortScan": 158_930,
    "Web Attack": 2_180,
    "Heartbleed": 11,
}

# Full-corpus protocol for the cleaned 18-source delivery corpus.  Every
# attack row from a TRAIN source is retained.  Only BENIGN is capped, by
# stable row hash, so that the untouched 4,084,201-row source-held test set is
# exactly 30% of the selected 13,614,003-row corpus (within integer rounding).
V3_TRAIN_QUOTAS: dict[str, int] = {
    "BENIGN": 8_239_774,
    "Bot": 144_535,
    "Brute Force": 94_094,
    "DDoS": 703_026,
    "DoS": 193_730,
    "Heartbleed": 11,
    "Infiltration": 61_457,
    "PortScan": 90_694,
    "Web Attack": 2_481,
}
V3_EXPECTED_TRAIN_ROWS = 9_529_802
V3_EXPECTED_TEST_ROWS = 4_084_201

TRAIN_QUOTAS = V3_TRAIN_QUOTAS
SUPPORTED_TRAIN_QUOTAS: dict[str, dict[str, int]] = {
    V2_SPLIT_MANIFEST_VERSION: V2_TRAIN_QUOTAS,
    V3_SPLIT_MANIFEST_VERSION: V3_TRAIN_QUOTAS,
}


def load_split_manifest(path: Path) -> dict[str, Any]:
    """Load and validate the supported source-holdout manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("version")
    expected_quotas = SUPPORTED_TRAIN_QUOTAS.get(str(version))
    if expected_quotas is None:
        raise ValueError(
            f"Unsupported split manifest version: {version!r}"
        )

    raw_roles = payload.get("roles")
    if not isinstance(raw_roles, dict):
        raise ValueError("Split manifest must define a roles mapping")
    roles = {
        str(role): tuple(str(name) for name in names)
        for role, names in raw_roles.items()
    }
    if roles != SOURCE_ROLES:
        raise ValueError("Split manifest roles do not match the supported protocol")
    validate_source_assignment(
        (source for names in roles.values() for source in names),
        roles=roles,
    )

    raw_quotas = payload.get("train_quotas")
    if not isinstance(raw_quotas, dict):
        raise ValueError("Split manifest must define train_quotas")
    quotas = {str(label): int(value) for label, value in raw_quotas.items()}
    if quotas != expected_quotas:
        raise ValueError("Split manifest train_quotas do not match the supported protocol")
    if payload.get("final_test_locked") is not True:
        raise ValueError("Split manifest must lock the final test set")
    if float(payload.get("calibration_size", -1.0)) != 0.0:
        raise ValueError("Source-held protocol does not permit a calibration split")
    if payload.get("random_row_split") is not False:
        raise ValueError("Source-held protocol cannot use a random row split")
    if version == V3_SPLIT_MANIFEST_VERSION:
        if int(payload.get("expected_train_rows", -1)) != V3_EXPECTED_TRAIN_ROWS:
            raise ValueError("V3 manifest expected_train_rows is invalid")
        if int(payload.get("expected_test_rows", -1)) != V3_EXPECTED_TEST_ROWS:
            raise ValueError("V3 manifest expected_test_rows is invalid")
        if sum(quotas.values()) != V3_EXPECTED_TRAIN_ROWS:
            raise ValueError("V3 train quotas do not sum to expected_train_rows")

    return {
        **payload,
        "roles": roles,
        "train_quotas": quotas,
    }


def source_role(source_file: str) -> str:
    """Return the manifest role for a source filename."""
    name = Path(source_file).name
    matches = [role for role, names in SOURCE_ROLES.items() if name in names]
    if len(matches) != 1:
        raise ValueError(f"Source is not assigned exactly one split role: {name!r}")
    return matches[0]


def row_hash(
    source_file: str,
    row_number: int,
    *,
    flow_id: Any = "",
    timestamp: Any = "",
) -> str:
    """Create a stable row identity before leaky columns are removed."""
    payload = "|".join(
        (str(source_file), str(row_number), str(flow_id), str(timestamp))
    ).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


def stable_quota(
    frame: pd.DataFrame,
    *,
    label_column: str = "Label",
    quota_by_label: dict[str, int] = TRAIN_QUOTAS,
    hash_column: str = "_row_hash",
) -> pd.DataFrame:
    """Select a fixed class quota using hash order, never a random generator."""
    if hash_column not in frame.columns:
        raise ValueError(f"Missing deterministic identity column: {hash_column}")
    if label_column not in frame.columns:
        raise ValueError(f"Missing label column: {label_column}")

    pieces: list[pd.DataFrame] = []
    for label, group in frame.groupby(label_column, sort=True, dropna=False):
        requested = quota_by_label.get(str(label), len(group))
        if requested < 0:
            raise ValueError(f"Negative quota for {label!r}")
        ordered = group.assign(_stable_order=group[hash_column].astype(str))
        ordered = ordered.sort_values("_stable_order", kind="mergesort")
        pieces.append(ordered.head(requested).drop(columns=["_stable_order"]))
    if not pieces:
        return frame.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


def deterministic_source_split(
    frame: pd.DataFrame,
    *,
    label_column: str,
    quotas: dict[str, int] | None = None,
    roles: dict[str, tuple[str, ...]] = SOURCE_ROLES,
    source_column: str = "source_file",
    hash_column: str = "_row_hash",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by capture role, quota training rows, and leave test rows locked.

    Source membership is decided before any quota is applied. Training sources
    are selected deterministically by class and row hash, while every row from
    a final-test source is preserved. The protocol intentionally has no
    calibration partition.
    """
    required = {label_column, source_column, hash_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Deterministic source split is missing columns: {missing}")

    source_names = frame[source_column].map(lambda value: Path(str(value)).name)
    validate_source_assignment(source_names.unique(), roles=roles)
    role_by_source = {
        Path(source).name: role
        for role, sources in roles.items()
        for source in sources
    }
    row_roles = source_names.map(role_by_source)

    train_pool = frame.loc[row_roles.eq("train")]
    test = frame.loc[row_roles.eq("test")].sort_values(hash_column, kind="mergesort")
    if train_pool.empty or test.empty:
        raise ValueError(
            f"Source split requires non-empty train and test rows; "
            f"train={len(train_pool)}, test={len(test)}"
        )

    train = stable_quota(
        train_pool,
        label_column=label_column,
        quota_by_label=quotas or TRAIN_QUOTAS,
        hash_column=hash_column,
    )
    calibration = frame.iloc[0:0].copy()
    return (
        train.reset_index(drop=True),
        calibration,
        test.reset_index(drop=True),
    )


def validate_source_assignment(
    source_files: Iterable[str],
    *,
    roles: dict[str, tuple[str, ...]] = SOURCE_ROLES,
) -> None:
    """Reject unknown or multiply assigned source files."""
    assigned: dict[str, str] = {}
    for role, names in roles.items():
        for name in names:
            if name in assigned:
                raise ValueError(f"Source assigned to both {assigned[name]} and {role}: {name}")
            assigned[name] = role
    unknown = sorted({Path(name).name for name in source_files} - set(assigned))
    if unknown:
        raise ValueError(f"Sources missing from split manifest: {unknown}")


def split_manifest(path: Path | None = None) -> dict[str, Any]:
    """Return the versioned manifest, optionally writing it as JSON."""
    payload: dict[str, Any] = {
        "version": SPLIT_MANIFEST_VERSION,
        "roles": {role: list(names) for role, names in SOURCE_ROLES.items()},
        "train_quotas": dict(TRAIN_QUOTAS),
        "metadata_columns": ["dataset_id", "source_file", "capture_window", "_row_hash"],
        "final_test_locked": True,
        "calibration_size": 0.0,
        "intended_train_test_ratio": "70/30",
        "random_row_split": False,
    }
    if SPLIT_MANIFEST_VERSION == V3_SPLIT_MANIFEST_VERSION:
        payload.update({
            "expected_train_rows": V3_EXPECTED_TRAIN_ROWS,
            "expected_test_rows": V3_EXPECTED_TEST_ROWS,
            "selection_policy": (
                "retain every available TRAIN attack row; "
                "cap BENIGN by stable _row_hash"
            ),
        })
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
