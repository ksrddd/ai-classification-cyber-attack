from __future__ import annotations

import pandas as pd
import pytest

from src.data.deterministic_split import (
    SOURCE_ROLES,
    deterministic_source_split,
    load_split_manifest,
    row_hash,
    source_role,
    split_manifest,
    stable_quota,
    validate_source_assignment,
)
from train import parse_args


def test_source_roles_are_disjoint_and_cover_manifest() -> None:
    names = [name for values in SOURCE_ROLES.values() for name in values]
    validate_source_assignment(names)
    assert len(names) == len(set(names))
    assert source_role("02-28-2018.csv") == "train"
    assert source_role("03-01-2018.csv") == "test"


def test_row_hash_and_quota_are_reproducible_without_rng() -> None:
    rows = pd.DataFrame(
        {
            "Label": ["BENIGN"] * 4 + ["Bot"] * 2,
            "_row_hash": [row_hash("x.csv", i) for i in range(6)],
            "value": range(6),
        }
    )
    first = stable_quota(rows, quota_by_label={"BENIGN": 2, "Bot": 1})
    second = stable_quota(rows.sample(frac=1, random_state=99), quota_by_label={"BENIGN": 2, "Bot": 1})
    assert first.sort_values("_row_hash")["_row_hash"].tolist() == second.sort_values("_row_hash")["_row_hash"].tolist()
    assert first["Label"].value_counts().to_dict() == {"BENIGN": 2, "Bot": 1}


def test_unknown_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="not assigned"):
        source_role("unknown.csv")
    with pytest.raises(ValueError, match="missing"):
        validate_source_assignment(["02-14-2018.csv", "unknown.csv"])


def test_source_protocol_has_no_calibration_partition() -> None:
    rows = []
    for index, source in enumerate(
        source for names in SOURCE_ROLES.values() for source in names
    ):
        rows.append({
            "source_file": source,
            "_row_hash": row_hash(source, index),
            "Label": "BENIGN",
            "feature": float(index),
        })
    train, calibration, test = deterministic_source_split(
        pd.DataFrame(rows), label_column="Label", quotas={"BENIGN": 100}
    )
    assert calibration.empty
    assert len(train) + len(test) == len(rows)
    assert set(train["source_file"]) == set(SOURCE_ROLES["train"])
    assert set(test["source_file"]) == set(SOURCE_ROLES["test"])
    assert split_manifest()["calibration_size"] == 0.0


def test_manifest_loader_returns_normalized_protocol(project_root) -> None:
    manifest = load_split_manifest(
        project_root / "configs" / "splits" / "source_holdout_v2_70_30.json"
    )
    assert manifest["roles"] == SOURCE_ROLES
    assert manifest["calibration_size"] == 0.0


def test_train_parser_supports_no_calibration_and_explicit_gpu() -> None:
    args = parse_args(["--threshold-validation-size", "0", "--accelerator", "gpu"])
    assert args["threshold_validation_size"] == 0.0
    assert args["accelerator"] == "gpu"
