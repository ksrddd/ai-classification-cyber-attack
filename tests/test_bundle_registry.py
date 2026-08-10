"""Tests for per-class split counts surfaced by the bundle registry.

The dashboard's Distributions page renders a Train column beside a Test column.
Both are read from the same payload, so a train count that is silently absent
looks identical to one that is genuinely unrecorded -- and for a long time
every CSE-CIC-IDS2018 bundle hardcoded an empty dict, which meant the page
could never show train counts for that layout even when the run had written
them to disk.

These tests pin both halves of that contract: the counts are read where they
exist, and they stay absent -- not zero, not inferred -- where they do not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.bundles.registry import _train_supports

CLASSES = ["Benign", "Bot", "SQL Injection"]


def _write_manifest(path: Path, counts: dict[str, dict[str, int]]) -> None:
    payload = {
        "version": "ids2018_temporal_v1",
        "split_protocol": "ids2018_temporal_v1",
        "expected_counts": counts,
    }
    (path / "split_manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_train_counts_read_from_temporal_manifest(tmp_path: Path) -> None:
    """A temporal bundle records both sides; both must reach the payload."""
    _write_manifest(
        tmp_path,
        {
            "Benign": {"train": 174446, "test": 74769},
            "Bot": {"train": 3702, "test": 1587},
            "SQL Injection": {"train": 1, "test": 1},
        },
    )

    counts = _train_supports(tmp_path, CLASSES)

    assert counts == {"Benign": 174446, "Bot": 3702, "SQL Injection": 1}


def test_absent_manifest_yields_absent_counts_not_zeros(tmp_path: Path) -> None:
    """A pre-backfill random-split bundle records the counts nowhere.

    The distinction that matters: an empty mapping renders as "not recorded",
    while a mapping of zeros would claim every class contributed no training
    rows -- a statement the run never made.
    """
    assert _train_supports(tmp_path, CLASSES) == {}
    assert _train_supports(tmp_path, CLASSES, {"sample_size": 300_000}) == {}


def test_metadata_counts_used_when_no_manifest(tmp_path: Path) -> None:
    """Random-split bundles carry the counts in metadata instead."""
    meta = {"per_class_n_train": {"Benign": 174446, "Bot": 3702, "Ghost": 5}}

    assert _train_supports(tmp_path, CLASSES, meta) == {"Benign": 174446, "Bot": 3702}


def test_manifest_wins_over_metadata(tmp_path: Path) -> None:
    """A bundle carrying both is read from the manifest.

    The manifest is derived from the split itself and is the record the
    analysis step validates, so it is the one that must win -- silently
    preferring the other would make the dashboard and the analysis disagree
    about the same run.
    """
    _write_manifest(tmp_path, {"Benign": {"train": 111, "test": 5}})
    meta = {"per_class_n_train": {"Benign": 999}}

    assert _train_supports(tmp_path, CLASSES, meta) == {"Benign": 111}


def test_classes_outside_the_bundle_are_dropped(tmp_path: Path) -> None:
    """The manifest is not trusted to agree with the bundle's class list."""
    _write_manifest(
        tmp_path,
        {
            "Benign": {"train": 10, "test": 5},
            "Ghost Class": {"train": 99, "test": 99},
        },
    )

    assert _train_supports(tmp_path, CLASSES) == {"Benign": 10}


def test_malformed_entries_are_skipped(tmp_path: Path) -> None:
    """A half-written manifest must not crash the whole bundle listing."""
    _write_manifest(
        tmp_path,
        {
            "Benign": {"train": 10, "test": 5},
            "Bot": {"test": 5},  # no train side
        },
    )

    assert _train_supports(tmp_path, CLASSES) == {"Benign": 10}


def test_unreadable_manifest_yields_absent_counts(tmp_path: Path) -> None:
    (tmp_path / "split_manifest.json").write_text("{ not json", encoding="utf-8")

    assert _train_supports(tmp_path, CLASSES) == {}


@pytest.mark.parametrize(
    "bundle",
    ["300k_temporal", "300k_temporal_tuned", "500k_temporal"],
)
def test_shipped_temporal_bundles_sum_to_their_train_split(bundle: str) -> None:
    """Cross-check against the real bundles: the parts must equal the whole.

    ``n_train`` is derived independently -- sample size minus the summed test
    supports read from a per-class report -- so agreement here means the
    manifest and the per-model reports describe the same split rather than two
    drifting records of it.
    """
    root = Path(__file__).resolve().parents[1]
    results = root / "results" / "ids2018" / bundle
    if not (results / "split_manifest.json").is_file():
        pytest.skip(f"{bundle} is not present in this checkout")

    manifest = json.loads((results / "split_manifest.json").read_text(encoding="utf-8"))
    classes = list(manifest["expected_counts"])

    counts = _train_supports(results, classes)
    assert len(counts) == len(classes)
    assert sum(counts.values()) == manifest["expected_train_rows"]

    # The test side of the same manifest must match what training actually
    # scored, otherwise the train side sitting next to it is not trustworthy.
    report = next(results.glob("*/per_class_report.csv"))
    df = pd.read_csv(report)
    label_col = df.columns[0]
    supports = {
        str(row[label_col]): int(row["support"])
        for _, row in df.iterrows()
        if str(row[label_col]) in set(classes)
    }
    expected_test = {k: v["test"] for k, v in manifest["expected_counts"].items()}
    assert supports == expected_test


@pytest.mark.parametrize("bundle", ["300k", "500k", "300k_balanced_weighting"])
def test_backfilled_random_bundles_agree_with_their_reports(bundle: str) -> None:
    """The backfilled counts must reproduce the split training actually scored.

    This is the guard on ``scripts/backfill_ids2018_counts.py``: it rebuilds a
    split rather than reading one, so the recorded test side is checked against
    the per-class report written at training time. The train side is only
    trustworthy because the test side beside it came from the same rebuild.
    """
    root = Path(__file__).resolve().parents[1]
    meta_path = root / "models" / "ids2018" / bundle / "metadata.json"
    results = root / "results" / "ids2018" / bundle
    if not meta_path.is_file() or not results.is_dir():
        pytest.skip(f"{bundle} is not present in this checkout")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("per_class_n_train"):
        pytest.skip(f"{bundle} has not been backfilled")

    # A random-split bundle must never acquire a manifest: analyze.py treats
    # that file's presence as proof the split was chronological.
    assert not (results / "split_manifest.json").is_file()

    report = next(results.glob("*/per_class_report.csv"))
    df = pd.read_csv(report)
    label_col = df.columns[0]
    skip = {"accuracy", "macro avg", "weighted avg"}
    supports = {
        str(row[label_col]): int(row["support"])
        for _, row in df.iterrows()
        if str(row[label_col]) not in skip
    }

    assert meta["per_class_n_test"] == supports
    assert sum(meta["per_class_n_train"].values()) == meta["sample_size"] - sum(supports.values())
