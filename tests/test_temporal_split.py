"""Tests for the CICIDS2017 chronological split protocol.

The protocol's whole claim is that ``_row_index`` orders flows in time and that
splitting on it keeps every class on both sides. Both halves are tested here.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.temporal_split import (
    CICIDS2017_SCHEDULE,
    CICIDS2017_SOURCES,
    build_temporal_manifest,
    is_temporal_manifest,
    load_temporal_manifest,
    normalize_raw_label,
    temporal_source_split,
    validate_capture_chronology,
    verify_split_against_manifest,
)

WEDNESDAY = "Wednesday-workingHours.pcap_ISCX.csv"
MONDAY = "Monday-WorkingHours.pcap_ISCX.csv"


def _frame(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    """Build a cache-shaped frame from (source_file, Label, _row_index)."""
    return pd.DataFrame(
        rows, columns=["source_file", "Label", "_row_index"]
    ).assign(feature=lambda d: np.arange(len(d), dtype=float))


def _capture(source: str, label: str, start: int, n: int) -> list[tuple[str, str, int]]:
    return [(source, label, start + i) for i in range(n)]


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------
def test_split_is_chronological_within_each_class() -> None:
    frame = _frame(_capture(WEDNESDAY, "DoS", 0, 100))

    train, calibration, test = temporal_source_split(
        frame, test_size=0.30, min_test_per_class=3,
    )

    assert calibration.empty
    assert len(train) == 70
    assert len(test) == 30
    # Every training row precedes every test row for this class.
    assert train["_row_index"].max() < test["_row_index"].min()


def test_every_class_survives_on_both_sides() -> None:
    rows = (
        _capture(MONDAY, "BENIGN", 0, 1000)
        + _capture(WEDNESDAY, "BENIGN", 0, 500)
        + _capture(WEDNESDAY, "DoS", 500, 200)
        + _capture(WEDNESDAY, "Heartbleed", 700, 11)
    )
    frame = _frame(rows)

    train, _, test = temporal_source_split(
        frame, test_size=0.30, min_test_per_class=3,
    )

    for label in ("BENIGN", "DoS", "Heartbleed"):
        assert (train["Label"] == label).any(), f"{label} missing from train"
        assert (test["Label"] == label).any(), f"{label} missing from test"


def test_late_burst_class_still_reaches_training_set() -> None:
    """Heartbleed regression.

    Its 11 flows sit in one ~20 minute burst at ~86% through the Wednesday
    capture. A file-level 70/30 cutoff would put all 11 in test and leave the
    model with nothing to learn from, which is why the cut is per (file, class).
    """
    rows = (
        _capture(WEDNESDAY, "BENIGN", 0, 1000)
        # All Heartbleed rows land beyond the file-level 70% cutoff.
        + _capture(WEDNESDAY, "Heartbleed", 860, 11)
    )
    frame = _frame(rows)

    train, _, test = temporal_source_split(
        frame, test_size=0.30, min_test_per_class=3,
    )

    assert int((train["Label"] == "Heartbleed").sum()) == 7
    assert int((test["Label"] == "Heartbleed").sum()) == 4


def test_split_is_deterministic_under_input_shuffling() -> None:
    """No RNG anywhere: the split depends only on row content."""
    rows = (
        _capture(MONDAY, "BENIGN", 0, 200)
        + _capture(WEDNESDAY, "DoS", 0, 50)
        + _capture(WEDNESDAY, "Heartbleed", 90, 11)
    )
    frame = _frame(rows)
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)

    train_a, _, test_a = temporal_source_split(frame)
    train_b, _, test_b = temporal_source_split(shuffled)

    key = ["source_file", "Label", "_row_index"]
    assert sorted(map(tuple, train_a[key].to_numpy())) == \
        sorted(map(tuple, train_b[key].to_numpy()))
    assert sorted(map(tuple, test_a[key].to_numpy())) == \
        sorted(map(tuple, test_b[key].to_numpy()))


def test_train_and_test_are_disjoint() -> None:
    frame = _frame(
        _capture(MONDAY, "BENIGN", 0, 300) + _capture(WEDNESDAY, "DoS", 0, 90)
    )

    train, _, test = temporal_source_split(frame)

    key = ["source_file", "Label", "_row_index"]
    overlap = set(map(tuple, train[key].to_numpy())) & set(
        map(tuple, test[key].to_numpy())
    )
    assert not overlap
    assert len(train) + len(test) == len(frame)


def test_single_row_class_goes_to_training() -> None:
    """A class cannot be learned from zero rows; scoring it is the lesser loss."""
    frame = _frame(
        _capture(MONDAY, "BENIGN", 0, 100) + _capture(WEDNESDAY, "Heartbleed", 5, 1)
    )

    train, _, test = temporal_source_split(frame, min_test_per_class=0)

    assert int((train["Label"] == "Heartbleed").sum()) == 1
    assert int((test["Label"] == "Heartbleed").sum()) == 0


def test_min_test_per_class_is_enforced() -> None:
    frame = _frame(
        _capture(MONDAY, "BENIGN", 0, 100) + _capture(WEDNESDAY, "DoS", 0, 4)
    )

    with pytest.raises(ValueError, match="min_test_per_class"):
        temporal_source_split(frame, min_test_per_class=3)


def test_rejects_invalid_test_size() -> None:
    frame = _frame(_capture(MONDAY, "BENIGN", 0, 10))

    with pytest.raises(ValueError, match="test_size"):
        temporal_source_split(frame, test_size=1.0)


def test_rejects_missing_order_column() -> None:
    frame = _frame(_capture(MONDAY, "BENIGN", 0, 10)).drop(columns=["_row_index"])

    with pytest.raises(ValueError, match="missing columns"):
        temporal_source_split(frame)


# ---------------------------------------------------------------------------
# Chronology validation
# ---------------------------------------------------------------------------
def test_chronology_accepts_published_order() -> None:
    """Wednesday: DoS (09:47-11:23) must precede Heartbleed (15:12-15:32)."""
    frame = _frame(
        _capture(WEDNESDAY, "DoS", 0, 100)
        + _capture(WEDNESDAY, "Heartbleed", 800, 11)
    )

    report = validate_capture_chronology(frame)

    assert report["valid"]
    assert report["checked_pairs"] == 1


def test_chronology_rejects_reversed_order() -> None:
    """If row order were not chronological, this is what it would look like."""
    frame = _frame(
        _capture(WEDNESDAY, "Heartbleed", 0, 11)
        + _capture(WEDNESDAY, "DoS", 100, 100)
    )

    with pytest.raises(ValueError, match="contradicts the published"):
        validate_capture_chronology(frame)

    report = validate_capture_chronology(frame, strict=False)
    assert not report["valid"]
    assert report["violations"]


def test_schedule_covers_every_2017_capture() -> None:
    assert len(CICIDS2017_SOURCES) == 8
    assert set(CICIDS2017_SCHEDULE) == set(CICIDS2017_SOURCES)
    # Monday is the only attack-free capture.
    empty = [name for name, windows in CICIDS2017_SCHEDULE.items() if not windows]
    assert empty == [MONDAY]


def test_normalize_raw_label_strips_embedded_en_dash() -> None:
    """Web Attack labels embed byte 0x96 between the family and the subtype."""
    assert normalize_raw_label("Web Attack \x96 XSS") == "Web Attack XSS"
    assert normalize_raw_label("  DoS   Hulk  ") == "DoS Hulk"
    assert normalize_raw_label(None) == ""


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
@pytest.fixture
def manifest_path(tmp_path):
    frame = _frame(
        _capture(MONDAY, "BENIGN", 0, 200)
        + _capture(WEDNESDAY, "DoS", 0, 100)
        + _capture(WEDNESDAY, "Heartbleed", 900, 11)
    )
    train, _, test = temporal_source_split(frame)
    path = tmp_path / "manifest.json"
    build_temporal_manifest(train, test, path=path)
    return path


def test_manifest_round_trips(manifest_path) -> None:
    payload = load_temporal_manifest(manifest_path)

    assert is_temporal_manifest(payload)
    assert payload["dataset_id"] == "CICIDS2017"
    assert payload["order_basis"] == "csv_row_index"
    assert payload["final_test_locked"] is True
    assert payload["random_row_split"] is False
    assert payload["calibration_size"] == 0.0
    assert set(payload["expected_counts"]) == {"BENIGN", "DoS", "Heartbleed"}


def test_manifest_detects_a_split_that_no_longer_matches(manifest_path) -> None:
    """The source-holdout protocol never checked this; the temporal one does."""
    manifest = load_temporal_manifest(manifest_path)
    frame = _frame(
        _capture(MONDAY, "BENIGN", 0, 200)
        + _capture(WEDNESDAY, "DoS", 0, 100)
        # One Heartbleed row fewer than the manifest recorded.
        + _capture(WEDNESDAY, "Heartbleed", 900, 10)
    )
    train, _, test = temporal_source_split(frame)

    verification = verify_split_against_manifest(train, test, manifest)

    assert not verification["valid"]
    assert any("Heartbleed" in message for message in verification["mismatches"])


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("version", "source_holdout_v3_full_70_30", "Unsupported temporal"),
        ("dataset_id", "CSE-CIC-IDS2018", "dataset_id"),
        ("order_basis", "timestamp", "order_basis"),
        ("final_test_locked", False, "final_test_locked"),
        ("random_row_split", True, "random_row_split"),
        ("calibration_size", 0.1, "calibration_size"),
        ("test_size", 1.5, "test_size"),
        ("sources", ["not-a-real-capture.csv"], "unknown CICIDS2017 sources"),
        ("expected_train_rows", 999_999, "expected_train_rows"),
    ],
)
def test_manifest_rejects_malformed_fields(
    manifest_path, tmp_path, field, value, match,
) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload[field] = value
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_temporal_manifest(broken)


def test_shipped_manifest_is_valid() -> None:
    """The manifest committed to configs/splits must load and be complete."""
    payload = load_temporal_manifest(
        "configs/splits/cicids2017_temporal_70_30.json"
    )

    assert payload["version"] == "cicids2017_temporal_v1"
    assert len(payload["sources"]) == 8
    assert len(payload["expected_counts"]) == 9
    total = sum(
        entry["train"] + entry["test"] for entry in payload["expected_counts"].values()
    )
    assert total == payload["expected_train_rows"] + payload["expected_test_rows"]
    # The protocol advertises 70/30; confirm the recorded counts deliver it.
    ratio = payload["expected_train_rows"] / total
    assert 0.699 < ratio < 0.701
