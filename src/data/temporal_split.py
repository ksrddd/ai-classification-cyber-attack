"""Temporal (chronological) split helpers for the CICIDS2017-only protocol.

Why this module exists
----------------------
The source-held-out protocol in :mod:`src.data.deterministic_split` assigns
*whole capture files* to train or test.  That is only possible because
CSE-CIC-IDS2018 supplies each attack family across several captures.  In
CICIDS2017 every attack class lives in exactly one capture file::

    Monday                     BENIGN only
    Tuesday                    Brute Force
    Wednesday                  DoS, Heartbleed
    Thursday-Morning-Web       Web Attack
    Thursday-Afternoon-Infil   Infiltration
    Friday-Morning             Bot
    Friday-Afternoon-DDos      DDoS
    Friday-Afternoon-PortScan  PortScan

so holding out any file leaves that file's attack class with zero training
rows.  Source holdout is therefore impossible for a 2017-only corpus.

The replacement protocol splits **chronologically inside each capture**: for
every ``(source_file, class)`` group the earliest ``1 - test_size`` of the
flows train the model and the latest ``test_size`` test it.  Every class is
present on both sides, and no test flow precedes the training flows of its own
class -- a stricter evaluation than the random row split used by most
CICIDS2017 literature.

Splitting per ``(source_file, class)`` rather than per ``source_file`` is
required, not cosmetic: Heartbleed's 11 flows all occur in a single ~20 minute
burst at roughly 86% through the Wednesday capture, so a single file-level
cutoff would place every one of them in test and none in train.

Where the chronology comes from
-------------------------------
The CICIDS2017 CSVs distributed as *MachineLearningCVE* carry no ``Timestamp``
column, so ordering uses ``_row_index`` -- the row's position in its source CSV
before cleaning, recorded by ``train._clean_one_frame``.

That row order is chronological, and :func:`validate_raw_label_chronology`
proves it rather than assuming it: it checks the observed order of every attack
sub-label against the attack schedule published by the Canadian Institute for
Cybersecurity.  All ten ordered sub-labels across the three multi-attack
captures agree with the published timetable.

One caveat worth stating in any write-up: CICFlowMeter emits a flow record when
the flow *terminates*, so the ordering is by flow completion, not flow start.
This is visible in the data -- DoS GoldenEye holds connections open, so its
records trail well past its 11:10-11:23 attack window.  Completion order is
still the correct ordering for an IDS evaluation, because a flow's features
only exist once the flow is complete.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

TEMPORAL_MANIFEST_VERSION = "cicids2017_temporal_v1"
TEMPORAL_MANIFEST_PREFIX = "cicids2017_temporal"
ORDER_BASIS = "csv_row_index"
DATASET_ID = "CICIDS2017"


class AttackWindow(NamedTuple):
    """One published attack window from the CIC CICIDS2017 timetable."""

    raw_label: str        # label string as it appears in the raw CSV
    canonical_label: str  # family after train.normalize_label
    start: str            # published start time, 24h clock
    end: str              # published end time, 24h clock


# Published CICIDS2017 attack schedule, in chronological order per capture.
# Source: Canadian Institute for Cybersecurity dataset documentation.
# Used as the ground truth against which observed row order is validated --
# it is never derived from the data, which is what makes the check meaningful.
CICIDS2017_SCHEDULE: dict[str, tuple[AttackWindow, ...]] = {
    "Monday-WorkingHours.pcap_ISCX.csv": (),
    "Tuesday-WorkingHours.pcap_ISCX.csv": (
        AttackWindow("FTP-Patator", "Brute Force", "09:20", "10:20"),
        AttackWindow("SSH-Patator", "Brute Force", "14:00", "15:00"),
    ),
    "Wednesday-workingHours.pcap_ISCX.csv": (
        AttackWindow("DoS slowloris", "DoS", "09:47", "10:10"),
        AttackWindow("DoS Slowhttptest", "DoS", "10:14", "10:35"),
        AttackWindow("DoS Hulk", "DoS", "10:43", "11:00"),
        AttackWindow("DoS GoldenEye", "DoS", "11:10", "11:23"),
        AttackWindow("Heartbleed", "Heartbleed", "15:12", "15:32"),
    ),
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": (
        AttackWindow("Web Attack Brute Force", "Web Attack", "09:20", "10:00"),
        AttackWindow("Web Attack XSS", "Web Attack", "10:15", "10:35"),
        AttackWindow("Web Attack Sql Injection", "Web Attack", "10:40", "10:42"),
    ),
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": (
        AttackWindow("Infiltration", "Infiltration", "14:19", "15:45"),
    ),
    "Friday-WorkingHours-Morning.pcap_ISCX.csv": (
        AttackWindow("Bot", "Bot", "10:02", "11:02"),
    ),
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": (
        AttackWindow("PortScan", "PortScan", "13:55", "15:29"),
    ),
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": (
        AttackWindow("DDoS", "DDoS", "15:56", "16:16"),
    ),
}

CICIDS2017_SOURCES: tuple[str, ...] = tuple(sorted(CICIDS2017_SCHEDULE))


def normalize_raw_label(value: object) -> str:
    """Canonicalise a raw label for schedule matching, keeping the sub-type.

    Web Attack labels embed byte 0x96 (a Windows-1252 en dash) between "Web
    Attack" and the sub-type.  Non-ASCII bytes become spaces and runs of
    whitespace collapse, so ``'Web Attack \\x96 XSS'`` matches the schedule's
    ``'Web Attack XSS'``.  Unlike ``train.normalize_label`` this deliberately
    preserves the sub-type, because the sub-types are what make the
    chronology check discriminating.
    """
    text = "" if value is None else str(value)
    ascii_only = "".join(ch if 0x20 <= ord(ch) < 0x7F else " " for ch in text)
    return " ".join(ascii_only.split())


# ---------------------------------------------------------------------------
# Chronology validation
# ---------------------------------------------------------------------------
def _position_stats(positions: np.ndarray, total: int) -> dict[str, float]:
    """Where a label sits in its capture, as fractions of the file length."""
    denominator = float(max(total - 1, 1))
    return {
        "n": int(positions.size),
        "min": float(positions.min() / denominator),
        "median": float(np.median(positions) / denominator),
        "max": float(positions.max() / denominator),
    }


def _check_order(observed: list[tuple[str, float]],
                 expected: list[str]) -> list[str]:
    """Return a violation message per out-of-order pair (empty when correct)."""
    rank = {label: i for i, label in enumerate(expected)}
    ordered = sorted(observed, key=lambda item: item[1])
    violations: list[str] = []
    for (left, left_pos), (right, right_pos) in zip(ordered, ordered[1:], strict=False):
        if rank[left] > rank[right]:
            violations.append(
                f"{left!r} (median position {left_pos:.4f}) precedes "
                f"{right!r} ({right_pos:.4f}), but the published schedule "
                f"puts {right!r} first"
            )
    return violations


def validate_raw_label_chronology(
    raw_dir: Path | str,
    *,
    schedule: dict[str, tuple[AttackWindow, ...]] | None = None,
    encoding: str = "latin-1",
    label_column: str = "Label",
    strict: bool = True,
) -> dict[str, Any]:
    """Prove raw CSV row order is chronological, from the published schedule.

    Reads only the label column of each capture, so it is cheap.  For every
    capture with two or more published attack windows, the observed median row
    position of each attack sub-label must follow the published order.  This is
    the evidence behind the ``order_basis = "csv_row_index"`` claim: the
    expected order comes from CIC's timetable, never from the data.

    Raises ``ValueError`` on any violation when ``strict``; always returns the
    per-label position statistics so they can be reported.
    """
    schedule = CICIDS2017_SCHEDULE if schedule is None else schedule
    raw_dir = Path(raw_dir)
    report: dict[str, Any] = {"captures": {}, "violations": [], "checked_pairs": 0}

    for source, windows in sorted(schedule.items()):
        path = raw_dir / source
        if not path.exists():
            raise FileNotFoundError(f"Capture file not found: {path}")
        if len(windows) < 2:
            # A single-attack capture cannot order anything.
            report["captures"][source] = {"ordered": None, "labels": {}}
            continue

        frame = pd.read_csv(path, encoding=encoding, low_memory=False)
        frame = frame.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
        if label_column not in frame.columns:
            raise ValueError(f"{path.name} has no {label_column!r} column")
        labels = frame[label_column].map(normalize_raw_label)
        total = len(frame)

        expected = [window.raw_label for window in windows]
        missing = [name for name in expected if not (labels == name).any()]
        if missing:
            raise ValueError(
                f"{source}: published attack label(s) {missing} absent from the "
                "capture; the schedule and the data disagree"
            )

        stats = {
            name: _position_stats(np.flatnonzero((labels == name).to_numpy()), total)
            for name in expected
        }
        observed = [(name, stats[name]["median"]) for name in expected]
        violations = _check_order(observed, expected)
        report["checked_pairs"] += len(expected) - 1
        report["captures"][source] = {
            "ordered": not violations,
            "expected_order": expected,
            "labels": stats,
        }
        report["violations"].extend(f"{source}: {message}" for message in violations)

    report["valid"] = not report["violations"]
    if strict and report["violations"]:
        raise ValueError(
            "Raw CSV row order contradicts the published CICIDS2017 attack "
            "schedule, so it cannot be used as a chronological ordering:\n  - "
            + "\n  - ".join(report["violations"])
        )
    return report


def validate_capture_chronology(
    frame: pd.DataFrame,
    *,
    label_column: str = "Label",
    source_column: str = "source_file",
    order_column: str = "_row_index",
    schedule: dict[str, tuple[AttackWindow, ...]] | None = None,
    benign_label: str = "BENIGN",
    strict: bool = True,
) -> dict[str, Any]:
    """Chronology check against the cleaned cache, using canonical labels.

    Weaker than :func:`validate_raw_label_chronology` -- label normalisation
    collapses the DoS and Web Attack sub-types, so only Wednesday retains two
    orderable attack families (DoS before Heartbleed).  Use this as the fast
    in-pipeline guard and the raw-label check as the published evidence.
    """
    schedule = CICIDS2017_SCHEDULE if schedule is None else schedule
    missing_columns = {label_column, source_column, order_column} - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Chronology check is missing columns: {sorted(missing_columns)}"
        )

    report: dict[str, Any] = {"captures": {}, "violations": [], "checked_pairs": 0}
    for source, group in frame.groupby(source_column, sort=True):
        windows = schedule.get(str(source))
        if windows is None:
            continue
        # Canonical families in published first-appearance order, de-duplicated.
        expected: list[str] = []
        for window in windows:
            if window.canonical_label not in expected:
                expected.append(window.canonical_label)
        expected = [name for name in expected if name != benign_label]
        if len(expected) < 2:
            report["captures"][str(source)] = {"ordered": None, "labels": {}}
            continue

        order = group[order_column].to_numpy()
        stats: dict[str, dict[str, float]] = {}
        for name in expected:
            mask = (group[label_column] == name).to_numpy()
            if not mask.any():
                raise ValueError(
                    f"{source}: expected class {name!r} is absent from the cache"
                )
            stats[name] = _position_stats(order[mask], int(order.max()) + 1)
        observed = [(name, stats[name]["median"]) for name in expected]
        violations = _check_order(observed, expected)
        report["checked_pairs"] += len(expected) - 1
        report["captures"][str(source)] = {
            "ordered": not violations,
            "expected_order": expected,
            "labels": stats,
        }
        report["violations"].extend(f"{source}: {message}" for message in violations)

    report["valid"] = not report["violations"]
    if strict and report["violations"]:
        raise ValueError(
            "Cached row order contradicts the published CICIDS2017 attack "
            "schedule:\n  - " + "\n  - ".join(report["violations"])
        )
    return report


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------
def temporal_source_split(
    frame: pd.DataFrame,
    *,
    label_column: str = "Label",
    source_column: str = "source_file",
    order_column: str = "_row_index",
    test_size: float = 0.30,
    min_test_per_class: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological 70/30 split inside every ``(source_file, class)`` group.

    Each group is ordered by ``order_column``; the earliest
    ``floor(n * (1 - test_size))`` rows train and the remainder test.  Groups of
    two or more always keep at least one row on each side, so a class confined
    to a single short burst (Heartbleed: 11 rows) still trains.  A group of one
    row goes entirely to train, since a class cannot be learned from nothing.

    Returns ``(train, calibration, test)``.  Calibration is always empty: the
    delivery protocol is a pure 70/30 with no threshold-selection partition,
    matching :func:`src.data.deterministic_split.deterministic_source_split`.

    Deterministic and RNG-free -- the result depends only on row content, so
    shuffling the input frame yields an identical split. A non-unique index is
    reset on entry, so a filtered or concatenated frame is accepted as-is.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1); got {test_size}")
    if min_test_per_class < 0:
        raise ValueError("min_test_per_class must be >= 0")
    missing_columns = {label_column, source_column, order_column} - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Temporal split is missing columns: {sorted(missing_columns)}"
        )

    # Positions are resolved with Index.get_indexer, which requires a unique
    # index. Callers naturally pass filtered or concatenated frames, so
    # normalise rather than raising InvalidIndexError from deep inside pandas.
    if not frame.index.is_unique:
        frame = frame.reset_index(drop=True)

    train_positions: list[np.ndarray] = []
    test_positions: list[np.ndarray] = []

    # Positional (not label) indexes throughout, so an unreset input index
    # cannot silently corrupt the selection.
    for _, group in frame.groupby([source_column, label_column], sort=True):
        positions = frame.index.get_indexer(group.index)
        order = group[order_column].to_numpy()
        # Stable sort keeps ties in their original file order.
        chronological = positions[np.argsort(order, kind="stable")]

        n = chronological.size
        if n == 1:
            train_positions.append(chronological)
            continue
        n_train = int(np.floor(n * (1.0 - test_size)))
        n_train = max(1, min(n - 1, n_train))
        train_positions.append(chronological[:n_train])
        test_positions.append(chronological[n_train:])

    train_idx = np.sort(np.concatenate(train_positions)) if train_positions else np.array([], dtype=np.int64)
    test_idx = np.sort(np.concatenate(test_positions)) if test_positions else np.array([], dtype=np.int64)

    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError(
            f"Temporal split produced an empty side (train={train_idx.size}, "
            f"test={test_idx.size})"
        )
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError("Temporal split produced overlapping train/test rows")

    train = frame.iloc[train_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)

    if min_test_per_class:
        short = {
            str(label): int(count)
            for label, count in test[label_column].value_counts().items()
            if count < min_test_per_class
        }
        if short:
            raise ValueError(
                f"Temporal split left classes below min_test_per_class="
                f"{min_test_per_class}: {short}. Lower min_test_per_class or "
                "drop those classes before splitting."
            )

    calibration = frame.iloc[0:0].reset_index(drop=True)
    return train, calibration, test


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def build_temporal_manifest(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    label_column: str = "Label",
    source_column: str = "source_file",
    test_size: float = 0.30,
    min_test_per_class: int = 3,
    version: str = TEMPORAL_MANIFEST_VERSION,
    path: Path | str | None = None,
) -> dict[str, Any]:
    """Record a produced split so later runs can be verified against it."""
    classes = sorted(set(train[label_column]) | set(test[label_column]))
    train_counts = train[label_column].value_counts()
    test_counts = test[label_column].value_counts()
    payload: dict[str, Any] = {
        "version": version,
        "dataset_id": DATASET_ID,
        "order_basis": ORDER_BASIS,
        "split_strategy": "per_source_per_class_chronological",
        "sources": sorted({str(name) for name in train[source_column].unique()}
                          | {str(name) for name in test[source_column].unique()}),
        "test_size": float(test_size),
        "min_test_per_class": int(min_test_per_class),
        "calibration_size": 0.0,
        "final_test_locked": True,
        "random_row_split": False,
        "expected_counts": {
            str(name): {
                "train": int(train_counts.get(name, 0)),
                "test": int(test_counts.get(name, 0)),
            }
            for name in classes
        },
        "expected_train_rows": int(len(train)),
        "expected_test_rows": int(len(test)),
        "chronology_validated": (
            "Row order verified against the published CIC CICIDS2017 attack "
            "schedule by temporal_split.validate_raw_label_chronology"
        ),
    }
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def is_temporal_manifest(payload: dict[str, Any]) -> bool:
    """Whether a loaded manifest belongs to this protocol."""
    return str(payload.get("version", "")).startswith(TEMPORAL_MANIFEST_PREFIX)


def load_temporal_manifest(path: Path | str) -> dict[str, Any]:
    """Load and validate a temporal split manifest.

    Unlike :func:`src.data.deterministic_split.load_split_manifest`, sources and
    expected counts live in the JSON rather than in compiled-in constants, so a
    re-derived split is recorded by writing a new manifest instead of editing
    Python.  Every structural field is still checked.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Temporal split manifest must be a JSON object")

    version = str(payload.get("version", ""))
    if not version.startswith(TEMPORAL_MANIFEST_PREFIX):
        raise ValueError(
            f"Unsupported temporal manifest version {version!r}; expected one "
            f"starting with {TEMPORAL_MANIFEST_PREFIX!r}"
        )
    if payload.get("dataset_id") != DATASET_ID:
        raise ValueError(
            f"Temporal manifest dataset_id must be {DATASET_ID!r}; "
            f"got {payload.get('dataset_id')!r}"
        )
    if payload.get("order_basis") != ORDER_BASIS:
        raise ValueError(
            f"Temporal manifest order_basis must be {ORDER_BASIS!r}; "
            f"got {payload.get('order_basis')!r}"
        )
    if payload.get("final_test_locked") is not True:
        raise ValueError("Temporal manifest must set final_test_locked = true")
    if payload.get("random_row_split") is not False:
        raise ValueError("Temporal manifest must set random_row_split = false")
    if float(payload.get("calibration_size", -1.0)) != 0.0:
        raise ValueError("Temporal manifest must set calibration_size = 0.0")

    test_size = float(payload.get("test_size", -1.0))
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"Temporal manifest test_size must be in (0, 1); got {test_size}")

    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Temporal manifest must list a non-empty 'sources' array")
    if len(set(sources)) != len(sources):
        raise ValueError("Temporal manifest 'sources' contains duplicates")
    unknown = sorted(set(map(str, sources)) - set(CICIDS2017_SOURCES))
    if unknown:
        raise ValueError(f"Temporal manifest lists unknown CICIDS2017 sources: {unknown}")

    expected = payload.get("expected_counts")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Temporal manifest must carry a non-empty 'expected_counts'")
    counts: dict[str, dict[str, int]] = {}
    for name, entry in expected.items():
        if not isinstance(entry, dict) or {"train", "test"} - set(entry):
            raise ValueError(
                f"expected_counts[{name!r}] must hold both 'train' and 'test'"
            )
        counts[str(name)] = {"train": int(entry["train"]), "test": int(entry["test"])}

    for side in ("train", "test"):
        declared = int(payload.get(f"expected_{side}_rows", -1))
        total = sum(entry[side] for entry in counts.values())
        if declared != total:
            raise ValueError(
                f"expected_{side}_rows ({declared}) does not match the sum of "
                f"expected_counts ({total})"
            )

    return {**payload, "sources": tuple(map(str, sources)), "expected_counts": counts,
            "test_size": test_size,
            "min_test_per_class": int(payload.get("min_test_per_class", 0))}


def verify_split_against_manifest(
    train: pd.DataFrame,
    test: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    label_column: str = "Label",
) -> dict[str, Any]:
    """Compare a produced split with the counts its manifest advertises.

    The source-holdout protocol never checked its own ``expected_*_rows``
    against reality; this closes that gap for the temporal protocol.
    """
    expected = manifest["expected_counts"]
    train_counts = train[label_column].value_counts()
    test_counts = test[label_column].value_counts()

    mismatches: list[str] = []
    for name in sorted(set(expected) | set(train_counts.index) | set(test_counts.index)):
        want = expected.get(str(name), {"train": 0, "test": 0})
        got = {
            "train": int(train_counts.get(name, 0)),
            "test": int(test_counts.get(name, 0)),
        }
        for side in ("train", "test"):
            if want[side] != got[side]:
                mismatches.append(
                    f"{name} {side}: manifest {want[side]}, actual {got[side]}"
                )

    return {
        "valid": not mismatches,
        "mismatches": mismatches,
        "actual_train_rows": int(len(train)),
        "actual_test_rows": int(len(test)),
        "expected_train_rows": int(manifest["expected_train_rows"]),
        "expected_test_rows": int(manifest["expected_test_rows"]),
    }
