"""Chronological train/test split for CSE-CIC-IDS2018.

Why a separate module from :mod:`src.data.temporal_split`
---------------------------------------------------------
The 2017 helper orders rows by ``_row_index`` because the *MachineLearningCVE*
CSVs ship no timestamp, and it validates that row order against CIC's published
attack timetable to earn the right to call it chronological. None of that
applies here: the 2018 exports carry a real ``Timestamp`` column, so ordering is
direct and needs no proxy and no schedule cross-check. Importing the 2017 module
would drag in ``CICIDS2017_SCHEDULE`` and a manifest validator that rejects any
``dataset_id`` other than ``CICIDS2017``.

The protocol
------------
For every ``(capture_day, class)`` group the earliest ``1 - test_size`` of the
flows train the model and the latest ``test_size`` test it. Grouping by day as
well as class matters for the two classes that were run on more than one day
(``Infilteration``, ``Brute Force -Web``): a single global cutoff per class
would send one whole day to test and leave the model never having seen that
day's conditions.

Timestamp is an ordering key here, never a feature
--------------------------------------------------
``config.TIMESTAMP_COL`` is dropped before training precisely because each
attack ran inside its own capture window, which makes wall-clock time a
near-perfect label proxy. That reasoning is unchanged. Using the column to
*decide which rows are held out* leaks nothing: the split is a property of the
evaluation protocol, not an input the model can read at predict time. The
driver drops the column immediately after splitting, and
:func:`assert_timestamp_absent` is the guard that proves it.

What this protocol is and is not
--------------------------------
Twelve of the fifteen classes occur on exactly one capture day, so for those the
split is chronological *within a single attack burst* rather than across
independent days. That is a weaker claim than the 2017 protocol makes and it
should be stated as such in any write-up: it removes the random-split
optimism -- no test flow precedes the training flows of its own class -- but it
does not demonstrate generalisation to a future, separately-staged campaign.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TEMPORAL_MANIFEST_VERSION = "ids2018_temporal_v1"
DATASET_ID = "CSE-CIC-IDS2018"
ORDER_BASIS = "timestamp"
SPLIT_PROTOCOL = "ids2018_temporal_v1"


def parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse the 2018 ``Timestamp`` column to datetimes.

    The exports use ``DD/MM/YYYY HH:MM:SS``, so ``dayfirst`` is required --
    without it pandas reads 02/03/2018 as 3 February and the March capture
    lands before the February ones, silently inverting the split.

    Raises rather than coercing: a NaT would be sorted to one end of its group
    and quietly biased the split, and every row in this corpus does parse.
    """
    parsed = pd.to_datetime(values, dayfirst=True, errors="coerce")
    bad = int(parsed.isna().sum())
    if bad:
        sample = values[parsed.isna()].head(3).tolist()
        raise ValueError(
            f"{bad} timestamp(s) failed to parse, e.g. {sample}. The temporal "
            "split cannot order rows it cannot date."
        )
    return parsed


def temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    *,
    test_size: float = 0.30,
    min_test_per_class: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chronological split inside every ``(capture_day, class)`` group.

    Groups of one row go entirely to train: a class cannot be learned from
    nothing, and a test row whose class has no training rows measures the
    protocol rather than the model. Groups of two or more always keep at least
    one row on each side.

    Deterministic and RNG-free -- the result depends only on row content, so
    re-running or reshuffling the input yields an identical split.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1); got {test_size}")
    if len(X) != len(y) or len(X) != len(timestamps):
        raise ValueError(
            f"Length mismatch: X={len(X)}, y={len(y)}, timestamps={len(timestamps)}"
        )

    ts = parse_timestamps(timestamps)
    # Positional indexing throughout, so a non-unique or pre-filtered index on
    # the caller's frame cannot silently corrupt the selection.
    day = ts.dt.date.to_numpy()
    labels = y.to_numpy()
    order = ts.to_numpy()

    train_positions: list[np.ndarray] = []
    test_positions: list[np.ndarray] = []

    keys = pd.DataFrame({"day": day, "label": labels})
    for _, group in keys.groupby(["day", "label"], sort=True):
        positions = group.index.to_numpy()
        # Stable sort keeps same-second ties in their original file order.
        chronological = positions[np.argsort(order[positions], kind="stable")]

        n = chronological.size
        if n == 1:
            train_positions.append(chronological)
            continue
        n_train = int(np.floor(n * (1.0 - test_size)))
        n_train = max(1, min(n - 1, n_train))
        train_positions.append(chronological[:n_train])
        test_positions.append(chronological[n_train:])

    train_idx = np.sort(np.concatenate(train_positions))
    test_idx = np.sort(np.concatenate(test_positions))

    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError(
            f"Temporal split produced an empty side "
            f"(train={train_idx.size}, test={test_idx.size})"
        )
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError("Temporal split produced overlapping train/test rows")

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    if min_test_per_class:
        short = {
            str(label): int(count)
            for label, count in y_test.value_counts().items()
            if count < min_test_per_class
        }
        if short:
            raise ValueError(
                f"Temporal split left classes below min_test_per_class="
                f"{min_test_per_class}: {short}"
            )

    # Classes that reach test with almost no support are not an error, but they
    # are the single most misreadable thing this split produces, so they are
    # named in the run log rather than left for someone to notice in a table.
    thin = {
        str(label): int(count)
        for label, count in y_test.value_counts().items()
        if count < 10
    }
    logger.info(
        "Temporal split -> train %s rows / test %s rows (%.0f/%.0f)",
        f"{len(X_train):,}",
        f"{len(X_test):,}",
        100 * (1 - test_size),
        100 * test_size,
    )
    if thin:
        logger.warning(
            "Classes with under 10 test flows -- their scores describe the "
            "sample, not the model: %s",
            ", ".join(f"{name} ({count})" for name, count in sorted(thin.items())),
        )
    return X_train, X_test, y_train, y_test


def assert_timestamp_absent(X: pd.DataFrame, timestamp_col: str) -> None:
    """Fail loudly if the ordering key survived into the feature matrix.

    The whole justification for reading Timestamp during the split is that it
    never reaches the model. That is worth an assertion rather than a comment:
    leaking it would inflate every score in the bundle and the resulting table
    would look like a strong result.
    """
    if timestamp_col in X.columns:
        raise AssertionError(
            f"{timestamp_col!r} is still in the feature matrix. It is a "
            "near-perfect label proxy (each attack ran in its own capture "
            "window) and must be dropped after the split."
        )


def build_temporal_manifest(
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    test_size: float,
    min_test_per_class: int,
    sample_size: int,
    path: Path | str | None = None,
) -> dict[str, Any]:
    """Record the produced split so a later run can be verified against it."""
    classes = sorted(set(y_train) | set(y_test))
    train_counts = y_train.value_counts()
    test_counts = y_test.value_counts()

    payload: dict[str, Any] = {
        "version": TEMPORAL_MANIFEST_VERSION,
        "dataset_id": DATASET_ID,
        "order_basis": ORDER_BASIS,
        "split_strategy": "per_capture_day_per_class_chronological",
        "split_protocol": SPLIT_PROTOCOL,
        "sample_size": int(sample_size),
        "test_size": float(test_size),
        "min_test_per_class": int(min_test_per_class),
        "random_row_split": False,
        "final_test_locked": True,
        "expected_counts": {
            str(name): {
                "train": int(train_counts.get(name, 0)),
                "test": int(test_counts.get(name, 0)),
            }
            for name in classes
        },
        "expected_train_rows": int(len(y_train)),
        "expected_test_rows": int(len(y_test)),
        "caveat": (
            "12 of 15 classes occur on a single capture day, so for those the "
            "ordering is within one attack burst rather than across days."
        ),
    }
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        logger.info("Split manifest written to %s", path)
    return payload
