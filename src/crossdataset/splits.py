"""Three-way 60/10/30 partitioning, chronological or random.

The protocol specifies train 60% / calibration 10% / test 30%, repeated over
five seeds. Two split modes are provided because they answer different
questions and disagreeing about which to use would be worse than running both:

``random``
    Stratified by class, seeded. This is the protocol's own split and the one
    whose numbers are directly comparable with the advisor's.

``chronological``
    Within every ``(group, class)`` bucket the earliest 60% train, the next 10%
    calibrate and the latest 30% test. No test flow precedes the training flows
    of its own class. This is the split the 2017 handoff used (at 70/30) and
    the one that does not let a model see the back half of an attack window
    while being scored on the front half.

Splitting inside ``(group, class)`` rather than over the corpus as a whole is
forced by how these captures were made: each attack class occurs in exactly one
capture file, so a corpus-wide time cutoff would hand entire classes to one
side. The 2017 handoff documents the same reasoning.

**Ordering keys differ between the two datasets and that is unavoidable.** The
*MachineLearningCVE* variant of CICIDS2017 ships no ``Timestamp`` column, so the
2017 corpus orders by ``_row_index``, the original CSV row position, which the
handoff validated against the published attack timetable. IDS2018 has a real
``Timestamp``. Both are chronological; they are simply derived differently, and
any write-up should say so.

A class too small to divide goes entirely to train. A test row whose class has
no training rows measures the protocol rather than the model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRAIN_FRAC, CALIB_FRAC, TEST_FRAC = 0.60, 0.10, 0.30


@dataclass(frozen=True)
class Partition:
    """Positional indices for the three parts, plus how it was made."""

    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    mode: str
    seed: int

    def __post_init__(self) -> None:
        sizes = {len(self.train), len(self.calibration), len(self.test)}
        total = len(self.train) + len(self.calibration) + len(self.test)
        overlap = (
            set(self.train.tolist())
            & set(self.test.tolist())
            | set(self.train.tolist()) & set(self.calibration.tolist())
            | set(self.calibration.tolist()) & set(self.test.tolist())
        )
        if overlap:
            raise ValueError(f"partitions overlap on {len(overlap)} row(s)")
        if not sizes or total == 0:
            raise ValueError("empty partition")

    def as_dict(self) -> dict[str, int | str]:
        return {
            "mode": self.mode,
            "seed": self.seed,
            "n_train": len(self.train),
            "n_calibration": len(self.calibration),
            "n_test": len(self.test),
        }


def _cut(n: int) -> tuple[int, int]:
    """Return the two cut points for a bucket of ``n`` rows.

    Guarantees: a bucket of 1 is all train; a bucket of 2+ always yields at
    least one test row, because a class present in training but absent from test
    cannot be scored at all. Calibration is the part that yields -- it is held
    out and unused in this pass, so starving it costs nothing measurable.
    """
    if n <= 1:
        return n, n
    if n == 2:
        return 1, 1
    n_train = max(1, int(round(n * TRAIN_FRAC)))
    n_calib = int(round(n * CALIB_FRAC))
    if n_train + n_calib >= n:
        n_calib = max(0, n - n_train - 1)
    return n_train, n_train + n_calib


def chronological_split(
    y: pd.Series,
    order_key: pd.Series,
    group: pd.Series | None = None,
    seed: int = 42,
) -> Partition:
    """Earliest 60% / next 10% / latest 30% inside every ``(group, class)``.

    ``seed`` is recorded but never consumed: the partition is a function of the
    data alone, so re-running under a different seed yields identical rows. That
    is a feature -- it means seed-to-seed variation in the results is model
    variance, uncontaminated by split variance.
    """
    if not (len(y) == len(order_key)) or (group is not None and len(group) != len(y)):
        raise ValueError("y, order_key and group must be the same length")

    labels = y.to_numpy()
    order = order_key.to_numpy()
    groups = np.zeros(len(y), dtype=int) if group is None else group.to_numpy()

    train: list[np.ndarray] = []
    calib: list[np.ndarray] = []
    test: list[np.ndarray] = []

    frame = pd.DataFrame({"_g": groups, "_y": labels})
    for _, bucket in frame.groupby(["_g", "_y"], sort=True):
        pos = bucket.index.to_numpy()
        pos = pos[np.argsort(order[pos], kind="stable")]
        a, b = _cut(len(pos))
        train.append(pos[:a])
        calib.append(pos[a:b])
        test.append(pos[b:])

    part = Partition(
        train=np.sort(np.concatenate(train)) if train else np.array([], dtype=int),
        calibration=np.sort(np.concatenate(calib)) if calib else np.array([], dtype=int),
        test=np.sort(np.concatenate(test)) if test else np.array([], dtype=int),
        mode="chronological",
        seed=seed,
    )
    logger.info("chronological split: %s", part.as_dict())
    return part


def random_split(y: pd.Series, seed: int = 42) -> Partition:
    """Stratified 60/10/30, shuffled within each class using ``seed``."""
    labels = y.to_numpy()
    rng = np.random.RandomState(seed)

    train: list[np.ndarray] = []
    calib: list[np.ndarray] = []
    test: list[np.ndarray] = []

    for cls in pd.unique(labels):
        pos = np.flatnonzero(labels == cls)
        rng.shuffle(pos)
        a, b = _cut(len(pos))
        train.append(pos[:a])
        calib.append(pos[a:b])
        test.append(pos[b:])

    part = Partition(
        train=np.sort(np.concatenate(train)),
        calibration=np.sort(np.concatenate(calib)),
        test=np.sort(np.concatenate(test)),
        mode="random",
        seed=seed,
    )
    logger.info("random split: %s", part.as_dict())
    return part
