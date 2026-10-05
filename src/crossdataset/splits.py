"""Two-way 70/30 partitioning, chronological or random.

Train 70% / test 30%, repeated over five seeds, the same shape as the 2017
handoff. There is no calibration part: an earlier 60/10/30 version held 10% out
and never read it, so the rows were simply lost. Two split modes are provided
because they answer different questions and disagreeing about which to use would
be worse than running both:

``random``
    Stratified by class, seeded. This is the protocol's own split and the one
    whose numbers are directly comparable with the advisor's.

``chronological``
    Within every ``(group, class)`` bucket the earliest 70% train and the latest
    30% test. No test flow precedes the training flows of its own class. This is
    the split the 2017 handoff used and the one that does not let a model see
    the back half of an attack window while being scored on the front half.

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

TRAIN_FRAC, TEST_FRAC = 0.70, 0.30


@dataclass(frozen=True)
class Partition:
    """Positional indices for the two parts, plus how it was made."""

    train: np.ndarray
    test: np.ndarray
    mode: str
    seed: int

    def __post_init__(self) -> None:
        if len(self.train) + len(self.test) == 0:
            raise ValueError("empty partition")
        overlap = set(self.train.tolist()) & set(self.test.tolist())
        if overlap:
            raise ValueError(f"partitions overlap on {len(overlap)} row(s)")

    def as_dict(self) -> dict[str, int | str]:
        return {
            "mode": self.mode,
            "seed": self.seed,
            "n_train": len(self.train),
            "n_test": len(self.test),
        }


def _cut(n: int) -> int:
    """Return how many of a bucket's ``n`` rows go to train.

    Guarantees: a bucket of 1 is all train; a bucket of 2+ always yields at
    least one test row, because a class present in training but absent from test
    cannot be scored at all.
    """
    if n <= 1:
        return n
    return min(max(1, int(round(n * TRAIN_FRAC))), n - 1)


def chronological_split(
    y: pd.Series,
    order_key: pd.Series,
    group: pd.Series | None = None,
    seed: int = 42,
) -> Partition:
    """Earliest 70% train / latest 30% test inside every ``(group, class)``.

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
    test: list[np.ndarray] = []

    frame = pd.DataFrame({"_g": groups, "_y": labels})
    for _, bucket in frame.groupby(["_g", "_y"], sort=True):
        pos = bucket.index.to_numpy()
        pos = pos[np.argsort(order[pos], kind="stable")]
        cut = _cut(len(pos))
        train.append(pos[:cut])
        test.append(pos[cut:])

    part = Partition(
        train=np.sort(np.concatenate(train)) if train else np.array([], dtype=int),
        test=np.sort(np.concatenate(test)) if test else np.array([], dtype=int),
        mode="chronological",
        seed=seed,
    )
    logger.info("chronological split: %s", part.as_dict())
    return part


def random_split(y: pd.Series, seed: int = 42) -> Partition:
    """Stratified 70/30, shuffled within each class using ``seed``."""
    labels = y.to_numpy()
    rng = np.random.RandomState(seed)

    train: list[np.ndarray] = []
    test: list[np.ndarray] = []

    for cls in pd.unique(labels):
        pos = np.flatnonzero(labels == cls)
        rng.shuffle(pos)
        cut = _cut(len(pos))
        train.append(pos[:cut])
        test.append(pos[cut:])

    part = Partition(
        train=np.sort(np.concatenate(train)),
        test=np.sort(np.concatenate(test)),
        mode="random",
        seed=seed,
    )
    logger.info("random split: %s", part.as_dict())
    return part
