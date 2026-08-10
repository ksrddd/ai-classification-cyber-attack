"""Draw the protocol's 300,000 rows per dataset, keeping rare classes whole.

The protocol asks for 300,000 real rows per dataset "preserving rare classes as
much as possible". Two readings of that phrase give very different corpora, and
the choice matters more than it looks.

**Max-min fair allocation** -- give every class an equal share, spilling the
surplus of the small ones onto the large -- would take CICIDS2017 from 86%
BENIGN to 32% BENIGN. Every rare class survives intact, but the class balance
is no longer the one the capture actually had, the majority-class baseline
moves, and a cross-dataset gap measured against it is measuring a corpus we
invented.

**Proportional allocation with a floor** -- keep every row of any class smaller
than ``keep_all_below``, then split what is left in proportion to the original
distribution -- preserves both the rare classes and the imbalance. It is also
what the existing 2018 sample already is, so the two sides stay comparable
without re-drawing the 2018 side.

This module implements the second. On CICIDS2017 it yields roughly 83% BENIGN,
matching both the source corpus and the 2018 sample.

Sampling inside a class is by ``RandomState(seed)``, so a run is reproducible
from its seed alone, and the seed is the protocol's only source of variation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Classes with fewer rows than this are taken whole. 10,000 sits above every
#: rare class in both corpora (2017's largest "rare" class is Brute Force at
#: 9,150) and far below the three volumetric families, so the split between
#: "keep all" and "sample down" falls in a genuine gap rather than mid-cluster.
KEEP_ALL_BELOW = 10_000


def allocate(
    counts: pd.Series, target: int, keep_all_below: int = KEEP_ALL_BELOW
) -> pd.Series:
    """Return rows to draw per class: all of the small ones, pro rata for the rest.

    If the classes taken whole already exceed ``target`` the allocation is
    returned unshrunk and the caller gets more rows than asked for -- silently
    discarding rare rows to hit a round number would defeat the purpose.
    """
    counts = counts[counts > 0]
    small = counts[counts <= keep_all_below]
    large = counts[counts > keep_all_below]

    budget = target - int(small.sum())
    if budget <= 0 or large.empty:
        logger.warning(
            "rare classes alone total %d rows against a target of %d; "
            "taking every class whole",
            int(counts.sum()),
            target,
        )
        return counts.copy()

    share = large / large.sum()
    quota = (share * budget).round().astype(int)
    quota = quota.clip(upper=large)

    # Rounding leaves a few rows unspent; hand them to the largest class so the
    # total lands exactly on target.
    drift = budget - int(quota.sum())
    if drift:
        biggest = large.idxmax()
        quota[biggest] = min(int(quota[biggest]) + drift, int(large[biggest]))

    return pd.concat([small, quota]).reindex(counts.index)


def sample(
    df: pd.DataFrame,
    label_col: str,
    target: int = 300_000,
    seed: int = 42,
    keep_all_below: int = KEEP_ALL_BELOW,
) -> pd.DataFrame:
    """Draw ``target`` rows, keeping rare classes whole.

    The returned frame preserves the input row order rather than the order rows
    were drawn in: for the 2017 corpus that order *is* the chronological one
    carried by ``_row_index``, and a shuffled frame would silently break the
    chronological split downstream.
    """
    counts = df[label_col].value_counts()
    quota = allocate(counts, target, keep_all_below)
    rng = np.random.RandomState(seed)

    picks: list[np.ndarray] = []
    for cls, n in quota.items():
        pos = np.flatnonzero((df[label_col] == cls).to_numpy())
        if len(pos) > n:
            pos = rng.choice(pos, size=int(n), replace=False)
        picks.append(pos)

    selected = np.sort(np.concatenate(picks))
    out = df.iloc[selected].reset_index(drop=True)

    logger.info(
        "sampled %d rows (target %d) across %d classes with seed %d",
        len(out),
        target,
        len(quota),
        seed,
    )
    return out
