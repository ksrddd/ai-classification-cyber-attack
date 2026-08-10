"""Bring both corpora to the same shape: 77 features, 7 classes, same cleaning.

Each loader returns a :class:`Corpus` -- features already renamed into the
canonical schema, labels already collapsed to the seven shared classes, the
shared cleaning rules already applied, and an ordering key for the
chronological split. From that point on the two datasets are interchangeable,
which is the whole reason this package exists.

Order of operations is fixed and matters:

1. rename to the canonical schema
2. collapse labels to the shared seven, dropping rows with no counterpart
3. clean
4. sample to the protocol's 300,000 rows

Dropping unmapped classes *before* sampling means the 300,000 rows are 300,000
usable rows. Sampling first would spend part of the budget on 2017's PortScan
and Heartbleed, which are then discarded, leaving fewer than asked for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.crossdataset.cleaning import CleaningReport, clean
from src.crossdataset.labels import map_to_shared
from src.crossdataset.sampling import sample
from src.crossdataset.schema import canonical_features, to_canonical

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IDS2017_PARQUET = PROJECT_ROOT / "data" / "processed" / "cicids2017_clean.parquet"
IDS2018_PARQUET = PROJECT_ROOT / "data" / "ids2018" / "sample_300k.parquet"

LABEL_COL = "Label"


@dataclass
class Corpus:
    """One dataset, canonicalised and ready to split."""

    name: str
    X: pd.DataFrame
    y: pd.Series
    order_key: pd.Series
    group: pd.Series
    order_basis: str
    cleaning: CleaningReport
    dropped_unmapped: int

    def class_counts(self) -> pd.Series:
        return self.y.value_counts()

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "n_rows": len(self.X),
            "n_features": self.X.shape[1],
            "order_basis": self.order_basis,
            "dropped_unmapped_rows": self.dropped_unmapped,
            "cleaning": self.cleaning.as_dict(),
            "class_counts": {k: int(v) for k, v in self.class_counts().items()},
        }


def _finish(
    name: str,
    frame: pd.DataFrame,
    labels: pd.Series,
    order_key: pd.Series,
    group: pd.Series,
    order_basis: str,
    *,
    keep_dst_port: bool,
    target: int,
    seed: int,
) -> Corpus:
    """Shared tail: drop unmapped, clean, sample. Kept in one place so the two
    datasets cannot drift apart through a copy-paste edit to only one of them."""
    feature_cols = list(canonical_features(keep_dst_port=keep_dst_port))

    mapped = map_to_shared(labels, name)
    keep = mapped.notna().to_numpy()
    dropped_unmapped = int((~keep).sum())

    work = frame.loc[keep, feature_cols].copy()
    work[LABEL_COL] = mapped[keep].to_numpy()
    work["_order"] = order_key[keep].to_numpy()
    work["_group"] = group[keep].to_numpy()

    work, report = clean(work, feature_cols, LABEL_COL)
    work = sample(work, LABEL_COL, target=target, seed=seed)

    corpus = Corpus(
        name=name,
        X=work[feature_cols].reset_index(drop=True),
        y=work[LABEL_COL].reset_index(drop=True),
        order_key=work["_order"].reset_index(drop=True),
        group=work["_group"].reset_index(drop=True),
        order_basis=order_basis,
        cleaning=report,
        dropped_unmapped=dropped_unmapped,
    )
    logger.info("%s ready: %s", name, corpus.as_dict())
    return corpus


def load_ids2017(
    *, keep_dst_port: bool = True, target: int = 300_000, seed: int = 42,
    path: Path = IDS2017_PARQUET,
) -> Corpus:
    """Load the handoff's cleaned 2017 corpus.

    ``_row_index`` is the ordering key -- this variant of CICIDS2017 has no
    ``Timestamp`` column, and the handoff validated that CSV row order matches
    the published attack timetable. ``source_file`` groups the split, because
    each attack class lives in exactly one capture.
    """
    df = pd.read_parquet(path)
    for col in ("_row_index", "source_file"):
        if col not in df.columns:
            raise KeyError(f"2017 corpus is missing {col!r}; the split needs it")
    return _finish(
        "ids2017",
        to_canonical(df, "ids2017", keep_dst_port=keep_dst_port),
        df[LABEL_COL].astype(str),
        df["_row_index"],
        df["source_file"].astype(str),
        "_row_index",
        keep_dst_port=keep_dst_port,
        target=target,
        seed=seed,
    )


def load_ids2018(
    *, keep_dst_port: bool = True, target: int = 300_000, seed: int = 42,
    path: Path = IDS2018_PARQUET,
) -> Corpus:
    """Load the 2018 sample.

    This file is already a 300,000-row stratified draw from the full corpus, so
    ``target`` mostly does nothing here -- the cleaning step removes about one
    percent and the sampler then has nothing to trim. Said plainly because it
    means the two sides reach 300,000 by different routes: 2017 is subsampled
    from 2.5M rows, 2018 was subsampled before it reached this repository.
    """
    df = pd.read_parquet(path)
    ts = pd.to_datetime(df["Timestamp"], format="mixed", dayfirst=True, errors="coerce")
    if ts.isna().any():
        raise ValueError(f"{int(ts.isna().sum())} unparseable timestamp(s) in 2018")
    return _finish(
        "ids2018",
        to_canonical(df, "ids2018", keep_dst_port=keep_dst_port),
        df[LABEL_COL].astype(str),
        ts.astype("int64"),
        ts.dt.date.astype(str),
        "Timestamp",
        keep_dst_port=keep_dst_port,
        target=target,
        seed=seed,
    )


def load_both(**kwargs) -> dict[str, Corpus]:
    return {"ids2017": load_ids2017(**kwargs), "ids2018": load_ids2018(**kwargs)}
