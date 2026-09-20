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

Both sides enter this pipeline as a **whole corpus** -- 2.4M rows for 2017,
11.4M for 2018 -- so both reach 300,000 by the same route. That was not always
true: ``load_ids2018`` used to read ``sample_300k.parquet``, a draw the 2018
training pipeline had already taken from ~13M rows *before* any cleaning. The
shared cleaning then removed a further 37,299 rows from it, almost all exact
duplicates, so 2018 arrived at 262,701 rows with a class balance no longer the
one the sampler had designed, while 2017 arrived at a full 300,000. The gap
measured between them carried that difference inside it. ``_finish`` now
refuses an input that is not a whole corpus, so the asymmetry cannot come back
unnoticed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.crossdataset.cleaning import CleaningReport, clean
from src.crossdataset.labels import map_to_shared
from src.crossdataset.sampling import sample
from src.crossdataset.schema import canonical_features, to_canonical

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IDS2017_PARQUET = PROJECT_ROOT / "data" / "processed" / "cicids2017_clean.parquet"
IDS2018_PARQUET = PROJECT_ROOT / "data" / "ids2018" / "cicids2018_clean.parquet"

LABEL_COL = "Label"

#: Below this many rows an input is a pre-drawn sample, not a corpus. Both real
#: inputs clear it by a wide margin (2.4M and 11.4M); the pre-sampled 2018 file
#: this loader used to read would have failed at 300,000.
MIN_FULL_CORPUS_ROWS = 1_000_000

#: Collapsing each source vocabulary onto the shared seven is many-to-one, so a
#: few rows legitimately become duplicates of each other at that point. More
#: than this share means the input was not cleaned with these rules at all.
MAX_LABEL_COLLAPSE_DUPLICATES = 0.001


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
    datasets cannot drift apart through a copy-paste edit to only one of them.

    Three guards make the "same conditions" claim checkable rather than assumed:

    * the input must be a whole corpus, so cleaning precedes sampling on both
      sides rather than only on one;
    * cleaning must drop no non-finite row, because both parquets were built
      with these exact rules and that step does not depend on the label; a
      handful of duplicates may still appear, since collapsing each source
      vocabulary onto the shared seven is many-to-one;
    * the draw must reach ``target``, since a side that lands short is no
      longer being compared at the protocol's size.
    """
    feature_cols = list(canonical_features(keep_dst_port=keep_dst_port))

    mapped = map_to_shared(labels, name)
    keep = mapped.notna().to_numpy()
    dropped_unmapped = int((~keep).sum())

    work = frame.loc[keep, feature_cols].copy()
    work[LABEL_COL] = mapped[keep].to_numpy()
    work["_order"] = order_key[keep].to_numpy()
    work["_group"] = group[keep].to_numpy()

    if len(work) < MIN_FULL_CORPUS_ROWS:
        raise ValueError(
            f"{name}: only {len(work):,} rows reached cleaning, below the "
            f"{MIN_FULL_CORPUS_ROWS:,} that marks a whole corpus. This loader must be "
            "given the full corpus so that cleaning precedes sampling; a file that was "
            "already sampled upstream puts this dataset through a different order of "
            "operations than the other one."
        )

    work, report = clean(work, feature_cols, LABEL_COL)

    # Dropping non-finite rows does not depend on the label, so a corpus built
    # with these rules must yield exactly zero here. Anything else means the
    # rules have drifted from the ones that produced the file.
    if report.rows_dropped_non_finite:
        raise ValueError(
            f"{name}: cleaning removed {report.rows_dropped_non_finite:,} non-finite "
            "row(s) from a corpus that should already satisfy these rules. Rebuild "
            "the parquet, or the two datasets are passing through different filters."
        )

    # Duplicates are different: the label collapse above is many-to-one, so rows
    # that were distinct under the source vocabulary can coincide under the
    # shared one. On 2018 this is 13 rows -- pairs of identical flows labelled
    # FTP-BruteForce and SSH-Bruteforce, or two of the three web variants, which
    # the 15-class file rightly kept apart. 2017 yields none because its parquet
    # already carries grouped labels. Both corpora therefore leave this function
    # deduplicated on the *shared* vocabulary, which is the symmetric outcome;
    # demanding zero here would demand that the two files be labelled at the
    # same granularity, which is a different and unnecessary requirement.
    #
    # A large drop is still drift: the pre-sampled 2018 file this loader used to
    # read lost 35,561 of 300,000 rows (11.9%) to duplicates alone.
    duplicate_fraction = report.rows_dropped_duplicate / max(report.rows_in, 1)
    if duplicate_fraction > MAX_LABEL_COLLAPSE_DUPLICATES:
        raise ValueError(
            f"{name}: cleaning removed {report.rows_dropped_duplicate:,} duplicate "
            f"row(s), {duplicate_fraction:.2%} of the corpus. Collapsing labels can "
            f"create a handful, but not more than "
            f"{MAX_LABEL_COLLAPSE_DUPLICATES:.2%} -- this input was not cleaned with "
            "these rules."
        )
    if report.rows_dropped_duplicate:
        logger.info(
            "%s: %d row(s) became duplicates once labels collapsed to the shared seven",
            name, report.rows_dropped_duplicate,
        )

    work = sample(work, LABEL_COL, target=target, seed=seed)
    if len(work) < target:
        raise ValueError(
            f"{name}: sampled {len(work):,} rows against a target of {target:,}. "
            "Both datasets must be compared at the same size."
        )
    if len(work) > target:
        # allocate() documents this: rare classes taken whole can exceed target.
        logger.warning(
            "%s: %s rows against a target of %s; rare classes alone exceed it",
            name, f"{len(work):,}", f"{target:,}",
        )

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
    """Load the full CSE-CIC-IDS2018 corpus.

    Built by ``scripts/build_ids2018_full.py`` from the ten raw daily CSVs:
    16,232,943 rows streamed into the canonical schema, then put through the
    shared cleaning rules once, leaving 11,445,955. Cleaning here is therefore a
    no-op, exactly as it is for 2017, and the protocol's 300,000 rows are drawn
    from the whole corpus on both sides.

    ``Timestamp`` is the ordering key and the capture day is the split group;
    unlike 2017 this export carries real wall-clock times.
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
