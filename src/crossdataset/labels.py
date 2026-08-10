"""Shared 7-class label vocabulary for CICIDS2017 <-> CSE-CIC-IDS2018.

The two datasets label the same attack families differently: 2017 runs its
attacks under nine names, 2018 under fifteen, and only some of them line up.
This module collapses both onto the seven classes they have in common, which
is the vocabulary the advisor's protocol specifies.

The seven are the natural intersection, not a choice:

===============  ==========================  ================================
Shared class     CICIDS2017 source           CSE-CIC-IDS2018 source
===============  ==========================  ================================
Benign           BENIGN                      Benign
Bot              Bot                         Bot
Infiltration     Infiltration                Infilteration (sic)
DDoS             DDoS                        three DDOS/DDoS variants
DoS              four DoS variants           four DoS variants
Brute Force      FTP-Patator, SSH-Patator    FTP-BruteForce, SSH-Bruteforce
Web Attack       three Web Attack variants   Brute Force -Web/-XSS, SQL Inj.
===============  ==========================  ================================

Two 2017 classes have no 2018 counterpart and are dropped: ``Heartbleed`` (11
test flows in the temporal bundle) and ``PortScan``. Dropping is the honest
option -- folding PortScan into another family would invent an equivalence the
capture does not support.

What the collapse does and does not fix
---------------------------------------
Measured on the 2018 300k sample (2026-08-09), collapsing 15 classes to 7
raises lightgbm's all-class macro-F1 from 0.7126 to 0.7880 while *lowering*
its macro over classes with enough test flows to measure from 0.8575 to
0.8193. The same sign flip appears on every model tested. Three effects:

1. ``Infilteration`` maps one-to-one, so the genuinely unlearnable class is
   untouched -- F1 0.1185 before, 0.1173 after.
2. Merging drags healthy classes down. ``SSH-Bruteforce`` scored 1.0000 and
   ``FTP-BruteForce`` 0.7946; together they score 0.8860, and the fact that
   one of them was solved perfectly disappears from the report.
3. ``Web Attack`` still holds only 7 test flows after absorbing all three
   tiny classes, so it remains unmeasurable. The tiny-class problem shrinks
   from four classes to one; it does not go away.

Report macro-F1 both over all classes and over the measurable ones. Reporting
only the first states the opposite of what the data shows.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

#: The seven classes both datasets contain, in descending typical frequency.
SHARED_CLASSES: tuple[str, ...] = (
    "Benign",
    "DDoS",
    "DoS",
    "Brute Force",
    "Bot",
    "Infiltration",
    "Web Attack",
)

#: Raw CSE-CIC-IDS2018 labels -> shared vocabulary. All fifteen map; none are
#: dropped. ``Infilteration`` is the dataset's own spelling, kept verbatim so
#: the lookup matches the files rather than the corrected name.
IDS2018_TO_SHARED: dict[str, str] = {
    "Benign": "Benign",
    "Bot": "Bot",
    "Infilteration": "Infiltration",
    "DDOS attack-HOIC": "DDoS",
    "DDOS attack-LOIC-UDP": "DDoS",
    "DDoS attacks-LOIC-HTTP": "DDoS",
    "DoS attacks-GoldenEye": "DoS",
    "DoS attacks-Hulk": "DoS",
    "DoS attacks-SlowHTTPTest": "DoS",
    "DoS attacks-Slowloris": "DoS",
    "FTP-BruteForce": "Brute Force",
    "SSH-Bruteforce": "Brute Force",
    "Brute Force -Web": "Web Attack",
    "Brute Force -XSS": "Web Attack",
    "SQL Injection": "Web Attack",
}

#: CICIDS2017 grouped labels -> shared vocabulary. Keys are the output of
#: ``src.data.label_mapping.map_to_multiclass``, not the raw CSV strings, so
#: the existing 2017 normalisation (which strips the Windows-1252 en-dash out
#: of the Web Attack labels) still runs first and is not duplicated here.
IDS2017_TO_SHARED: dict[str, str] = {
    "BENIGN": "Benign",
    "Bot": "Bot",
    "Infiltration": "Infiltration",
    "DDoS": "DDoS",
    "DoS": "DoS",
    "Brute Force": "Brute Force",
    "Web Attack": "Web Attack",
}

#: 2017 classes with no 2018 counterpart. Rows carrying these are dropped from
#: a cross-dataset run rather than forced into a family they do not belong to.
IDS2017_UNMAPPED: tuple[str, ...] = ("Heartbleed", "PortScan")

_MAPPERS = {"ids2017": IDS2017_TO_SHARED, "ids2018": IDS2018_TO_SHARED}


def map_to_shared(labels: pd.Series, dataset: str) -> pd.Series:
    """Map a label column onto the shared seven, NaN where there is no match.

    NaN rather than a sentinel string: the caller drops those rows, and a
    ``"Other"`` bucket would quietly become an eighth class that exists in one
    dataset only -- exactly the kind of asymmetry this module exists to remove.

    Parameters
    ----------
    labels:
        For ``ids2018``, the raw ``Label`` column. For ``ids2017``, the output
        of :func:`src.data.label_mapping.map_to_multiclass`.
    dataset:
        ``"ids2017"`` or ``"ids2018"``.
    """
    try:
        mapper = _MAPPERS[dataset]
    except KeyError:
        raise ValueError(
            f"dataset must be 'ids2017' or 'ids2018'; got {dataset!r}"
        ) from None

    mapped = labels.astype(str).map(mapper)
    dropped = mapped.isna().sum()
    if dropped:
        names = sorted(labels[mapped.isna()].astype(str).unique())
        logger.info(
            "%s: dropping %d row(s) in %d unmapped class(es): %s",
            dataset,
            dropped,
            len(names),
            ", ".join(names),
        )
    return mapped


def shared_class_counts(labels: pd.Series, dataset: str) -> pd.Series:
    """Row count per shared class, reindexed onto all seven.

    Reindexed so a class that is absent reads as 0 instead of vanishing from
    the table -- a missing row and a zero row mean different things when
    comparing the two datasets side by side.
    """
    mapped = map_to_shared(labels, dataset)
    return mapped.value_counts().reindex(SHARED_CLASSES, fill_value=0)
