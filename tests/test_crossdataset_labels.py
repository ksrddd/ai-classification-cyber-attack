"""Tests for the shared 7-class label vocabulary.

The collapse is where a cross-dataset run can go wrong without erroring: a
2018 label that fails to match its key becomes NaN, the row is dropped, and
the only trace is a smaller sample. These tests pin that every label on both
sides is accounted for, and that the two classes with no counterpart are
dropped deliberately rather than by accident.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.crossdataset.labels import (
    IDS2017_TO_SHARED,
    IDS2017_UNMAPPED,
    IDS2018_TO_SHARED,
    SHARED_CLASSES,
    map_to_shared,
    shared_class_counts,
)

CORPUS_2018 = Path("data/ids2018/cicids2018_clean.parquet")


def test_seven_shared_classes() -> None:
    assert len(SHARED_CLASSES) == 7
    assert len(set(SHARED_CLASSES)) == 7


def test_every_2018_label_maps() -> None:
    """All fifteen 2018 labels are accounted for; none silently dropped."""
    assert len(IDS2018_TO_SHARED) == 15
    assert set(IDS2018_TO_SHARED.values()) == set(SHARED_CLASSES)


def test_2018_misspelling_is_preserved_as_the_key() -> None:
    """2018 ships ``Infilteration``. The lookup must match the file, not the
    corrected spelling, or every Infiltration row drops out unnoticed."""
    assert IDS2018_TO_SHARED["Infilteration"] == "Infiltration"
    assert "Infiltration" not in IDS2018_TO_SHARED


def test_2017_maps_all_but_the_two_without_counterparts() -> None:
    assert set(IDS2017_TO_SHARED.values()) == set(SHARED_CLASSES)
    for name in IDS2017_UNMAPPED:
        assert name not in IDS2017_TO_SHARED


@pytest.mark.parametrize(
    ("raw", "shared"),
    [
        ("DDOS attack-HOIC", "DDoS"),
        ("DDOS attack-LOIC-UDP", "DDoS"),
        ("DDoS attacks-LOIC-HTTP", "DDoS"),
        ("DoS attacks-Hulk", "DoS"),
        ("DoS attacks-SlowHTTPTest", "DoS"),
        ("FTP-BruteForce", "Brute Force"),
        ("SSH-Bruteforce", "Brute Force"),
        ("Brute Force -Web", "Web Attack"),
        ("Brute Force -XSS", "Web Attack"),
        ("SQL Injection", "Web Attack"),
    ],
)
def test_2018_collapses_to_the_right_family(raw: str, shared: str) -> None:
    """``Brute Force -Web`` is a web attack, not a brute-force login.

    The name says Brute Force and the family is Web Attack; anyone reading the
    label alone would file it under the wrong class.
    """
    assert IDS2018_TO_SHARED[raw] == shared


def test_unmapped_2017_classes_become_nan() -> None:
    labels = pd.Series(["BENIGN", "Heartbleed", "PortScan", "Bot"])

    mapped = map_to_shared(labels, "ids2017")

    assert mapped.tolist()[0] == "Benign"
    assert mapped.tolist()[3] == "Bot"
    assert mapped.isna().sum() == 2


def test_unknown_dataset_rejected() -> None:
    with pytest.raises(ValueError, match="ids2017.*ids2018"):
        map_to_shared(pd.Series(["Benign"]), "cicids2019")


def test_counts_reindex_onto_all_seven() -> None:
    """An absent class must read as 0, not disappear from the table."""
    counts = shared_class_counts(pd.Series(["Benign", "Benign", "Bot"]), "ids2018")

    assert list(counts.index) == list(SHARED_CLASSES)
    assert counts["Benign"] == 2
    assert counts["Web Attack"] == 0


@pytest.mark.skipif(
    not CORPUS_2018.exists(), reason="2018 corpus parquet not present"
)
def test_real_2018_corpus_loses_no_rows() -> None:
    """Every label in the real file matches a key -- nothing drops out.

    Reads only the label column: the corpus is 11.4M rows.
    """
    labels = pd.read_parquet(CORPUS_2018, columns=["Label"])["Label"]

    mapped = map_to_shared(labels, "ids2018")

    assert mapped.isna().sum() == 0
    assert shared_class_counts(labels, "ids2018").sum() == len(labels)
