"""Tests for the shared 2017/2018 feature schema.

The mapping is 77 name pairs that mostly differ by abbreviation, which makes it
exactly the kind of table that can be wrong without looking wrong: swap two
entries and both sides still hold plausible float columns, the run completes,
and the score is quietly meaningless. These tests pin the pairs that would
survive casual review -- the twelve that reorder their words, and the one flag
column 2018 forgot to abbreviate.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.crossdataset.schema import (
    CANONICAL_FEATURES,
    DROPPED_IDS2017,
    DST_PORT,
    IDS2018_TO_CANONICAL,
    canonical_features,
    to_canonical,
)
from src.data.schema import EXPECTED_FEATURES

SAMPLE_2018 = Path("data/ids2018/sample_300k.parquet")


def test_both_sides_have_seventy_seven_columns() -> None:
    """The count the advisor's protocol specifies, on both sides."""
    assert len(CANONICAL_FEATURES) == 77
    assert len(IDS2018_TO_CANONICAL) == 77


def test_mapping_is_a_bijection() -> None:
    """No two 2018 columns may collapse onto one canonical name.

    A duplicate target would silently drop a real measurement and duplicate
    another -- the frame would still have 77 columns, so nothing downstream
    would notice.
    """
    targets = list(IDS2018_TO_CANONICAL.values())
    assert len(set(targets)) == len(targets)
    assert set(targets) == set(CANONICAL_FEATURES)


def test_canonical_matches_the_2017_feature_list() -> None:
    """Canonical names are the 2017 names, minus the known duplicate column.

    Ties this module to the project's existing source of truth, so a future
    edit to one and not the other fails here instead of at training time.
    """
    expected = tuple(c for c in EXPECTED_FEATURES if c not in DROPPED_IDS2017)
    assert CANONICAL_FEATURES == expected


@pytest.mark.parametrize(
    ("ids2018_name", "canonical_name"),
    [
        # The twelve that reorder their words. A rule-based renamer that only
        # expanded abbreviations would mis-handle every one of these.
        ("Pkt Len Min", "Min Packet Length"),
        ("Pkt Len Max", "Max Packet Length"),
        ("Pkt Size Avg", "Average Packet Size"),
        ("Fwd Seg Size Avg", "Avg Fwd Segment Size"),
        ("Bwd Seg Size Avg", "Avg Bwd Segment Size"),
        ("Fwd Byts/b Avg", "Fwd Avg Bytes/Bulk"),
        ("Fwd Pkts/b Avg", "Fwd Avg Packets/Bulk"),
        ("Fwd Blk Rate Avg", "Fwd Avg Bulk Rate"),
        ("Bwd Byts/b Avg", "Bwd Avg Bytes/Bulk"),
        ("Init Fwd Win Byts", "Init_Win_bytes_forward"),
        ("Fwd Act Data Pkts", "act_data_pkt_fwd"),
        ("Fwd Seg Size Min", "min_seg_size_forward"),
    ],
)
def test_reordered_names_pair_correctly(
    ids2018_name: str, canonical_name: str
) -> None:
    assert IDS2018_TO_CANONICAL[ids2018_name] == canonical_name


def test_cwe_flag_count_is_not_abbreviated() -> None:
    """2018 shortens every ``Count`` to ``Cnt`` except this one.

    A blanket ``Count -> Cnt`` rule looks correct against fourteen columns and
    breaks on the fifteenth. Pinned so nobody "tidies" the table into a rule.
    """
    assert IDS2018_TO_CANONICAL["CWE Flag Count"] == "CWE Flag Count"
    assert "CWE Flag Cnt" not in IDS2018_TO_CANONICAL


def test_dst_port_can_be_excluded() -> None:
    """The two pipelines disagree about this column; both answers must exist."""
    assert DST_PORT in canonical_features()
    without = canonical_features(keep_dst_port=False)
    assert DST_PORT not in without
    assert len(without) == 76


def test_to_canonical_orders_columns_and_ignores_extras() -> None:
    frame = pd.DataFrame(
        {name: [1.0] for name in IDS2018_TO_CANONICAL},
        columns=list(IDS2018_TO_CANONICAL),
    )
    frame["Label"] = "Benign"
    frame["Timestamp"] = "14/02/2018 08:31:01"

    out = to_canonical(frame, "ids2018")

    assert list(out.columns) == list(CANONICAL_FEATURES)


def test_to_canonical_raises_on_a_missing_column() -> None:
    """Silence here would be worse than failure.

    A reindex would fill the gap with NaN, the preprocessor's median imputer
    would fill that with a plausible number, and the run would report a score
    for a feature that was never present.
    """
    frame = pd.DataFrame({name: [1.0] for name in IDS2018_TO_CANONICAL})
    frame = frame.drop(columns=["Flow Duration"])

    with pytest.raises(KeyError, match="Flow Duration"):
        to_canonical(frame, "ids2018")


def test_to_canonical_rejects_an_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="ids2017.*ids2018"):
        to_canonical(pd.DataFrame(), "ids2019")


@pytest.mark.skipif(
    not SAMPLE_2018.exists(), reason="2018 sample parquet not present"
)
def test_real_2018_sample_converts() -> None:
    """The mapping must hold against the actual file, not just a fixture."""
    frame = pd.read_parquet(SAMPLE_2018)
    out = to_canonical(frame, "ids2018")

    assert list(out.columns) == list(CANONICAL_FEATURES)
    assert len(out) == len(frame)
