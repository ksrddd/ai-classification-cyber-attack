"""Tests for the shared cleaning, sampling and splitting rules.

These three steps decide which rows each dataset brings to a cross-dataset
comparison. Every failure mode they have is silent: a cleaning rule that fires
on one corpus and not the other, a sampler that quietly drops the class with 36
rows, a chronological split that leaks the back half of an attack window into
training. None of them raise; they just move the number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.crossdataset.cleaning import assert_idempotent, clean
from src.crossdataset.sampling import allocate, sample
from src.crossdataset.splits import chronological_split, random_split

FEATURES = ["a", "b"]


def _frame(rows: list[tuple[float, float, str]], meta: list[int] | None = None):
    df = pd.DataFrame(rows, columns=[*FEATURES, "Label"])
    df["_row_index"] = meta if meta is not None else range(len(df))
    return df


# ----------------------------------------------------------------- cleaning
def test_infinite_rows_are_dropped_not_imputed() -> None:
    """Imputing would teach the model that an undefined byte rate is typical."""
    df = _frame([(1.0, 2.0, "Benign"), (np.inf, 2.0, "Bot"), (3.0, 4.0, "DoS")])

    out, report = clean(df, FEATURES)

    assert len(out) == 2
    assert report.rows_dropped_non_finite == 1
    assert "Bot" not in set(out["Label"])


def test_duplicates_compare_features_and_label_only() -> None:
    """Metadata must not keep two identical measurements apart.

    The 2017 corpus was deduplicated this way. If 2018 kept both copies because
    their ``_row_index`` differed, it would carry duplicate evidence 2017 does
    not -- a difference between the datasets that nobody chose.
    """
    df = _frame(
        [(1.0, 2.0, "Benign"), (1.0, 2.0, "Benign"), (1.0, 2.0, "Bot")],
        meta=[10, 999, 11],
    )

    out, report = clean(df, FEATURES)

    assert report.rows_dropped_duplicate == 1
    assert len(out) == 2


def test_metadata_columns_survive_cleaning() -> None:
    """``_row_index`` is the only chronological signal the 2017 export carries."""
    df = _frame([(1.0, 2.0, "Benign"), (np.inf, 1.0, "Bot")])

    out, _ = clean(df, FEATURES)

    assert "_row_index" in out.columns
    assert out["_row_index"].tolist() == [0]


def test_idempotence_check_passes_on_clean_data_and_fails_otherwise() -> None:
    clean_df = _frame([(1.0, 2.0, "Benign"), (3.0, 4.0, "Bot")])
    assert_idempotent(clean_df, FEATURES)

    dirty = _frame([(1.0, 2.0, "Benign"), (np.nan, 4.0, "Bot")])
    with pytest.raises(AssertionError, match="non-finite"):
        assert_idempotent(dirty, FEATURES)


# ----------------------------------------------------------------- sampling
def test_rare_classes_are_taken_whole() -> None:
    """Infiltration has 36 rows in 2017. Proportional allocation would give it 4."""
    counts = pd.Series({"Benign": 2_000_000, "DoS": 190_000, "Infiltration": 36})

    quota = allocate(counts, target=300_000, keep_all_below=10_000)

    assert quota["Infiltration"] == 36


def test_allocation_hits_the_target_exactly() -> None:
    counts = pd.Series({"Benign": 2_000_000, "DoS": 190_000, "DDoS": 128_000,
                        "Bot": 1_948, "Infiltration": 36})

    quota = allocate(counts, target=300_000, keep_all_below=10_000)

    assert quota.sum() == 300_000
    assert (quota <= counts).all()


def test_allocation_preserves_the_class_imbalance() -> None:
    """A max-min fair split would take 2017 from 86% Benign to 32%, changing the
    majority baseline and with it the meaning of every gap measured against it."""
    counts = pd.Series({"Benign": 2_072_254, "DoS": 193_730, "DDoS": 128_014,
                        "Brute Force": 9_150, "Bot": 1_948, "Infiltration": 36})

    quota = allocate(counts, target=300_000, keep_all_below=10_000)

    assert 0.80 < quota["Benign"] / quota.sum() < 0.88


def test_sample_preserves_row_order() -> None:
    """Row order is the chronological signal; a shuffle here breaks the split.

    ``keep_all_below`` is lowered so the class is actually sampled down --
    at the default it would fall under the keep-whole threshold and the test
    would pass without exercising the draw at all.
    """
    df = pd.DataFrame({"Label": ["Benign"] * 100, "_row_index": range(100)})

    out = sample(df, "Label", target=20, seed=1, keep_all_below=10)

    assert len(out) == 20
    assert out["_row_index"].is_monotonic_increasing


def test_sample_is_reproducible_from_its_seed() -> None:
    df = pd.DataFrame({"Label": ["Benign"] * 100, "_row_index": range(100)})

    a = sample(df, "Label", target=20, seed=7, keep_all_below=10)
    b = sample(df, "Label", target=20, seed=7, keep_all_below=10)
    c = sample(df, "Label", target=20, seed=8, keep_all_below=10)

    assert a["_row_index"].tolist() == b["_row_index"].tolist()
    assert a["_row_index"].tolist() != c["_row_index"].tolist()


# ------------------------------------------------------------------- splits
def test_chronological_split_never_puts_a_later_row_in_train() -> None:
    """The property the whole split exists for."""
    y = pd.Series(["Bot"] * 100)
    order = pd.Series(range(100))

    part = chronological_split(y, order, seed=42)

    assert order.iloc[part.train].max() < order.iloc[part.test].min()


def test_chronological_split_is_independent_of_seed() -> None:
    """The 2017 handoff states the partition does not consume the seed.

    Keeping that true means seed-to-seed variation in results is model variance
    and nothing else.
    """
    y = pd.Series(["Bot"] * 50 + ["DoS"] * 50)
    order = pd.Series(range(100))

    a = chronological_split(y, order, seed=1)
    b = chronological_split(y, order, seed=999)

    assert a.train.tolist() == b.train.tolist()
    assert a.test.tolist() == b.test.tolist()


def test_split_happens_inside_each_group_and_class() -> None:
    """Each attack class lives in one capture file, so a corpus-wide time cutoff
    would hand whole classes to one side of the split."""
    y = pd.Series(["Bot"] * 10 + ["DoS"] * 10)
    order = pd.Series(list(range(10)) + list(range(100, 110)))
    group = pd.Series(["fri"] * 10 + ["wed"] * 10)

    part = chronological_split(y, order, group, seed=42)

    assert set(y.iloc[part.train]) == {"Bot", "DoS"}
    assert set(y.iloc[part.test]) == {"Bot", "DoS"}


def test_a_class_of_one_row_goes_entirely_to_train() -> None:
    """A test row whose class has no training rows scores the protocol, not the
    model."""
    y = pd.Series(["Benign"] * 20 + ["Heartbleed"])
    order = pd.Series(range(21))

    part = chronological_split(y, order, seed=42)

    assert "Heartbleed" in set(y.iloc[part.train])
    assert "Heartbleed" not in set(y.iloc[part.test])


@pytest.mark.parametrize("splitter", ["random", "chronological"])
def test_partitions_are_disjoint_and_complete(splitter: str) -> None:
    y = pd.Series(["Benign"] * 600 + ["Bot"] * 300 + ["DoS"] * 100)
    order = pd.Series(range(1000))

    part = (
        random_split(y, seed=3)
        if splitter == "random"
        else chronological_split(y, order, seed=3)
    )

    everything = np.concatenate([part.train, part.calibration, part.test])
    assert sorted(everything.tolist()) == list(range(1000))


def test_random_split_hits_the_sixty_ten_thirty_shape() -> None:
    y = pd.Series(["Benign"] * 1000)

    part = random_split(y, seed=42)

    assert len(part.train) == pytest.approx(600, abs=2)
    assert len(part.calibration) == pytest.approx(100, abs=2)
    assert len(part.test) == pytest.approx(300, abs=2)


def test_random_split_varies_with_seed() -> None:
    """Unlike the chronological one -- this is the split the protocol's five
    seeds are meant to vary."""
    y = pd.Series(["Benign"] * 200)

    a = random_split(y, seed=1)
    b = random_split(y, seed=2)

    assert a.train.tolist() != b.train.tolist()
