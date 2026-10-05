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

from src.crossdataset import loaders
from src.crossdataset.cleaning import assert_idempotent, clean
from src.crossdataset.sampling import allocate, sample
from src.crossdataset.schema import canonical_features
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

    everything = np.concatenate([part.train, part.test])
    assert sorted(everything.tolist()) == list(range(1000))


@pytest.mark.parametrize("splitter", ["random", "chronological"])
def test_split_hits_the_seventy_thirty_shape(splitter: str) -> None:
    y = pd.Series(["Benign"] * 1000)

    part = (
        random_split(y, seed=42)
        if splitter == "random"
        else chronological_split(y, pd.Series(range(1000)), seed=42)
    )

    assert len(part.train) == pytest.approx(700, abs=2)
    assert len(part.test) == pytest.approx(300, abs=2)
    assert not hasattr(part, "calibration")


def test_a_class_of_two_rows_still_reaches_test() -> None:
    y = pd.Series(["Benign"] * 20 + ["Rare"] * 2)

    part = random_split(y, seed=0)

    assert "Rare" in set(y.iloc[part.train])
    assert "Rare" in set(y.iloc[part.test])


def test_random_split_varies_with_seed() -> None:
    """Unlike the chronological one -- this is the split the protocol's five
    seeds are meant to vary."""
    y = pd.Series(["Benign"] * 200)

    a = random_split(y, seed=1)
    b = random_split(y, seed=2)

    assert a.train.tolist() != b.train.tolist()


# ------------------------------------------------------- loader guards
class TestFinishGuards:
    """The three conditions that make "same conditions" checkable.

    Every one of them protects against a failure that is silent: the loader
    still returns a Corpus, the run still finishes, and the only evidence is a
    number in the gap table that is a little different from the right one.
    """

    # Classes must sit above sampling.KEEP_ALL_BELOW or the sampler takes every
    # one of them whole and no draw can land on a target.
    ROWS = 36_000
    TARGET = 20_000
    # 2017 spells the majority class in capitals; "Benign" is a 2018 label and
    # map_to_shared would drop every row carrying it.
    CLASSES = ("BENIGN", "DoS", "DDoS")

    @classmethod
    def _corpus(cls, n_rows: int = ROWS, *, dupes: int = 0):
        """A synthetic frame shaped like a canonicalised 2017 corpus."""
        cols = list(canonical_features(keep_dst_port=True))
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            rng.integers(1, 1_000_000, size=(n_rows, len(cols))).astype("float64"),
            columns=cols,
        )
        labels = pd.Series([cls.CLASSES[i % len(cls.CLASSES)] for i in range(n_rows)])
        if dupes:
            # Label has to be copied too: clean() dedups on features *and* label.
            frame.iloc[:dupes] = frame.iloc[dupes : 2 * dupes].to_numpy()
            labels.iloc[:dupes] = labels.iloc[dupes : 2 * dupes].to_numpy()
        return frame, labels, pd.Series(range(n_rows)), pd.Series(["mon"] * n_rows)

    def _finish(self, frame, labels, order, group, **kw):
        return loaders._finish(
            "ids2017", frame, labels, order, group, "_row_index",
            keep_dst_port=kw.pop("keep_dst_port", True),
            target=kw.pop("target", self.TARGET),
            seed=42,
        )

    def test_a_pre_sampled_input_is_refused(self, monkeypatch) -> None:
        """The bug this guard exists for.

        ``load_ids2018`` used to read a 300k file the 2018 pipeline had already
        drawn from ~13M rows before cleaning. Cleaning then removed a further
        37,299 rows from that draw -- almost all duplicates -- so 2018 reached
        262,701 rows with a class balance the sampler never designed, while
        2017 reached a full 300,000 by clean-then-sample. Nothing raised.
        """
        monkeypatch.setattr(loaders, "MIN_FULL_CORPUS_ROWS", 100_000)

        with pytest.raises(ValueError, match="whole corpus"):
            self._finish(*self._corpus())

    def test_a_non_finite_row_means_the_rules_drifted(self, monkeypatch) -> None:
        """Dropping non-finite rows does not depend on the label, so a corpus
        built with these rules must yield exactly zero."""
        monkeypatch.setattr(loaders, "MIN_FULL_CORPUS_ROWS", 100)
        frame, labels, order, group = self._corpus()
        frame.loc[0, "Flow Duration"] = np.inf

        with pytest.raises(ValueError, match="non-finite"):
            self._finish(frame, labels, order, group)

    def test_a_few_duplicates_from_the_label_collapse_are_allowed(
        self, monkeypatch, caplog
    ) -> None:
        """Collapsing each source vocabulary onto the shared seven is
        many-to-one, so rows distinct under the source labels can coincide
        under the shared ones. On the real 2018 corpus this is 13 rows: pairs
        of identical flows labelled FTP-BruteForce and SSH-Bruteforce, or two
        of the three web variants. Demanding zero would demand that both
        parquets be labelled at the same granularity, which they are not.
        """
        monkeypatch.setattr(loaders, "MIN_FULL_CORPUS_ROWS", 100)

        with caplog.at_level("INFO"):
            corpus = self._finish(*self._corpus(dupes=5))

        assert corpus.cleaning.rows_dropped_duplicate == 5
        assert len(corpus.X) == self.TARGET
        assert "labels collapsed" in caplog.text

    def test_a_large_duplicate_drop_is_still_refused(self, monkeypatch) -> None:
        """The pre-sampled 2018 file lost 11.9% of its rows to duplicates."""
        monkeypatch.setattr(loaders, "MIN_FULL_CORPUS_ROWS", 100)

        with pytest.raises(ValueError, match="not cleaned with these rules"):
            self._finish(*self._corpus(dupes=self.ROWS // 4))

    def test_dropping_the_port_column_may_legitimately_dedup(self, monkeypatch) -> None:
        """76-column dedup can merge rows that 77 kept apart, so the no-op claim
        is asserted only on the basis the files were cleaned with."""
        monkeypatch.setattr(loaders, "MIN_FULL_CORPUS_ROWS", 100)
        frame, labels, order, group = self._corpus()
        # Identical but for the port: one row on 76 columns, two on 77.
        frame.iloc[0] = frame.iloc[1]
        labels.iloc[0] = labels.iloc[1]
        frame.loc[0, "Destination Port"] = 9999

        corpus = self._finish(frame, labels, order, group, keep_dst_port=False)

        assert corpus.cleaning.rows_dropped_duplicate == 1
        assert len(corpus.X) == self.TARGET

    def test_a_short_draw_is_refused(self, monkeypatch) -> None:
        """A side that lands under target is no longer compared at the
        protocol's size -- which is exactly how 2018 reached 262,701."""
        monkeypatch.setattr(loaders, "MIN_FULL_CORPUS_ROWS", 100)

        with pytest.raises(ValueError, match="same size"):
            self._finish(*self._corpus(), target=self.ROWS * 2)

    def test_a_clean_full_corpus_passes_every_guard(self, monkeypatch) -> None:
        monkeypatch.setattr(loaders, "MIN_FULL_CORPUS_ROWS", 100)

        corpus = self._finish(*self._corpus())

        assert len(corpus.X) == self.TARGET
        assert corpus.cleaning.rows_dropped == 0
        assert corpus.dropped_unmapped == 0
        assert list(corpus.X.columns) == list(canonical_features(keep_dst_port=True))
