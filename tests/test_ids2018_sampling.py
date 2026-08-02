"""Tests for the CSE-CIC-IDS2018 nested stratified sampler.

The nesting guarantee (300k subset of 500k subset of 1M) is the property the
sample-size ladder rests on, and it is not obvious from reading the code, so
it is pinned here. A synthetic corpus index is used -- no CSVs are touched,
so these run in milliseconds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.ids2018.data_loader import CorpusIndex, FileIndex, plan_sample

# Class proportions roughly mirroring the real corpus: one dominant Benign
# class, a mid-sized attack, and two very rare ones.
CLASS_SIZES = {
    "Benign": 80_000,
    "DDOS attack-HOIC": 12_000,
    "Bot": 800,
    "Brute Force -XSS": 60,
    "SQL Injection": 9,
}
LADDER = [1_000, 5_000, 20_000]


@pytest.fixture
def index() -> CorpusIndex:
    """A synthetic corpus index with an unusable row every 1000 positions."""
    classes = np.array(sorted(CLASS_SIZES), dtype=str)
    rng = np.random.default_rng(0)

    codes = np.concatenate(
        [np.full(CLASS_SIZES[name], i, dtype=np.int16) for i, name in enumerate(classes)]
    )
    rng.shuffle(codes)
    # Stand-ins for the embedded header rows found in every real file.
    codes[::1000] = -1

    split = len(codes) // 2
    files = [
        FileIndex(path=Path(f"synthetic-{i}.csv"), n_rows=len(part), codes=part)
        for i, part in enumerate([codes[:split], codes[split:]])
    ]
    offsets = np.array([0, split, len(codes)], dtype=np.int64)
    return CorpusIndex(files=files, classes=classes, offsets=offsets)


def test_nested_samples_are_subsets(index: CorpusIndex) -> None:
    """Each rung of the ladder must contain every row of the rung below it."""
    samples = [set(plan_sample(index, n, mode="nested").tolist()) for n in LADDER]

    for smaller, larger in zip(samples, samples[1:], strict=False):
        assert smaller <= larger


def test_sample_size_is_exact(index: CorpusIndex) -> None:
    for n in LADDER:
        assert len(plan_sample(index, n, mode="nested")) == n


def test_selection_is_deterministic(index: CorpusIndex) -> None:
    """Same seed, same rows -- reruns must not reshuffle the dataset."""
    first = plan_sample(index, 5_000, mode="nested", random_state=42)
    second = plan_sample(index, 5_000, mode="nested", random_state=42)
    assert np.array_equal(first, second)


def test_every_class_is_present_at_every_size(index: CorpusIndex) -> None:
    """Rare attacks must survive even the smallest rung."""
    codes = index.codes
    for n in LADDER:
        present = np.unique(codes[plan_sample(index, n, mode="nested")])
        assert len(present) == len(index.classes)
        # Enough members to allow a later stratified train/test split.
        counts = np.bincount(codes[plan_sample(index, n, mode="nested")])
        assert counts.min() >= 2


def test_unusable_rows_are_never_selected(index: CorpusIndex) -> None:
    codes = index.codes
    assert (codes[plan_sample(index, 5_000, mode="nested")] >= 0).all()


def test_proportions_track_the_corpus(index: CorpusIndex) -> None:
    """Common classes keep their share; only the floored rare ones deviate."""
    codes = index.codes
    valid = codes[codes >= 0]
    selected = plan_sample(index, 20_000, mode="nested")

    corpus_share = np.bincount(valid, minlength=len(index.classes)) / len(valid)
    sample_share = np.bincount(codes[selected], minlength=len(index.classes)) / len(selected)

    for i, name in enumerate(index.classes):
        if corpus_share[i] > 0.001:  # rarer classes are floor-adjusted by design
            assert sample_share[i] == pytest.approx(corpus_share[i], abs=0.005), name


def test_independent_mode_is_not_nested(index: CorpusIndex) -> None:
    """Documents *why* nested mode exists: independent draws do not nest."""
    small = set(plan_sample(index, 1_000, mode="independent").tolist())
    large = set(plan_sample(index, 20_000, mode="independent").tolist())
    assert not small <= large
