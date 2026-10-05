"""Tests for ``scorecard_table`` in ``scripts/analyze_crossdataset.py``.

Recovered percent is a ratio against each model's own ceiling, so on its own it
can rank a weak model above a strong one. The scorecard keeps the macro-F1
mean and spread beside it, and these tests pin the three things a reader relies
on: the spread is the sample standard deviation, a model with nothing random in
it reports exactly zero, and the recovered percent is the one ``gap_table``
publishes rather than a second computation that could drift from it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_crossdataset.py"
SEEDS = [42, 43, 44]


@pytest.fixture(scope="module")
def analysis():
    spec = importlib.util.spec_from_file_location("analyze_crossdataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _results() -> pd.DataFrame:
    """Two models, both directions, three seeds, chronological split only.

    ``noisy`` varies with the seed; ``fixed`` does not, as logistic regression
    does not under a chronological split.
    """
    scores = {
        # (model, train, test): macro-F1 per seed
        ("noisy", "ids2017", "ids2017"): [0.70, 0.74, 0.72],
        ("noisy", "ids2018", "ids2018"): [0.80, 0.82, 0.84],
        ("noisy", "ids2017", "ids2018"): [0.20, 0.30, 0.25],
        ("noisy", "ids2018", "ids2017"): [0.40, 0.50, 0.45],
        ("fixed", "ids2017", "ids2017"): [0.40, 0.40, 0.40],
        ("fixed", "ids2018", "ids2018"): [0.60, 0.60, 0.60],
        ("fixed", "ids2017", "ids2018"): [0.15, 0.15, 0.15],
        ("fixed", "ids2018", "ids2017"): [0.35, 0.35, 0.35],
    }
    rows = []
    for (model, train, test), values in scores.items():
        for seed, value in zip(SEEDS, values, strict=True):
            rows.append(
                {
                    "mode": "chronological", "seed": seed, "train": train, "test": test,
                    "model": model, "f1_macro_all": value,
                    "f1_macro_measurable": value + 0.05, "baseline_f1_macro": 0.13,
                }
            )
    return pd.DataFrame(rows)


def _cell(card: pd.DataFrame, model: str, train: str, test: str) -> pd.Series:
    hit = card[(card["model"] == model) & (card["train"] == train) & (card["test"] == test)]
    assert len(hit) == 1
    return hit.iloc[0]


def test_one_row_per_cell_including_the_ceilings(analysis) -> None:
    card = analysis.scorecard_table(_results())

    assert len(card) == 8  # 2 models x 4 (train, test) pairs
    assert set(card["role"]) == {"ceiling", "transfer"}
    assert (card.loc[card["role"] == "ceiling", "train"]
            == card.loc[card["role"] == "ceiling", "test"]).all()


def test_spread_is_the_sample_standard_deviation(analysis) -> None:
    card = analysis.scorecard_table(_results())

    cell = _cell(card, "noisy", "ids2018", "ids2017")
    assert cell["f1_macro"] == pytest.approx(0.45)
    assert cell["f1_macro_sd"] == pytest.approx(np.std([0.40, 0.50, 0.45], ddof=1), abs=1e-4)
    assert (cell["f1_macro_min"], cell["f1_macro_max"]) == (0.40, 0.50)
    assert cell["n_seeds"] == 3


def test_a_model_with_nothing_random_in_it_reports_zero_spread(analysis) -> None:
    card = analysis.scorecard_table(_results())

    assert _cell(card, "fixed", "ids2018", "ids2017")["f1_macro_sd"] == 0.0


def test_recovered_percent_is_the_one_gap_table_publishes(analysis) -> None:
    results = _results()
    card = analysis.scorecard_table(results)
    gaps = analysis.gap_table(results, analysis.MACRO_F1)

    for _, gap in gaps.iterrows():
        cell = _cell(card, gap["model"], gap["train"], gap["test"])
        assert cell["recovered_pct"] == pytest.approx(gap["recovered_pct"])
        assert cell["f1_macro"] == pytest.approx(gap["transfer"], abs=1e-4)


def test_ceiling_rows_carry_no_recovered_percent(analysis) -> None:
    """It would be 100% by construction, which reads as the best result in the table."""
    card = analysis.scorecard_table(_results())

    assert card.loc[card["role"] == "ceiling", "recovered_pct"].isna().all()


def test_recovered_percent_alone_can_rank_a_weak_model_first(analysis) -> None:
    """The reason the scorecard exists: low ceiling, high ratio, lower score."""
    card = analysis.scorecard_table(_results())
    transfer = card[(card["role"] == "transfer") & (card["test"] == "ids2017")].set_index("model")

    # 'fixed' recovers more of its (low) ceiling but scores lower in absolute terms
    assert transfer.loc["fixed", "recovered_pct"] > transfer.loc["noisy", "recovered_pct"]
    assert transfer.loc["fixed", "f1_macro"] < transfer.loc["noisy", "f1_macro"]


def test_measurable_variant_is_carried_alongside(analysis) -> None:
    card = analysis.scorecard_table(_results())

    cell = _cell(card, "noisy", "ids2017", "ids2018")
    assert cell["f1_macro_measurable"] == pytest.approx(0.30)


def test_gap_table_columns_are_unchanged(analysis) -> None:
    """``src/bundles/registry.py`` reads these by name; the scorecard must not touch them."""
    gaps = analysis.gap_table(_results(), analysis.MACRO_F1)

    assert list(gaps.columns) == [
        "mode", "train", "test", "model", "floor", "ceiling", "transfer",
        "gap", "recovered_pct", "f1_macro_all_sd",
    ]
