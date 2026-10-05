"""Turn a cross-dataset run into the tables the write-up needs.

Five questions, in the order they have to be answered:

1. **How much does the split inflate the within-dataset ceiling?**
   The protocol splits at random, which lets flows from the same attack window
   land on both sides. Comparing random against chronological on the *same*
   corpus gives the size of that inflation -- and since the transfer gap is
   measured against this ceiling, an inflated ceiling inflates the gap.

2. **What does crossing datasets cost?** Ceiling minus transfer, per model, per
   split mode.

3. **How much of the achievable range does transfer recover?** Ceiling and
   floor differ per model and per mode, so the raw gap is hard to compare
   across rows; the fraction of (ceiling - majority baseline) recovered is not.

4. **Which classes survive the crossing?** A macro average over seven classes
   hides that one of them may be carrying all of it.

5. **What are the scores themselves?** Recovered percent is a ratio against each
   model's own ceiling, so it ranks a model with a low ceiling above one that
   scores higher in absolute terms. ``scorecard_table`` keeps the macro-F1 --
   mean, standard deviation and range over the seeds -- for the transfer cells
   and for the ceiling cells, with recovered percent beside it.

Every figure is a mean over the protocol's five seeds with the spread beside
it. A gap of 0.5 means nothing if the seed-to-seed spread is 0.4.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.crossdataset.labels import SHARED_CLASSES  # noqa: E402

METRIC = "f1_macro_all"
#: ``scorecard_table`` is always macro-F1, so it does not follow ``--metric``.
MACRO_F1 = "f1_macro_all"


def _agg(df: pd.DataFrame, keys: list[str], metric: str) -> pd.DataFrame:
    out = df.groupby(keys)[metric].agg(["mean", "std", "count"]).reset_index()
    return out.rename(columns={"mean": metric, "std": f"{metric}_sd"})


def leakage_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Within-dataset score under each split mode, and the difference."""
    within = df[~df.transfer]
    t = _agg(within, ["test", "model", "mode"], metric)
    wide = t.pivot(index=["test", "model"], columns="mode", values=metric)
    if {"random", "chronological"} <= set(wide.columns):
        wide["inflation"] = (wide["random"] - wide["chronological"]).round(4)
    return wide.round(4).reset_index()


def gap_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Ceiling, transfer, gap and recovered fraction for each direction."""
    agg = _agg(df, ["mode", "train", "test", "model"], metric)
    base = _agg(df, ["mode", "train", "test", "model"], "baseline_f1_macro")

    ceiling = (
        agg[agg.train == agg.test]
        .rename(columns={metric: "ceiling"})[["mode", "test", "model", "ceiling"]]
    )
    transfer = agg[agg.train != agg.test].rename(columns={metric: "transfer"})
    floor = base.rename(columns={"baseline_f1_macro": "floor"})[
        ["mode", "train", "test", "model", "floor"]
    ]

    out = transfer.merge(ceiling, on=["mode", "test", "model"], how="left")
    out = out.merge(floor, on=["mode", "train", "test", "model"], how="left")
    out["gap"] = (out["transfer"] - out["ceiling"]).round(4)
    out["recovered_pct"] = (
        (out["transfer"] - out["floor"]) / (out["ceiling"] - out["floor"]) * 100
    ).round(1)
    cols = ["mode", "train", "test", "model", "floor", "ceiling", "transfer",
            "gap", "recovered_pct", f"{metric}_sd"]
    return out[cols].round(4)


def scorecard_table(df: pd.DataFrame) -> pd.DataFrame:
    """Macro-F1 mean, spread and range for every cell of the 2x2 transfer matrix.

    ``gap_table`` reports recovered percent, a ratio against each model's own
    ceiling and floor: two models with very different ceilings can post the same
    percentage on very different scores, and a model with a weak ceiling can top
    the ranking while scoring below a stronger one. This table keeps the score
    itself, for the ceiling cells (``train == test``) as well as the transfer
    cells, with recovered percent beside it. ``recovered_pct`` is read from
    ``gap_table`` rather than recomputed, so the two files cannot disagree.

    Always macro-F1, whatever ``--metric`` the other tables were built with: the
    column names say so. ``f1_macro`` averages every class present in the test
    split, ``f1_macro_measurable`` only those with at least ``MEASURABLE_MIN``
    (``scripts/run_crossdataset.py``) test rows -- the project reports both,
    since a class with ten test rows weighs as much as one with ten thousand.

    ``f1_macro_sd`` is the sample standard deviation over seeds (``ddof=1``, as
    in ``gap_table``). It is only as wide as what the seed moves. The corpora
    are drawn once, so under the chronological split the partition is fixed and
    the seed reaches the model's initialisation alone; a model with no random
    component (logistic regression) therefore reports exactly 0, which means
    "nothing varied", not "stable". Under the random split the seed also moves
    the partition. Neither includes the variance of the 300,000-row draw.
    """
    keys = ["mode", "train", "test", "model"]
    card = (
        df.groupby(keys)
        .agg(
            n_seeds=("seed", "nunique"),
            f1_macro=(MACRO_F1, "mean"),
            f1_macro_sd=(MACRO_F1, "std"),
            f1_macro_min=(MACRO_F1, "min"),
            f1_macro_max=(MACRO_F1, "max"),
            f1_macro_measurable=("f1_macro_measurable", "mean"),
            f1_macro_measurable_sd=("f1_macro_measurable", "std"),
            baseline_f1_macro=("baseline_f1_macro", "mean"),
        )
        .reset_index()
    )
    card.insert(4, "role", np.where(card["train"] == card["test"], "ceiling", "transfer"))

    recovered = gap_table(df, MACRO_F1)[keys + ["recovered_pct"]]
    card = card.merge(recovered, on=keys, how="left")
    return card.round(4)


def per_class_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean per-class F1 for every direction, plus the test support."""
    f1_cols = [f"f1__{c}" for c in SHARED_CLASSES if f"f1__{c}" in df.columns]
    n_cols = [f"n__{c}" for c in SHARED_CLASSES if f"n__{c}" in df.columns]
    grouped = df.groupby(["mode", "train", "test", "model"])[f1_cols].mean()
    support = df.groupby(["mode", "test"])[n_cols].first()
    return grouped.round(4).reset_index(), support.reset_index()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", default="protocol_v1")
    ap.add_argument("--metric", default=METRIC,
                    choices=["f1_macro_all", "f1_macro_measurable", "accuracy"])
    args = ap.parse_args(argv)

    run_dir = PROJECT_ROOT / "results" / "crossdataset" / args.run_name
    df = pd.read_csv(run_dir / "results.csv")
    print(f"{len(df)} result rows | seeds {sorted(df.seed.unique())} | "
          f"modes {sorted(df['mode'].unique())} | models {df.model.nunique()}")

    leak = leakage_table(df, args.metric)
    gaps = gap_table(df, args.metric)
    card = scorecard_table(df)
    per_class, support = per_class_table(df)

    leak.to_csv(run_dir / "summary_leakage.csv", index=False)
    gaps.to_csv(run_dir / "summary_gap.csv", index=False)
    card.to_csv(run_dir / "summary_macro_f1.csv", index=False)
    per_class.to_csv(run_dir / "summary_per_class.csv", index=False)
    support.to_csv(run_dir / "summary_support.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== split inflation: within-dataset, random vs chronological ===")
    print(leak.to_string(index=False))
    print("\n=== transfer gap ===")
    print(gaps.sort_values(["mode", "train", "recovered_pct"],
                           ascending=[True, True, False]).to_string(index=False))
    print("\n=== macro-F1 mean, sd and recovered percent, per cell ===")
    print(card.sort_values(["mode", "train", "test", "f1_macro"],
                           ascending=[True, True, True, False]).to_string(index=False))
    print("\n=== per-class F1 (transfer directions only) ===")
    tr = per_class[per_class.train != per_class.test]
    print(tr.to_string(index=False))
    print("\n=== test support ===")
    print(support.to_string(index=False))
    print(f"\nwrote 5 summary CSVs to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
