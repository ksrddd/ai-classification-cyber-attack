"""Turn a cross-dataset run into the tables the write-up needs.

Four questions, in the order they have to be answered:

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

Every figure is a mean over the protocol's five seeds with the spread beside
it. A gap of 0.5 means nothing if the seed-to-seed spread is 0.4.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.crossdataset.labels import SHARED_CLASSES  # noqa: E402

METRIC = "f1_macro_all"


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
    per_class, support = per_class_table(df)

    leak.to_csv(run_dir / "summary_leakage.csv", index=False)
    gaps.to_csv(run_dir / "summary_gap.csv", index=False)
    per_class.to_csv(run_dir / "summary_per_class.csv", index=False)
    support.to_csv(run_dir / "summary_support.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== split inflation: within-dataset, random vs chronological ===")
    print(leak.to_string(index=False))
    print("\n=== transfer gap ===")
    print(gaps.sort_values(["mode", "train", "recovered_pct"],
                           ascending=[True, True, False]).to_string(index=False))
    print("\n=== per-class F1 (transfer directions only) ===")
    tr = per_class[per_class.train != per_class.test]
    print(tr.to_string(index=False))
    print("\n=== test support ===")
    print(support.to_string(index=False))
    print(f"\nwrote 4 summary CSVs to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
