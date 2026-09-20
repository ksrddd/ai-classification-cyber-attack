"""Did dropping the dataset-identifying features improve transfer?

Applies the rule recorded in ``ablation/PREREGISTRATION.md`` before any of these
runs existed. It is written out again here so the verdict and the rule it was
judged by sit in the same place:

    Ablation helps only if, at some K, the mean recovered_pct across both models
    and both transfer directions rises by at least 5 percentage points against
    K=0, and that rise exceeds twice the seed-to-seed standard deviation.

``recovered_pct`` rather than raw transfer F1, because transfer can rise simply
by the within-dataset ceiling falling toward it -- a model made worse at
everything looks like a model that transfers better. The ceiling is reported
alongside so that failure mode is visible rather than inferred.

Run::

    python scripts/analyze_ablation.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("analyze_ablation")

RESULTS = PROJECT_ROOT / "results" / "crossdataset"
BASELINE = "protocol_v2"
METRIC = "f1_macro_all"

#: The pre-registered thresholds. Both must be cleared.
MIN_GAIN_PCT = 5.0
MIN_SD_MULTIPLE = 2.0


def gap_table(df: pd.DataFrame) -> pd.DataFrame:
    """Ceiling, transfer and recovered fraction -- same definitions as
    ``analyze_crossdataset.gap_table``, so the numbers are comparable."""
    agg = (
        df.groupby(["mode", "train", "test", "model"])[METRIC]
        .agg(["mean", "std"]).reset_index()
        .rename(columns={"mean": METRIC, "std": "sd"})
    )
    base = (
        df.groupby(["mode", "train", "test", "model"])["baseline_f1_macro"]
        .mean().reset_index().rename(columns={"baseline_f1_macro": "floor"})
    )

    ceiling = (
        agg[agg.train == agg.test]
        .rename(columns={METRIC: "ceiling"})[["mode", "test", "model", "ceiling"]]
    )
    transfer = agg[agg.train != agg.test].rename(columns={METRIC: "transfer"})

    out = transfer.merge(ceiling, on=["mode", "test", "model"], how="left")
    out = out.merge(base, on=["mode", "train", "test", "model"], how="left")
    out["gap"] = out["transfer"] - out["ceiling"]
    out["recovered_pct"] = (
        (out["transfer"] - out["floor"]) / (out["ceiling"] - out["floor"]) * 100
    )
    return out


def per_seed_recovered(df: pd.DataFrame) -> pd.Series:
    """recovered_pct computed per seed, so its spread can be measured.

    The headline figure averages over seeds; the decision rule needs to know how
    much that figure wanders on its own, or a five-point rise means nothing.
    """
    out = []
    for _seed, chunk in df.groupby("seed"):
        table = gap_table(chunk)
        out.append(table["recovered_pct"].mean())
    return pd.Series(out, index=sorted(df["seed"].unique()))


def load(run: str) -> pd.DataFrame | None:
    path = RESULTS / run / "results.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", nargs="*", type=int, default=[5, 10, 20])
    ap.add_argument(
        "--models", nargs="*", default=None,
        help="restrict to these models; default is every model present in all "
             "runs being compared",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    runs: dict[int, pd.DataFrame] = {}
    baseline = load(BASELINE)
    if baseline is None:
        logger.error("baseline %s not found", BASELINE)
        return 2
    runs[0] = baseline
    for k in args.k:
        df = load(f"ablation_k{k}")
        if df is None:
            logger.warning("ablation_k%d not found -- skipping", k)
            continue
        runs[k] = df

    # Every run must be summarised over the same models, or K=0 averaged over
    # seven would be compared against K=20 averaged over two and the difference
    # would be the model set rather than the ablation.
    common = set.intersection(*(set(df.model.unique()) for df in runs.values()))
    if args.models:
        missing = set(args.models) - common
        if missing:
            logger.error("not present in every run: %s", ", ".join(sorted(missing)))
            return 2
        common = set(args.models)
    models = sorted(common)
    for k in runs:
        runs[k] = runs[k][runs[k].model.isin(models)]
    logger.info("comparing %d model(s) across K = %s:\n  %s",
                len(models), sorted(runs), ", ".join(models))

    rows, spread = [], {}
    for k, df in sorted(runs.items()):
        table = gap_table(df)
        spread[k] = per_seed_recovered(df)
        for mode, chunk in table.groupby("mode"):
            rows.append({
                "k": k,
                "mode": mode,
                "ceiling": chunk["ceiling"].mean(),
                "transfer": chunk["transfer"].mean(),
                "gap": chunk["gap"].mean(),
                "recovered_pct": chunk["recovered_pct"].mean(),
            })
    summary = pd.DataFrame(rows).round(4)

    logger.info("\n=== mean over both models and both transfer directions ===")
    logger.info(summary.to_string(index=False))

    logger.info("\n=== per-seed recovered_pct (all cells averaged) ===")
    for k, series in sorted(spread.items()):
        logger.info("  k=%-3d %s   mean=%.2f  sd=%.2f",
                    k, " ".join(f"{v:6.2f}" for v in series),
                    series.mean(), series.std())

    logger.info("\n=== recovered_pct per model, averaged over modes and directions ===")
    per_model = []
    for k, df in sorted(runs.items()):
        t = gap_table(df)
        for model, chunk in t.groupby("model"):
            per_model.append({"model": model, "k": k,
                              "recovered_pct": chunk["recovered_pct"].mean()})
    pm = (pd.DataFrame(per_model)
          .pivot(index="model", columns="k", values="recovered_pct").round(1))
    pm["change"] = (pm[max(runs)] - pm[0]).round(1)
    logger.info(pm.to_string())
    pm.to_csv(RESULTS / "ablation" / "per_model.csv")

    logger.info("\n=== pre-registered decision rule ===")
    base_mean = spread[0].mean()
    base_sd = spread[0].std()
    logger.info("  K=0 baseline recovered_pct %.2f (seed sd %.2f)", base_mean, base_sd)
    logger.info("  helps only if gain >= %.1f points AND gain > %.0f x sd (%.2f)",
                MIN_GAIN_PCT, MIN_SD_MULTIPLE, MIN_SD_MULTIPLE * base_sd)

    verdict_helps = False
    for k in sorted(k for k in runs if k):
        gain = spread[k].mean() - base_mean
        clears_gain = gain >= MIN_GAIN_PCT
        clears_sd = gain > MIN_SD_MULTIPLE * base_sd
        ceiling_now = summary[summary.k == k]["ceiling"].mean()
        ceiling_0 = summary[summary.k == 0]["ceiling"].mean()
        logger.info(
            "  k=%-3d gain %+6.2f pts  | >=5pts %-5s | >2sd %-5s | ceiling %.4f -> %.4f",
            k, gain, clears_gain, clears_sd, ceiling_0, ceiling_now,
        )
        verdict_helps |= clears_gain and clears_sd

    logger.info("\n" + "=" * 68)
    if verdict_helps:
        logger.info("VERDICT: ablation helps at at least one K by the pre-registered rule")
    else:
        logger.info(
            "VERDICT: ablation does not help. No K clears the rule recorded before "
            "the runs.\nThis is what adversarial validation predicted: the corpora "
            "stay separable at AUC >= 0.985\nwith the top 40 of 60 features removed, "
            "so the fingerprint is carried redundantly by\nnearly the whole feature "
            "space and no small removal can take it away."
        )

    out = RESULTS / "ablation" / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
