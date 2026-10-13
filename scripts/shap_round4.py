"""SHAP round 4: are the Benign pushes in missed attacks on the features that shifted most?

Rules are in results/crossdataset/importance_comparison/SHAP_ROUND4_PREREGISTRATION.md.
Reuses the per-row positive Benign SHAP saved by rounds 2 and 3
(blocks*.joblib); no model is refit. Only the feature set changes: instead of
the fingerprint set F, each class gets S_c = the 13 features where that class
moved most between 2017 and 2018 (1-Wasserstein on pooled ranks, train rows).

Run::

    python scripts/shap_round4.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata, spearmanr, wasserstein_distance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import dataset_vs_attack_importance as dva  # noqa: E402
import shap_importance as r1  # noqa: E402
import shap_round2 as r2  # noqa: E402
from run_crossdataset import partition  # noqa: E402

from src.crossdataset.labels import SHARED_CLASSES  # noqa: E402
from src.crossdataset.loaders import load_ids2017, load_ids2018  # noqa: E402

logger = logging.getLogger("shap_round4")

CMP = r2.CMP
OUT_DIR = CMP / "shap_round4"
PREREG = CMP / "SHAP_ROUND4_PREREGISTRATION.md"

MAIN_K = 13  # size of the round 2 consensus set
EXTRA_K = (5, 26)
SHIFT_ROWS = 20_000
SHIFT_SEED = 2026
BLOCKS = {
    "lightgbm": CMP / "shap_round2" / "blocks.joblib",
    "random_forest": CMP / "shap_round3" / "blocks_random_forest.joblib",
    "xgboost": CMP / "shap_round3" / "blocks_xgboost.joblib",
    "catboost": CMP / "shap_round3" / "blocks_catboost.joblib",
}
ATTACKS = [c for c in SHARED_CLASSES if c != "Benign"]
MAIN_DIRECTION = "ids2017->ids2018"


def class_rows(corpus, cls: str, rng: np.random.Generator) -> pd.DataFrame:
    """Class ``cls`` in the chronological train part, at most SHIFT_ROWS rows."""
    train = partition(corpus, "chronological", 0).train
    idx = train[(corpus.y.iloc[train].astype(str) == cls).to_numpy()]
    if len(idx) > SHIFT_ROWS:
        idx = np.sort(rng.choice(idx, size=SHIFT_ROWS, replace=False))
    return corpus.X.iloc[idx]


def ranked_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """1-Wasserstein and KS between a and b after the pooled empirical CDF."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    u = rankdata(np.concatenate([a, b])) / (len(a) + len(b))
    w1 = wasserstein_distance(u[: len(a)], u[len(a):])
    return float(w1), float(ks_2samp(a, b).statistic)


def shift_table(c17, c18, features: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(SHIFT_SEED)
    rows = []
    for cls in ATTACKS:
        x17, x18 = class_rows(c17, cls, rng), class_rows(c18, cls, rng)
        logger.info("  shift %-12s rows 2017 %d, 2018 %d", cls, len(x17), len(x18))
        for f in features:
            w1, ks = ranked_shift(x17[f].to_numpy(float), x18[f].to_numpy(float))
            rows.append({"class": cls, "feature": f, "w1": w1, "ks": ks,
                         "n_2017": len(x17), "n_2018": len(x18)})
    return pd.DataFrame(rows)


def top_k(shift: pd.DataFrame, cls: str, k: int) -> list[str]:
    t = shift[shift["class"] == cls].dropna(subset=["w1"])
    t = t.sort_values(["w1", "ks", "feature"], ascending=[False, False, True])
    return list(t["feature"].head(k))


def feature_gaps(blocks, direction, cls, features, seeds) -> np.ndarray:
    """d(f) = mean over seeds of share of f in missed - share of f in caught."""
    eye = np.eye(len(features))
    d, _ = r2.class_gaps(blocks, direction, cls, eye, seeds)
    return d


def null_test(blocks, sets_by_class, features, seeds) -> pd.DataFrame:
    """Round 2 null, but the real set depends on the class. Draws restart per k."""
    rows = []
    for k in (MAIN_K, *EXTRA_K):
        rng = np.random.default_rng(r2.DRAW_SEED)
        draws = np.zeros((r2.N_DRAWS, len(features)))
        for i in range(r2.N_DRAWS):
            draws[i, rng.choice(len(features), size=k, replace=False)] = 1.0
        for direction in ("ids2017->ids2018", "ids2018->ids2017"):
            for cls in ATTACKS:
                S = sets_by_class[k][cls]
                real = np.isin(features, S)[None, :].astype(float)
                G, info = r2.class_gaps(blocks, direction, cls, real, seeds)
                G_null, _ = r2.class_gaps(blocks, direction, cls, draws, seeds)
                G = float(G[0])
                eligible = (min(info["n_missed_total"]) >= r1.MIN_MISSED
                            and min(info["n_caught"]) >= r1.MIN_CAUGHT and not np.isnan(G))
                p = (float((1 + np.sum(G_null >= G)) / (1 + r2.N_DRAWS))
                     if not np.isnan(G) else np.nan)
                rows.append({
                    "k": k, "direction": direction, "class": cls, "eligible": eligible,
                    "G": round(G, 4), "null_mean": round(float(np.nanmean(G_null)), 4),
                    "null_95": round(float(np.nanpercentile(G_null, 95)), 4),
                    "p": round(p, 4),
                    "min_missed_total": min(info["n_missed_total"]),
                    "min_caught": min(info["n_caught"]),
                })
    return pd.DataFrame(rows)


def judge(table: pd.DataFrame) -> dict:
    main = table[(table["k"] == MAIN_K) & (table["direction"] == MAIN_DIRECTION)
                 & table["eligible"]]
    n = len(main)
    per_class = {}
    for _, r in main.iterrows():
        if r["G"] > 0 and r["p"] < r2.ALPHA / n:
            per_class[r["class"]] = "supported"
        elif r["p"] >= r2.ALPHA:
            per_class[r["class"]] = "not supported"
        else:
            per_class[r["class"]] = "unclear"
    sup = sum(v == "supported" for v in per_class.values())
    nos = sum(v == "not supported" for v in per_class.values())
    verdict = ("supported" if n and sup > n / 2
               else "not supported" if n and nos > n / 2 else "partial")
    return {"overall": verdict, "per_class": per_class}


def overall(per_model: dict[str, str]) -> str:
    votes = list(per_model.values())
    if sum(v == "supported" for v in votes) >= 3:
        return "class-level shift explains the misses"
    if sum(v == "not supported" for v in votes) >= 3:
        return "not supported"
    return "partial"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--cached-shift", action="store_true",
                    help="reuse shift_table.csv instead of loading the data again")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(args.out_dir / "run.log", encoding="utf-8")],
    )
    prereg_sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    logger.info("pre-registration sha256 %s", prereg_sha)
    started = time.perf_counter()

    seeds = r1.SEEDS
    features = dva.features_from_findings()
    consensus = r2.feature_sets(features)["consensus"]

    shift_csv = args.out_dir / "shift_table.csv"
    if args.cached_shift and shift_csv.exists():
        shift = pd.read_csv(shift_csv)
        logger.info("loaded shift table from %s", shift_csv)
    else:
        r2.lean_parquet_reads()
        c17, c18 = load_ids2017(target=300_000), load_ids2018(target=300_000)
        logger.info("rows: 2017 %d, 2018 %d", len(c17.X), len(c18.X))
        shift = shift_table(c17, c18, features)
        shift.to_csv(shift_csv, index=False)
        del c17, c18

    sets_by_class = {k: {cls: top_k(shift, cls, k) for cls in ATTACKS}
                     for k in (MAIN_K, *EXTRA_K)}
    overlap = {cls: sorted(set(sets_by_class[MAIN_K][cls]) & set(consensus)) for cls in ATTACKS}
    for cls in ATTACKS:
        logger.info("S %-12s overlap with consensus F %d/%d: %s", cls, len(overlap[cls]),
                    MAIN_K, sets_by_class[MAIN_K][cls])

    tables, verdicts, spearman = [], {}, []
    for model_name, path in BLOCKS.items():
        blocks = joblib.load(path)
        table = null_test(blocks, sets_by_class, features, seeds)
        table.insert(0, "model", model_name)
        tables.append(table)
        verdicts[model_name] = judge(table)
        logger.info("%-13s %s %s", model_name, verdicts[model_name]["overall"],
                    verdicts[model_name]["per_class"])
        for direction in ("ids2017->ids2018", "ids2018->ids2017"):
            for cls in ATTACKS:
                d = feature_gaps(blocks, direction, cls, features, seeds)
                w = shift[shift["class"] == cls].set_index("feature")["w1"].reindex(features)
                ok = np.isfinite(d) & np.isfinite(w.to_numpy())
                rho = spearmanr(w.to_numpy()[ok], d[ok]).statistic if ok.sum() > 2 else np.nan
                spearman.append({"model": model_name, "direction": direction, "class": cls,
                                 "spearman_shift_vs_d": round(float(rho), 3)})

    table = pd.concat(tables, ignore_index=True)
    table.to_csv(args.out_dir / "null_test.csv", index=False)
    pd.DataFrame(spearman).to_csv(args.out_dir / "spearman.csv", index=False)

    final = overall({m: v["overall"] for m, v in verdicts.items()})
    logger.info("overall: %s", final)
    summary = {
        "preregistration_sha256": prereg_sha, "seeds": list(seeds), "n_draws": r2.N_DRAWS,
        "draw_seed": r2.DRAW_SEED, "shift_rows": SHIFT_ROWS, "shift_seed": SHIFT_SEED,
        "main_k": MAIN_K, "sets": {str(k): v for k, v in sets_by_class.items()},
        "overlap_with_consensus": overlap, "verdicts": verdicts, "overall": final,
        "minutes": round((time.perf_counter() - started) / 60, 1),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("done in %.1f min -> %s", summary["minutes"], args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
