"""Is the adversarial-validation AUC of 1.0 real, or an artefact?

An AUC that sits at 1.0000 in five of six classes deserves to be disbelieved
until it survives an attempt to break it. Seven checks, each able to fail:

A. **How is train/test split?** Reported, not assumed.
B. **How many seeds?** One draw proves nothing about stability.
C. **Do duplicate or near-duplicate rows straddle the split?** A test row with
   an identical or almost-identical twin in train is answered from memory. The
   corpora are deduplicated internally, but that says nothing about *near*
   duplicates, and nothing about rows shared *between* the corpora.
D. **Is any column a provenance field?** A column whose ranges do not overlap
   separates the corpora by construction and makes the result circular.
E. **Is anything fitted before the split?** A scaler fitted on all rows leaks
   test statistics into training.
F. **Null control.** Split one corpus into random halves, label them 0 and 1,
   and run the identical procedure. A method that reports AUC near 1.0 there is
   measuring itself, not the data.
G. **Does a simple model get there too?** A gradient-boosted ensemble with 300
   trees and 60 columns can carve a boundary out of very little. A decision
   stump and a logistic regression cannot. If the simple models land near the
   same AUC, the separation is broad and shallow rather than an artefact of
   model capacity.
H. **Label shuffle.** The same real rows the headline used, with the
   2017/2018 labels permuted before the split. F asks whether the procedure
   can invent separation inside one corpus; H holds the actual cross-corpus
   feature distribution fixed and removes only the link between a row and its
   label. Anything above chance is the pipeline reaching the label by a route
   other than the features.

Every check that depends on a draw (C, F, G, H) runs on all five seeds. An
earlier version ran them on seed 42 alone.

Run for every testable class, not just the largest
--------------------------------------------------
An earlier version ran on Benign alone -- the class with the most rows and, as
it happens, the *only* one that does not reach 1.0000. That checked the class
nobody was suspicious of and generalised to the ones they were. The small
classes need it more, not less: Web Attack has 859 rows a side, and the fewer
rows there are the easier it is for a model to memorise all of them, which
produces an AUC of 1.0 with no underlying signal at all.

Run::

    python scripts/diagnose_av.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.crossdataset.labels import SHARED_CLASSES  # noqa: E402
from src.crossdataset.loaders import load_ids2017, load_ids2018  # noqa: E402

logger = logging.getLogger("diagnose_av")

AUDIT = PROJECT_ROOT / "results" / "crossdataset" / "feature_mapping_audit" / "findings.json"
OUT = PROJECT_ROOT / "results" / "crossdataset" / "adversarial_validation" / "diagnostics.json"

SEEDS = (42, 43, 44, 45, 46)
#: Cap per side. Smaller than the real run: this script asks how the procedure
#: behaves, not what the fourth decimal place is.
MAX_PER_SIDE = 8_000
#: Same floor the real run uses, so the same six classes are covered.
MIN_PER_SIDE = 500
TEST_SIZE = 0.3
#: "Near duplicate" means closer than this quantile of the train-to-train
#: nearest-neighbour distances -- a threshold taken from the data rather than
#: guessed, which also makes it its own null expectation.
NEAR_DUP_QUANTILE = 0.001


def classifier(seed: int):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=300,
        num_leaves=63,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        verbose=-1,
        random_state=seed,
    )


def model_ladder(seed: int) -> dict:
    """Increasing capacity, so an AUC can be read against what produced it.

    Each scaler is a step inside a Pipeline, so its statistics are learned from
    the training split and replayed onto test -- fitting one on all rows first
    is exactly the leak check E asks about.
    """
    return {
        "stump": DecisionTreeClassifier(max_depth=1, random_state=seed),
        "tree_depth3": DecisionTreeClassifier(max_depth=3, random_state=seed),
        "logreg": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)
        ),
        # Flow features span eight decades and are heavily right-skewed. A rank
        # transform is what gives a linear model a fair chance on them.
        "logreg_rank": make_pipeline(
            QuantileTransformer(output_distribution="normal", random_state=seed, n_quantiles=200),
            LogisticRegression(max_iter=2000, random_state=seed),
        ),
        "lightgbm": classifier(seed),
    }


def representative_features(columns: list[str]) -> list[str]:
    groups = json.loads(AUDIT.read_text(encoding="utf-8"))["duplicate_groups"]
    drop = {c for g in groups for c in g[1:]}
    return [c for c in columns if c not in drop]


def balanced(a: pd.DataFrame, b: pd.DataFrame, feats: list[str], seed: int):
    n = min(len(a), len(b), MAX_PER_SIDE)
    rng = np.random.default_rng(seed)
    ia = rng.choice(len(a), size=n, replace=False)
    ib = rng.choice(len(b), size=n, replace=False)
    X = pd.concat([a.iloc[ia][feats], b.iloc[ib][feats]], ignore_index=True)
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    return X, y, n


def split(X, y, seed: int):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)


def auc_once(X, y, seed: int) -> float:
    X_tr, X_te, y_tr, y_te = split(X, y, seed)
    m = classifier(seed).fit(X_tr, y_tr)
    return float(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]))


def run_ladder(X, y, seed: int) -> dict[str, float]:
    X_tr, X_te, y_tr, y_te = split(X, y, seed)
    out = {}
    for label, model in model_ladder(seed).items():
        model.fit(X_tr, y_tr)
        out[label] = round(float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])), 4)
    return out


def near_duplicate_share(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> float:
    """Fraction of test rows sitting closer to train than train sits to itself.

    Distances are taken on columns rescaled by the training median and IQR, so
    one wide column such as Flow Duration cannot dominate. The threshold is the
    0.1st percentile of the train-to-train nearest-neighbour distances, which
    makes 0.1% the rate chance alone would produce.
    """
    med = X_tr.median()
    iqr = (X_tr.quantile(0.75) - X_tr.quantile(0.25)).replace(0, np.nan)
    scale = iqr.fillna((X_tr.quantile(0.99) - X_tr.quantile(0.01)).replace(0, 1))
    scale = scale.replace(0, 1)
    Ztr = ((X_tr - med) / scale).to_numpy()
    Zte = ((X_te - med) / scale).to_numpy()
    nn = NearestNeighbors(n_neighbors=2).fit(Ztr)
    d_tr = nn.kneighbors(Ztr)[0][:, 1]
    d_te = nn.kneighbors(Zte, n_neighbors=1)[0][:, 0]
    return float((d_te <= np.quantile(d_tr, NEAR_DUP_QUANTILE)).mean())


def diagnose(name: str, a: pd.DataFrame, b: pd.DataFrame, feats: list[str]) -> dict:
    """Every check, for one shared class."""
    # D -- can any single column do it alone?
    y_all = np.r_[np.zeros(len(a), int), np.ones(len(b), int)]
    best_single, best_feature, disjoint = 0.0, "", []
    for f in feats:
        s = roc_auc_score(y_all, np.r_[a[f].to_numpy(), b[f].to_numpy()])
        # Either direction counts: a column that is reliably *lower* in 2018
        # names the corpus just as well as one that is higher.
        s = max(s, 1 - s)
        if s > best_single:
            best_single, best_feature = s, f
        lo_a, hi_a = a[f].quantile([0.001, 0.999])
        lo_b, hi_b = b[f].quantile([0.001, 0.999])
        if hi_a < lo_b or hi_b < lo_a:
            disjoint.append(f)

    # C -- rows shared between the corpora. Independent of any draw.
    across = len(pd.merge(a[feats].drop_duplicates(), b[feats].drop_duplicates(), how="inner"))

    # Every check that depends on a draw runs on every seed. The first version
    # ran the model ladder, the near-duplicate check and the null control on
    # seed 42 alone and reported that one draw as the answer -- the headline
    # AUC had a spread across five seeds and nothing checking it did.
    rungs = list(model_ladder(SEEDS[0]))
    straddle, near = [], []
    ladder = {k: [] for k in rungs}
    shuffled = {k: [] for k in rungs}
    nulls = {c: {k: [] for k in rungs} for c in ("ids2017", "ids2018")}
    n = 0

    for seed in SEEDS:
        X, y, n = balanced(a, b, feats, seed)
        X_tr, X_te, _, _ = split(X, y, seed)
        straddle.append(len(pd.merge(X_tr.drop_duplicates(), X_te.drop_duplicates(), how="inner")))
        near.append(near_duplicate_share(X_tr, X_te))

        # G -- model ladder. Its lightgbm rung is the same classifier on the
        # same draw and split as the headline run, so the headline AUC falls
        # out of it rather than being fitted a second time.
        for k, v in run_ladder(X, y, seed).items():
            ladder[k].append(v)

        # H -- label shuffle. The same real rows the headline used, with the
        # 2017/2018 labels permuted before the split. The random-halves null
        # below draws one corpus and so asks a different question; this one
        # holds the actual cross-corpus feature distribution fixed and removes
        # only the link between a row and its label. Anything above chance here
        # would be the pipeline reaching the label by some route other than the
        # features.
        y_perm = np.random.default_rng(seed).permutation(y)
        for k, v in run_ladder(X, y_perm, seed).items():
            shuffled[k].append(v)

        # F -- null control: one corpus split into random halves, fresh halves
        # on every seed rather than one fixed split.
        for corpus_name, corpus in (("ids2017", a), ("ids2018", b)):
            idx = np.random.default_rng(seed).permutation(len(corpus))
            half = len(idx) // 2
            Xn, yn, _ = balanced(
                corpus.iloc[idx[:half]].reset_index(drop=True),
                corpus.iloc[idx[half : 2 * half]].reset_index(drop=True),
                feats,
                seed,
            )
            for k, v in run_ladder(Xn, yn, seed).items():
                nulls[corpus_name][k].append(v)

    def mean(v: list[float]) -> float:
        return round(float(np.mean(v)), 4)

    def sd(v: list[float]) -> float:
        return round(float(np.std(v)), 5)

    return {
        "shared_class": name,
        "n_per_side": n,
        "auc_mean": mean(ladder["lightgbm"]),
        "auc_sd": sd(ladder["lightgbm"]),
        "auc_per_seed": [round(v, 4) for v in ladder["lightgbm"]],
        "best_single_feature_auc": round(best_single, 4),
        # Which column, and where its centre sits on each side. Without the
        # name the AUC says "something separates them" and stops; with it the
        # reader can go and look at the column.
        "best_single_feature": best_feature,
        "best_single_feature_median": {
            "ids2017": float(a[best_feature].median()),
            "ids2018": float(b[best_feature].median()),
        },
        "disjoint_columns": disjoint,
        "rows_identical_across_corpora": int(across),
        # The worst seed, not the mean. One row answered from memory is a
        # leak whichever draw it turned up in, and averaging would dilute it.
        "rows_identical_across_split": int(max(straddle)),
        "rows_identical_across_split_per_seed": [int(v) for v in straddle],
        "near_duplicate_test_share": round(float(np.mean(near)), 5),
        "near_duplicate_test_share_max": round(float(np.max(near)), 5),
        "near_duplicate_test_share_per_seed": [round(v, 5) for v in near],
        # Means over every seed. Same keys and shape as the single-seed
        # version, so everything already reading them keeps working.
        "model_ladder": {k: mean(v) for k, v in ladder.items()},
        "model_ladder_sd": {k: sd(v) for k, v in ladder.items()},
        "model_ladder_per_seed": {k: [round(x, 4) for x in v] for k, v in ladder.items()},
        "label_shuffle": {k: mean(v) for k, v in shuffled.items()},
        "label_shuffle_sd": {k: sd(v) for k, v in shuffled.items()},
        "null_ladder": {c: {k: mean(v) for k, v in r.items()} for c, r in nulls.items()},
        "null_ladder_sd": {c: {k: sd(v) for k, v in r.items()} for c, r in nulls.items()},
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)]
    )

    c17, c18 = load_ids2017(), load_ids2018()
    feats = representative_features(list(c17.X.columns))
    pooled = pd.concat([c17.X[feats], c18.X[feats]], ignore_index=True)
    feats = [f for f in feats if pooled[f].nunique(dropna=False) > 1]

    logger.info("")
    logger.info(
        "A. split      random stratified %.0f/%.0f, identical for every class; " "only n differs",
        100 * (1 - TEST_SIZE),
        100 * TEST_SIZE,
    )
    logger.info(
        "E. preprocess LightGBM on raw columns -- nothing fitted before the "
        "split; ladder scalers sit inside a Pipeline"
    )
    logger.info("features: %d | seeds: %s", len(feats), list(SEEDS))

    results = []
    for name in SHARED_CLASSES:
        a = c17.X[(c17.y == name).to_numpy()].reset_index(drop=True)
        b = c18.X[(c18.y == name).to_numpy()].reset_index(drop=True)
        if len(a) < MIN_PER_SIDE or len(b) < MIN_PER_SIDE:
            logger.info("  %-12s skipped: %d / %d rows", name, len(a), len(b))
            continue
        logger.info("  %-12s diagnosing ...", name)
        results.append(diagnose(name, a, b, feats))

    logger.info("")
    logger.info("=" * 96)
    logger.info(
        "%-12s %8s %16s %10s %9s %8s %8s",
        "class",
        "n/side",
        "AUC (5 seeds)",
        "best 1-col",
        "dup>split",
        "near%",
        "disjoint",
    )
    for r in results:
        logger.info(
            "%-12s %8s %9.4f ±%.4f %10.4f %9d %7.2f%% %8d  %s",
            r["shared_class"],
            f"{r['n_per_side']:,}",
            r["auc_mean"],
            r["auc_sd"],
            r["best_single_feature_auc"],
            r["rows_identical_across_split"],
            100 * r["near_duplicate_test_share"],
            len(r["disjoint_columns"]),
            "{} ({:g} -> {:g})".format(
                r["best_single_feature"],
                r["best_single_feature_median"]["ids2017"],
                r["best_single_feature_median"]["ids2018"],
            ),
        )

    keys = list(model_ladder(SEEDS[0]))

    logger.info("")
    logger.info("G. model ladder, mean ± sd over %d seeds", len(SEEDS))
    logger.info("%-12s %s", "class", " ".join(f"{k:>18}" for k in keys))
    for r in results:
        cells = [f'{r["model_ladder"][k]:.4f} ±{r["model_ladder_sd"][k]:.4f}' for k in keys]
        logger.info("%-12s %s", r["shared_class"], " ".join(f"{c:>18}" for c in cells))

    logger.info("")
    logger.info("H. label shuffle -- the same real rows, 2017/2018 labels permuted")
    logger.info("%-12s %s", "class", " ".join(f"{k:>18}" for k in keys))
    for r in results:
        cells = [f'{r["label_shuffle"][k]:.4f} ±{r["label_shuffle_sd"][k]:.4f}' for k in keys]
        logger.info("%-12s %s", r["shared_class"], " ".join(f"{c:>18}" for c in cells))

    logger.info("")
    logger.info("F. null control -- one corpus in random halves (ids2017 / ids2018)")
    logger.info("%-12s %s", "class", " ".join(f"{k:>18}" for k in keys))
    for r in results:
        cells = [
            f'{r["null_ladder"]["ids2017"][k]:.4f} / {r["null_ladder"]["ids2018"][k]:.4f}'
            for k in keys
        ]
        logger.info("%-12s %s", r["shared_class"], " ".join(f"{c:>18}" for c in cells))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "split": f"random stratified {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%}, same for every class",
                "preprocessing": "none (LightGBM on raw features); ladder scalers inside a Pipeline",
                "seeds": list(SEEDS),
                "max_per_side": MAX_PER_SIDE,
                "near_dup_quantile": NEAR_DUP_QUANTILE,
                # Every check that depends on a draw -- the ladder, label
                # shuffle, near-duplicates, straddling rows, the null -- runs
                # once per seed; the single-seed version is superseded.
                "draw_dependent_checks_run_on": "every seed",
                "n_features": len(feats),
                "classes": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("")
    logger.info("wrote %s", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
