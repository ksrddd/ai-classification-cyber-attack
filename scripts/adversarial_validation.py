"""Which features say which dataset a flow came from?

A model trained on one corpus and scored on the other recovers little of its
own ceiling. That gap has two possible explanations and they call for opposite
responses: either the attack behaviour genuinely differs between 2017 and 2018,
or the corpora carry a *fingerprint* -- a capture artefact the model latches
onto, which cannot generalise because it describes the lab rather than the
attack. Adversarial validation separates them. Train a classifier whose only
job is to name the source dataset: whatever it finds useful is fingerprint.

Four decisions that make the answer mean something
--------------------------------------------------
**One class at a time.** Pooling every class lets the classifier separate the
corpora on class *composition* -- 2017 is 83% Benign and 2018 88% -- which says
nothing about whether the features mean the same thing. Run within a class and
that route is closed.

**Balanced.** Equal rows per side, or the classifier wins by predicting the
larger corpus and the AUC describes the sampling rather than the data.

**Duplicate columns collapsed.** The mapping audit found ten groups of columns
that are the same measurement under more than one name. Permutation importance
splits credit between duplicates -- shuffle one and the model reads the other,
so both look unimportant -- which would hide exactly the features this script
exists to find. One representative per group is kept.

**Several seeds.** The first version of this script ran one. An AUC of 1.0000
from a single draw is exactly the result that should not be believed on one
sample, and reporting a spread costs almost nothing.

On reading the result
---------------------
``scripts/diagnose_av.py`` establishes that the AUC here is not an artefact.
No duplicate row straddles the split in any class, nothing is fitted before
it, and the whole procedure returns 0.46-0.53 when one corpus is split into
random halves instead.

What it also establishes is that the separation does not have the same shape
in every class, and an early version of this docstring got that wrong by
running the diagnostics on Benign alone. In Benign it is broad and shallow:
the best single column reaches 0.766, a decision stump 0.717, and it takes the
whole feature space to get to 0.999. In the five attack classes a single
threshold on a single column already reaches 0.94-1.00 -- and in Web Attack
``min_seg_size_forward`` does not even overlap between corpora. Benign drifted;
the attack classes were staged differently. The ablation in
``scripts/run_crossdataset.py`` is still the right next question, because the
signal remains redundant enough that removing the top 40 of 60 columns leaves
the AUC at 1.0.

Run::

    python scripts/adversarial_validation.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.crossdataset.labels import SHARED_CLASSES  # noqa: E402
from src.crossdataset.loaders import load_ids2017, load_ids2018  # noqa: E402

logger = logging.getLogger("adversarial_validation")

AUDIT = PROJECT_ROOT / "results" / "crossdataset" / "feature_mapping_audit" / "findings.json"
OUT_DIR = PROJECT_ROOT / "results" / "crossdataset" / "adversarial_validation"

#: A class needs this many rows on *both* sides before a balanced draw can say
#: anything. 2017 holds 36 Infiltration flows, so that class is not testable.
MIN_PER_SIDE = 500
#: Cap per side per class: the signal saturates long before this and the
#: permutation pass is the expensive step.
MAX_PER_SIDE = 20_000
#: Repeats per feature in the permutation pass. Five is enough to separate the
#: leaders from the noise floor without making the run quadratic in patience.
N_REPEATS = 5
#: Feature counts to re-fit without, to show whether the signal is carried by a
#: few columns or spread across many. Directly sizes the ablation, so the range
#: runs far enough to find where separability actually breaks -- an earlier
#: version stopped at 10, where the AUC had not moved, which measured nothing.
ABLATION_K = (1, 3, 5, 10, 20, 30, 40, 50, 55)
#: The protocol's seeds. Each varies the draw, the split and the model together.
SEEDS = (42, 43, 44, 45, 46)

TEST_SIZE = 0.3


def representative_features(features: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Keep one column per duplicate group; report what each one stands for."""
    if not AUDIT.is_file():
        raise FileNotFoundError(
            f"{AUDIT} not found -- run scripts/verify_feature_mapping.py first. "
            "Without the duplicate groups this script would split permutation "
            "credit between identical columns and under-rank both."
        )
    groups = json.loads(AUDIT.read_text(encoding="utf-8"))["duplicate_groups"]

    dropped: dict[str, list[str]] = {}
    remove: set[str] = set()
    for group in groups:
        keep, *rest = [c for c in features if c in group]
        dropped[keep] = rest
        remove.update(rest)

    kept = [f for f in features if f not in remove]
    logger.info(
        "features: %d columns -> %d after collapsing %d duplicate group(s)",
        len(features), len(kept), len(groups),
    )
    return kept, dropped


def balanced_frame(
    a: pd.DataFrame, b: pd.DataFrame, features: list[str], seed: int
) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Equal rows from each corpus, labelled by origin."""
    n = min(len(a), len(b), MAX_PER_SIDE)
    rng = np.random.default_rng(seed)
    pick_a = rng.choice(len(a), size=n, replace=False)
    pick_b = rng.choice(len(b), size=n, replace=False)

    X = pd.concat(
        [a.iloc[pick_a][features], b.iloc[pick_b][features]], ignore_index=True
    )
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    return X, y, n


def build_classifier(seed: int):
    """LightGBM: no scaling, so the raw scale differences stay visible.

    Standardising first would fold part of the very signal being measured into
    the preprocessing -- a column whose median differs by four decades looks
    much less distinctive once both sides are centred and divided by their own
    spread. Nothing is fitted before the split, so there is no route for test
    statistics to reach training.
    """
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=300, num_leaves=63, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        n_jobs=-1, verbose=-1, random_state=seed,
    )


def one_seed(X: pd.DataFrame, y: np.ndarray, features: list[str], seed: int):
    """AUC, per-feature importance and the ablation curve for a single draw."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    model = build_classifier(seed).fit(X_tr, y_tr)
    auc = float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))

    imp = permutation_importance(
        model, X_te, y_te, scoring="roc_auc",
        n_repeats=N_REPEATS, random_state=seed, n_jobs=1,
    )
    order = pd.Series(imp.importances_mean, index=features).sort_values(ascending=False)

    ablation = {}
    for k in ABLATION_K:
        if k >= len(features):
            continue
        remaining = [f for f in features if f not in set(order.head(k).index)]
        m = build_classifier(seed).fit(X_tr[remaining], y_tr)
        ablation[k] = float(roc_auc_score(y_te, m.predict_proba(X_te[remaining])[:, 1]))

    return auc, imp.importances_mean, ablation


def run_class(name: str, a: pd.DataFrame, b: pd.DataFrame,
              features: list[str], seeds: tuple[int, ...]) -> dict | None:
    """Adversarial validation within one shared class, over every seed."""
    if len(a) < MIN_PER_SIDE or len(b) < MIN_PER_SIDE:
        logger.info("  %-12s skipped: %d / %d rows, below %d on one side",
                    name, len(a), len(b), MIN_PER_SIDE)
        return None

    aucs, importances, curves, n = [], [], [], 0
    for seed in seeds:
        X, y, n = balanced_frame(a, b, features, seed)
        auc, imp, ablation = one_seed(X, y, features, seed)
        aucs.append(auc)
        importances.append(imp)
        curves.append(ablation)

    imp_matrix = np.vstack(importances)
    ranked = (
        pd.DataFrame({
            "feature": features,
            "importance": imp_matrix.mean(axis=0),
            "sd": imp_matrix.std(axis=0),
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    curve = {k: float(np.mean([c[k] for c in curves])) for k in curves[0]}

    logger.info(
        "  %-12s n=%s/side  AUC=%.4f ±%.4f  | top: %s",
        name, f"{n:,}", float(np.mean(aucs)), float(np.std(aucs)),
        ", ".join(ranked["feature"].head(3)),
    )
    logger.info("  %-12s AUC after dropping top-k: %s", "",
                {k: round(v, 4) for k, v in curve.items()})

    return {
        "shared_class": name,
        "n_per_side": n,
        "n_seeds": len(seeds),
        "auc": round(float(np.mean(aucs)), 4),
        "auc_sd": round(float(np.std(aucs)), 5),
        "auc_per_seed": [round(a, 4) for a in aucs],
        "auc_after_dropping_top_k": {str(k): round(v, 4) for k, v in curve.items()},
        "importance": ranked.to_dict("records"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])

    logger.info("loading both corpora under the protocol_v2 pipeline")
    c17, c18 = load_ids2017(), load_ids2018()

    features, stands_for = representative_features(list(c17.X.columns))
    # A column constant across both corpora carries no signal in either
    # direction; the preprocessor drops these in the real runs too.
    pooled = pd.concat([c17.X[features], c18.X[features]], ignore_index=True)
    varying = [f for f in features if pooled[f].nunique(dropna=False) > 1]
    if len(varying) < len(features):
        logger.info("dropping %d column(s) constant across both corpora: %s",
                    len(features) - len(varying),
                    ", ".join(f for f in features if f not in varying))
    features = varying
    logger.info("%d distinct measurements enter adversarial validation, %d seeds",
                len(features), len(args.seeds))

    results = []
    logger.info("per-class adversarial validation (2017 = 0, 2018 = 1)")
    for name in SHARED_CLASSES:
        a = c17.X[(c17.y == name).to_numpy()]
        b = c18.X[(c18.y == name).to_numpy()]
        row = run_class(name, a, b, features, tuple(args.seeds))
        if row:
            results.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "findings.json").write_text(
        json.dumps(
            {
                "n_features": len(features),
                "features": features,
                "duplicate_representatives": stands_for,
                "min_per_side": MIN_PER_SIDE,
                "max_per_side": MAX_PER_SIDE,
                "n_repeats": N_REPEATS,
                "seeds": list(args.seeds),
                "classes": results,
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    long = pd.concat([
        pd.DataFrame(r["importance"]).assign(shared_class=r["shared_class"], auc=r["auc"])
        for r in results
    ], ignore_index=True)
    long.to_csv(args.out_dir / "importance.csv", index=False)

    logger.info("=" * 70)
    logger.info("mean permutation importance across classes -- the fingerprint ranking")
    overall = (
        long.groupby("feature")["importance"]
        .agg(["mean", "max", "count"])
        .sort_values("mean", ascending=False)
    )
    overall.to_csv(args.out_dir / "ranking.csv")
    for feature, row in overall.head(15).iterrows():
        logger.info("  %-28s mean=%.4f  max=%.4f", feature, row["mean"], row["max"])
    logger.info("wrote findings.json, importance.csv and ranking.csv to %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
