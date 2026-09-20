"""Prove the stacking ensemble cannot see the test set.

Reading the code says it cannot: ``build_stacking`` passes ``cv=3`` and
``passthrough=False``, so scikit-learn builds the meta-learner's training
matrix with ``cross_val_predict`` -- out-of-fold by construction -- and the
meta-learner never sees a raw feature. Code reading is not evidence, and
stacking was the model whose transfer score moved most when the corpora were
rebuilt, so it is the one worth putting under a test that can fail.

Four checks, each able to fail on its own:

1. **Disjointness.** No row is in both train and test, and -- the failure the
   corpus rebuild fixed -- no *duplicate* flow straddles the two. Before the
   rebuild the 2018 sample was deduplicated after sampling, so the same flow
   could be trained on and scored on.
2. **Out-of-fold meta-features.** Rebuild both candidate matrices, the
   out-of-fold one and the in-sample one, fit a fresh meta-learner on each, and
   see which set of coefficients the real ensemble reproduces.
3. **Label shuffle.** The end-to-end check, and the only one that covers the
   preprocessor, the split and the ensemble at once: destroy the relationship
   between features and labels, and any score above the majority baseline is
   information that arrived some other way.
4. **LabelSafeXGBClassifier alignment.** Not leakage but the correctness risk
   next to it: the subclass re-encodes ``y`` per fold and reports the original
   codes in ``classes_`` so ``cross_val_predict`` can pad the missing
   probability columns. If that padding is wrong the meta-learner is fed
   misaligned columns and stacking's score is wrong in either direction.

Run::

    python scripts/verify_stacking_leakage.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.crossdataset.labels import SHARED_CLASSES  # noqa: E402
from src.crossdataset.loaders import load_ids2017, load_ids2018  # noqa: E402
from src.crossdataset.splits import chronological_split  # noqa: E402
from src.ids2018.models import (  # noqa: E402
    LabelSafeXGBClassifier,
    build_model,
    fit_model,
)
from src.ids2018.preprocessing import Ids2018Preprocessor  # noqa: E402

logger = logging.getLogger("verify_stacking")

#: Enough rows for the ensemble to behave like the real one, few enough that
#: the whole audit runs in minutes. Stacking refits three base learners cv+1
#: times, so this is twelve fits.
AUDIT_ROWS = 40_000
SEED = 42

OUT_DIR = PROJECT_ROOT / "results" / "crossdataset" / "stacking_leakage_audit"

failures: list[str] = []
#: Every check, in the order it ran. Written to JSON so the evidence survives
#: the terminal -- the other three audits in this series leave a file behind and
#: this one did not, which made it the only result that had to be taken on
#: trust.
records: list[dict] = []
_section = ""


def check(name: str, passed: bool, detail: str) -> None:
    logger.info("  %-4s %-34s %s", "PASS" if passed else "FAIL", name, detail)
    records.append({"section": _section, "check": name, "passed": passed,
                    "detail": detail})
    if not passed:
        failures.append(f"{name}: {detail}")


def section(title: str) -> None:
    global _section
    _section = title
    logger.info(title)


# ----------------------------------------------------------------------
def check_disjointness(corpus, part, features: list[str]) -> None:
    """No shared row, and no shared *flow*, between train and test."""
    section("1. train/test disjointness")

    overlap = set(part.train.tolist()) & set(part.test.tolist())
    check("no shared row index", not overlap, f"{len(overlap)} shared position(s)")

    # The leak the corpus rebuild closed: identical flows on both sides.
    frame = corpus.X[features].copy()
    frame["_y"] = corpus.y.to_numpy()
    tr = frame.iloc[part.train].drop_duplicates()
    te = frame.iloc[part.test].drop_duplicates()
    shared = pd.merge(tr, te, how="inner")
    check(
        "no duplicate flow across split",
        len(shared) == 0,
        f"{len(shared):,} distinct flow(s) appear in both train and test",
    )


def check_out_of_fold(model, X, y) -> None:
    """Show directly which matrix the meta-learner was trained on.

    An earlier version of this check compared a base learner's in-sample
    accuracy against its out-of-fold accuracy and demanded a visible gap. That
    was the wrong instrument twice over. The gap is 0.0015 here because the
    within-dataset task is easy -- a base learner at 99.7% out-of-fold has no
    room left to overfit into -- so the check failed on a healthy ensemble. And
    it was measuring the wrong thing anyway: training a meta-learner on
    in-sample base predictions is a methodology flaw that *degrades* the test
    score, not a route for test data to reach the model.

    The direct question is which matrix ``final_estimator_`` was actually fitted
    on. Rebuild both candidates -- one out-of-fold, one in-sample -- fit a fresh
    copy of the meta-learner on each, and see which set of coefficients the
    real ensemble reproduces. Only one of them can match.
    """
    section("2. meta-features are out-of-fold")

    check(
        "cv is set",
        model.cv not in (None, 0, 1),
        f"StackingClassifier(cv={model.cv}) -- cross_val_predict builds the meta matrix",
    )
    check(
        "passthrough is off",
        model.passthrough is False,
        f"passthrough={model.passthrough} -- meta-learner sees only OOF probabilities",
    )

    fitted = fit_model("stacking", clone(model), X, y)

    oof_blocks, in_sample_blocks = [], []
    for _, estimator in model.estimators:
        oof_blocks.append(
            cross_val_predict(clone(estimator), X, y, cv=model.cv,
                              method="predict_proba")
        )
        fitted_base = clone(estimator).fit(X, y)
        in_sample_blocks.append(fitted_base.predict_proba(X))

    candidates = {
        "out-of-fold": np.hstack(oof_blocks),
        "in-sample": np.hstack(in_sample_blocks),
    }
    distance = {}
    for name, matrix in candidates.items():
        meta = clone(model.final_estimator).fit(matrix, y)
        distance[name] = float(
            np.abs(meta.coef_ - fitted.final_estimator_.coef_).mean()
        )

    winner = min(distance, key=distance.get)
    check(
        "meta-learner was fitted on OOF",
        winner == "out-of-fold",
        f"coefficient distance: out-of-fold {distance['out-of-fold']:.6f}, "
        f"in-sample {distance['in-sample']:.6f} -- closest is {winner!r}",
    )


def check_label_shuffle(X_tr, y_tr, X_te, y_te) -> None:
    """With the feature/label relationship destroyed, nothing may be learnable."""
    section("3. label shuffle (end-to-end)")

    rng = np.random.default_rng(SEED)
    y_shuffled = rng.permutation(y_tr)

    model = fit_model("stacking", build_model("stacking", "cpu"), X_tr, y_shuffled)
    pred = np.asarray(model.predict(X_te)).ravel().astype(int)

    present = np.unique(y_te)
    shuffled_f1 = f1_score(y_te, pred, labels=present, average="macro", zero_division=0)
    majority = np.full_like(y_te, np.bincount(y_tr).argmax())
    baseline_f1 = f1_score(
        y_te, majority, labels=present, average="macro", zero_division=0
    )

    check(
        "shuffled labels are not learnable",
        shuffled_f1 <= baseline_f1 + 0.02,
        f"macro-F1 {shuffled_f1:.4f} against a majority baseline of {baseline_f1:.4f}",
    )

    real = fit_model("stacking", build_model("stacking", "cpu"), X_tr, y_tr)
    real_pred = np.asarray(real.predict(X_te)).ravel().astype(int)
    real_f1 = f1_score(y_te, real_pred, labels=present, average="macro", zero_division=0)
    check(
        "real labels are learnable",
        real_f1 > shuffled_f1 + 0.10,
        f"real macro-F1 {real_f1:.4f} vs shuffled {shuffled_f1:.4f} -- the test has power",
    )


def check_label_safe_xgb() -> None:
    """A fold missing an interior class must still yield aligned columns."""
    section("4. LabelSafeXGBClassifier column alignment")

    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(600, 6))
    # Codes 0,1,3,4: code 2 is absent, exactly what a fold hits when a class
    # has too few rows to appear in it. Plain XGBoost rejects this outright.
    y = np.repeat([0, 1, 3, 4], 150)

    model = LabelSafeXGBClassifier(n_estimators=10, max_depth=3, verbosity=0)
    model.fit(X, y)

    check(
        "classes_ reports the original codes",
        np.array_equal(model.classes_, np.array([0, 1, 3, 4])),
        f"classes_={model.classes_.tolist()}",
    )
    check(
        "predict returns original codes",
        set(np.unique(model.predict(X)).tolist()).issubset({0, 1, 3, 4}),
        f"predicted codes {sorted(set(np.unique(model.predict(X)).tolist()))}",
    )
    # The claim in the docstring: cross_val_predict reads classes_ and pads the
    # columns itself, so the meta-learner sees one column per class every time.
    proba = cross_val_predict(
        LabelSafeXGBClassifier(n_estimators=10, max_depth=3, verbosity=0),
        X, y, cv=3, method="predict_proba",
    )
    check(
        "cross_val_predict pads to one column per class",
        proba.shape == (600, 4),
        f"meta matrix shape {proba.shape}, expected (600, 4)",
    )
    check(
        "padded rows still sum to one",
        bool(np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)),
        f"row sums in [{proba.sum(axis=1).min():.4f}, {proba.sum(axis=1).max():.4f}]",
    )


# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Checks 2-4 are protocol-level and cannot differ by corpus, but check 1 can:
    # 2018 carried 4.7M duplicate flows before the rebuild, so it is the corpus
    # where a flow straddling the split was actually possible. Running both is
    # the difference between a claim about the protocol and a claim about the
    # data the protocol was run on.
    ap.add_argument("--dataset", choices=["ids2017", "ids2018"], default="ids2017")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("loading %s", args.dataset)
    corpus = load_ids2017() if args.dataset == "ids2017" else load_ids2018()
    features = list(corpus.X.columns)
    part = chronological_split(corpus.y, corpus.order_key, corpus.group, seed=SEED)

    check_disjointness(corpus, part, features)

    encoder = LabelEncoder().fit(np.array(SHARED_CLASSES))
    rng = np.random.default_rng(SEED)
    tr = rng.choice(part.train, size=min(AUDIT_ROWS, len(part.train)), replace=False)
    te = rng.choice(part.test, size=min(AUDIT_ROWS // 2, len(part.test)), replace=False)

    pre = Ids2018Preprocessor()
    X_tr = pre.fit_transform(corpus.X.iloc[tr])
    X_te = pre.transform(corpus.X.iloc[te])
    y_tr = encoder.transform(corpus.y.iloc[tr])
    y_te = encoder.transform(corpus.y.iloc[te])
    logger.info("audit set: train %s, test %s", X_tr.shape, X_te.shape)

    check_out_of_fold(build_model("stacking", "cpu"), X_tr, y_tr)
    check_label_shuffle(X_tr, y_tr, X_te, y_te)
    check_label_safe_xgb()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.dataset}.json"
    out.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "audit_rows": AUDIT_ROWS,
                "seed": SEED,
                "passed": not failures,
                "n_checks": len(records),
                "n_failed": len(failures),
                "checks": records,
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("wrote %s", out)

    logger.info("=" * 66)
    if failures:
        logger.error("FAIL: %d check(s) did not hold", len(failures))
        for f in failures:
            logger.error("  %s", f)
        return 1
    logger.info("PASS: stacking cannot see the test set (%s)", args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
