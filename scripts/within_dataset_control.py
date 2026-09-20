"""Does a capture boundary *inside* one corpus separate as well as the corpus boundary?

Adversarial validation tells CICIDS2017 from CSE-CIC-IDS2018 at AUC 0.9994 to
1.0000 in every testable class, and that was read as a dataset fingerprint --
something about how each corpus was produced marking its rows. The null control
already in ``diagnose_av.py`` splits one corpus into *random* halves and returns
0.46-0.53, which establishes that the procedure cannot invent separation. It
does not touch the claim, because shuffling rows mixes every capture day and
every hour into both halves: a difference that tracks *when* a flow was captured
cannot survive a random split, so its absence there says nothing.

This is the control that can actually fail. Split one corpus along a boundary
that the random split destroys -- early half against late half, or one set of
capture files against another -- and run the identical procedure. If those
separate about as well as 2017 against 2018 does, what was called a dataset
fingerprint is an ordinary file-and-time effect and the cross-dataset result is
measuring the passage of time. If they separate far less, the corpus boundary is
doing work no within-corpus boundary does.

Two design decisions carry the argument
---------------------------------------
**Matched n.** Every contrast for a class draws the same number of rows a side:
the fewest any available contrast can supply. AUC moves with sample size, so a
cross-dataset draw of 8,000 against a within-dataset draw of 400 would confound
what is being measured with what is being controlled for.

**Groups assigned by size, not by date.** Splitting capture files in
chronological order would rebuild the time contrast under a second name. Greedy
balancing puts early and late captures on both sides, leaving file identity
rather than date as what has to be separated.

A confound worth stating: in both corpora a capture file *is* a capture day, so
the group contrast is a time contrast at coarser granularity. The two differ in
granularity, not in kind. Together they answer whether *any* within-corpus
capture boundary separates as strongly as the corpus boundary -- which is the
question the fingerprint claim turns on.

The decision rule was written first, in
``results/crossdataset/adversarial_validation/CONTROL_PREREGISTRATION.md``.

Run::

    python scripts/within_dataset_control.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ``adversarial_validation`` is the sibling script, imported rather than copied
# so the feature set and the classifier are defined exactly once: the same
# duplicate-collapsed, constant-dropped 60 columns and the same LightGBM the
# main run uses. A second definition here could drift from it and the
# comparison would quietly stop being like-for-like.
from adversarial_validation import (  # noqa: E402
    build_classifier,
    representative_features,
)

from src.crossdataset.labels import SHARED_CLASSES  # noqa: E402
from src.crossdataset.loaders import load_ids2017, load_ids2018  # noqa: E402

logger = logging.getLogger("within_dataset_control")

OUT = (
    PROJECT_ROOT
    / "results"
    / "crossdataset"
    / "adversarial_validation"
    / "within_dataset_control.json"
)

SEEDS = (42, 43, 44, 45, 46)
TEST_SIZE = 0.3

#: Lower than the main run's 500. Too few rows lets a model memorise and pushes
#: the AUC *up*, which argues against the fingerprint conclusion rather than
#: for it -- so a small-n control errs in the conservative direction here.
MIN_PER_SIDE = 300
#: Upper bound per side. The signal saturates long before this and every
#: contrast has to be run at the same n anyway.
MAX_PER_SIDE = 8_000

#: Cross-dataset AUC at or above this marks a class as one the decision rule
#: applies to. Every class currently clears it.
SEPARABLE = 0.99


def halves_by_time(order: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Earliest half against latest half, by the corpus's own ordering key.

    The median is taken over the rows of this class only, so each class is cut
    at the midpoint of *its own* activity rather than of the corpus. For an
    attack class confined to one capture file that means splitting a single
    attack window in half, which is the strongest available within-corpus
    contrast for it.
    """
    rank = order.rank(method="first").to_numpy()
    cut = np.median(rank)
    return np.where(rank <= cut)[0], np.where(rank > cut)[0]


def halves_by_group(group: pd.Series) -> tuple[np.ndarray, np.ndarray] | None:
    """Capture files split into two sides of comparable size, or None.

    None when the class lives in a single capture, which is the normal case for
    2017 attack classes -- every one of them occurs in exactly one file, so
    there is no file boundary to test and only the time contrast applies.

    Largest-first into whichever side is currently lighter. Ordering the files
    by date instead would make this a second time split under another name.
    """
    sizes = group.value_counts()
    if len(sizes) < 2:
        return None

    sides: tuple[list[str], list[str]] = ([], [])
    totals = [0, 0]
    for name, count in sizes.items():
        target = 0 if totals[0] <= totals[1] else 1
        sides[target].append(str(name))
        totals[target] += int(count)
    if not sides[0] or not sides[1]:
        return None

    values = group.astype(str).to_numpy()
    return (
        np.where(np.isin(values, sides[0]))[0],
        np.where(np.isin(values, sides[1]))[0],
    )


def halves_at_random(n_rows: int, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """The existing null: a boundary that is known to mean nothing."""
    idx = np.random.default_rng(seed).permutation(n_rows)
    half = n_rows // 2
    return idx[:half], idx[half : 2 * half]


def draw(
    a: pd.DataFrame, b: pd.DataFrame, features: list[str], n: int, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """``n`` rows from each side, labelled by which side they came from."""
    rng = np.random.default_rng(seed)
    pick_a = rng.choice(len(a), size=n, replace=False)
    pick_b = rng.choice(len(b), size=n, replace=False)
    X = pd.concat([a.iloc[pick_a][features], b.iloc[pick_b][features]], ignore_index=True)
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    return X, y


def score_contrast(
    a: pd.DataFrame, b: pd.DataFrame, features: list[str], n: int
) -> dict[str, float | list[float]]:
    """AUC over every seed, for the ensemble and for a single threshold.

    The stump is carried alongside because it cannot memorise: one threshold on
    one column, whatever the sample size. Where the ensemble and the stump agree
    the separation is broad and shallow rather than an artefact of capacity, and
    that reading has to hold for the control as much as for the main result.
    """
    ensemble, stump = [], []
    for seed in SEEDS:
        X, y = draw(a, b, features, n, seed)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
        )
        model = build_classifier(seed).fit(X_tr, y_tr)
        ensemble.append(float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])))

        one = DecisionTreeClassifier(max_depth=1, random_state=seed).fit(X_tr, y_tr)
        stump.append(float(roc_auc_score(y_te, one.predict_proba(X_te)[:, 1])))

    return {
        "auc": round(float(np.mean(ensemble)), 4),
        "auc_sd": round(float(np.std(ensemble)), 5),
        "auc_per_seed": [round(v, 4) for v in ensemble],
        "stump": round(float(np.mean(stump)), 4),
    }


def contrasts_for(
    name: str, rows: dict[str, dict[str, pd.Series | pd.DataFrame]]
) -> dict[str, dict]:
    """Every boundary that can be drawn for one class, with what it is.

    Each entry carries the two frames plus a note saying which boundary it is,
    because "0.52" means something different depending on whether the boundary
    was two corpora, two capture days or nothing at all.
    """
    out: dict[str, dict] = {}

    a, b = rows["ids2017"]["X"], rows["ids2018"]["X"]
    out["cross"] = {
        "a": a,
        "b": b,
        "boundary": "CICIDS2017 against CSE-CIC-IDS2018",
        "kind": "cross",
    }

    for corpus in ("ids2017", "ids2018"):
        X = rows[corpus]["X"]
        early, late = halves_by_time(rows[corpus]["order"])
        out[f"time_{corpus}"] = {
            "a": X.iloc[early],
            "b": X.iloc[late],
            "boundary": f"earliest against latest half of {corpus}, by capture order",
            "kind": "within",
        }

        split = halves_by_group(rows[corpus]["group"])
        if split is not None:
            side_a, side_b = split
            groups = rows[corpus]["group"].astype(str)
            out[f"group_{corpus}"] = {
                "a": X.iloc[side_a],
                "b": X.iloc[side_b],
                "boundary": (
                    f"{groups.iloc[side_a].nunique()} capture(s) against "
                    f"{groups.iloc[side_b].nunique()} in {corpus}"
                ),
                "kind": "within",
                "sides": [
                    sorted(groups.iloc[side_a].unique().tolist()),
                    sorted(groups.iloc[side_b].unique().tolist()),
                ],
            }

        first, second = halves_at_random(len(X))
        out[f"null_{corpus}"] = {
            "a": X.iloc[first],
            "b": X.iloc[second],
            "boundary": f"two random halves of {corpus} -- no boundary at all",
            "kind": "null",
        }

    return out


def run_class(name: str, rows: dict, features: list[str]) -> dict | None:
    """One class, every contrast, all at the same rows per side."""
    available = contrasts_for(name, rows)
    n = min(min(len(c["a"]), len(c["b"])) for c in available.values())
    n = min(n, MAX_PER_SIDE)

    if n < MIN_PER_SIDE:
        logger.info(
            "  %-12s skipped: the tightest contrast supplies %d row(s) a side, " "below %d",
            name,
            n,
            MIN_PER_SIDE,
        )
        return None

    results = {}
    for key, contrast in available.items():
        scored = score_contrast(contrast["a"], contrast["b"], features, n)
        results[key] = {
            **scored,
            "boundary": contrast["boundary"],
            "kind": contrast["kind"],
            "n_available_per_side": int(min(len(contrast["a"]), len(contrast["b"]))),
            **({"sides": contrast["sides"]} if "sides" in contrast else {}),
        }

    within = [v["auc"] for v in results.values() if v["kind"] == "within"]
    row = {
        "shared_class": name,
        "n_per_side": int(n),
        "contrasts": results,
        "cross_auc": results["cross"]["auc"],
        # The *highest* within-dataset contrast, so the comparison is against
        # the strongest case for the rival explanation rather than the most
        # convenient one.
        "within_auc_max": round(max(within), 4) if within else None,
        "gap": round(results["cross"]["auc"] - max(within), 4) if within else None,
    }

    logger.info(
        "  %-12s n=%-6s cross=%.4f  within-max=%.4f  gap=%+.4f",
        name,
        f"{n:,}",
        row["cross_auc"],
        row["within_auc_max"],
        row["gap"],
    )
    for key, value in results.items():
        logger.info(
            "  %-12s   %-16s AUC %.4f ±%.4f  stump %.4f   %s",
            "",
            key,
            value["auc"],
            value["auc_sd"],
            value["stump"],
            value["boundary"],
        )
    return row


def verdict(rows: list[dict]) -> dict:
    """Apply the pre-registered rule. No thresholds decided after the fact."""
    judged = [r for r in rows if r["cross_auc"] >= SEPARABLE and r["gap"] is not None]
    if not judged:
        return {"verdict": "inconclusive", "reason": "no class reached the cross-dataset floor"}

    confirmed = all(r["gap"] >= 0.10 and r["within_auc_max"] < 0.95 for r in judged)
    refuted = sum(r["within_auc_max"] >= SEPARABLE for r in judged) > len(judged) / 2

    if confirmed:
        name, reason = (
            "fingerprint confirmed",
            "every separable class keeps its within-dataset AUC at least 0.10 lower "
            "and below 0.95",
        )
    elif refuted:
        name, reason = (
            "fingerprint refuted",
            "a within-corpus capture boundary separates as strongly as the corpus "
            "boundary in most separable classes",
        )
    else:
        name, reason = (
            "partial",
            "the rule's two branches both fail; the per-class table is the finding",
        )
    return {
        "verdict": name,
        "reason": reason,
        "n_classes_judged": len(judged),
        "min_gap": round(min(r["gap"] for r in judged), 4),
        "max_within_auc": round(max(r["within_auc_max"] for r in judged), 4),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    c17, c18 = load_ids2017(), load_ids2018()
    features, _ = representative_features(list(c17.X.columns))
    pooled = pd.concat([c17.X[features], c18.X[features]], ignore_index=True)
    features = [f for f in features if pooled[f].nunique(dropna=False) > 1]
    logger.info("%d features, seeds %s, matched n per class", len(features), list(SEEDS))
    logger.info("")

    results = []
    for name in SHARED_CLASSES:
        rows = {}
        for corpus in (c17, c18):
            mask = (corpus.y == name).to_numpy()
            rows[corpus.name] = {
                "X": corpus.X[mask].reset_index(drop=True),
                "order": corpus.order_key[mask].reset_index(drop=True),
                "group": corpus.group[mask].reset_index(drop=True),
            }
        row = run_class(name, rows, features)
        if row:
            results.append(row)
        logger.info("")

    decided = verdict(results)
    logger.info("=" * 78)
    logger.info("verdict: %s -- %s", decided["verdict"], decided["reason"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "question": (
                    "Does a capture boundary inside one corpus separate as well "
                    "as the boundary between corpora?"
                ),
                "seeds": list(SEEDS),
                "min_per_side": MIN_PER_SIDE,
                "max_per_side": MAX_PER_SIDE,
                "separable_threshold": SEPARABLE,
                "n_features": len(features),
                "preregistration": "CONTROL_PREREGISTRATION.md",
                **decided,
                "classes": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
