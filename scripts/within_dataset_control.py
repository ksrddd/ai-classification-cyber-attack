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
# so the feature set is defined exactly once: the same duplicate-collapsed,
# constant-dropped 60 columns the main run measured over. A second definition
# here could drift from it and the comparison would quietly stop being
# like-for-like. The models come from the transfer protocol's own factory for
# the same reason.
from adversarial_validation import representative_features  # noqa: E402

from src.crossdataset.labels import SHARED_CLASSES  # noqa: E402
from src.crossdataset.loaders import load_ids2017, load_ids2018  # noqa: E402
from src.ids2018.models import build_model, fit_model  # noqa: E402
from src.ids2018.preprocessing import Ids2018Preprocessor  # noqa: E402

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

#: The protocol's seven models, in the order ``run_crossdataset.py`` lists
#: them. A control that holds only for the model that produced the headline is
#: a weaker claim than one that holds for every model in the study, and the
#: seven disagree enough elsewhere in this project that the question is worth
#: asking rather than assuming.
PROTOCOL_MODELS = (
    "lightgbm",
    "xgboost",
    "catboost",
    "random_forest",
    "mlp",
    "logistic_regression",
    "stacking",
)
#: Reported beside them, and not one of the seven: a decision stump is the
#: capacity floor. One threshold on one column cannot memorise a draw at any
#: sample size, so where it separates, the separation is in the features.
STUMP = "stump"

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


def seed_model(model, seed: int):
    """Point every ``random_state`` inside ``model`` at ``seed``.

    Copied in spirit from ``run_crossdataset.py``: ``build_model`` bakes in the
    project's RANDOM_STATE, so without this the five seeds would vary the draw
    and the split but leave every model identical, and the reported spread
    would understate what re-running actually costs. ``deep=True`` reaches
    stacking's base learners; catboost spells the key ``random_seed``.
    """
    keys = [
        k
        for k in model.get_params(deep=True)
        if k.split("__")[-1] in ("random_state", "random_seed")
    ]
    if keys:
        model.set_params(**dict.fromkeys(keys, seed))
    return model


def score_one(name: str, X_tr, y_tr, X_te, y_te, seed: int) -> float:
    """Fit one model on the training half and score AUC on the held-out half.

    The stump is built here rather than through ``build_model`` because it is
    not one of the protocol's models -- it is the capacity floor they are read
    against, and giving it a depth of one is the whole point.
    """
    if name == STUMP:
        model = DecisionTreeClassifier(max_depth=1, random_state=seed)
        model.fit(X_tr, y_tr)
    else:
        model = fit_model(name, seed_model(build_model(name), seed), X_tr, y_tr)
    return float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))


def score_contrast(
    a: pd.DataFrame,
    b: pd.DataFrame,
    features: list[str],
    n: int,
    models: tuple[str, ...] = PROTOCOL_MODELS + (STUMP,),
) -> dict:
    """AUC over every seed, for every model, on one boundary.

    Every model sees the *same* draw and the same split at each seed, so a
    difference between two rows is the model and nothing else. Preprocessing is
    fitted on the training half alone and applied to the other, exactly as the
    transfer protocol does it -- the linear and neural models need the scaling
    and the trees are indifferent to it, and fitting it on the whole draw first
    would leak the held-out half into every number here.

    ``auc`` and ``stump`` stay at the top level and stay LightGBM and the
    stump, because the earlier single-model run is quoted in the report and in
    the pre-registered decision rule; the rest arrives under ``per_model``.
    """
    scores: dict[str, list[float]] = {name: [] for name in models}

    for seed in SEEDS:
        X, y = draw(a, b, features, n, seed)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
        )
        pre = Ids2018Preprocessor()
        Xt = pre.fit_transform(X_tr)
        Xe = pre.transform(X_te)
        for name in models:
            scores[name].append(score_one(name, Xt, y_tr, Xe, y_te, seed))

    per_model = {
        name: {
            "auc": round(float(np.mean(v)), 4),
            "auc_sd": round(float(np.std(v)), 5),
            "auc_per_seed": [round(x, 4) for x in v],
        }
        for name, v in scores.items()
    }
    headline = per_model.get("lightgbm") or next(iter(per_model.values()))
    return {
        "auc": headline["auc"],
        "auc_sd": headline["auc_sd"],
        "auc_per_seed": headline["auc_per_seed"],
        "stump": per_model[STUMP]["auc"] if STUMP in per_model else None,
        "per_model": per_model,
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


def summarise_models(results: dict[str, dict]) -> dict[str, dict]:
    """The same cross-against-within comparison, once per model.

    A conclusion that holds only for the model that produced the headline is a
    weaker claim than one every model agrees on, and this is the table that
    says which of the two we have. ``within_auc_max`` takes the highest
    within-dataset boundary *for that model*, keeping each row internally
    consistent rather than mixing one model's cross figure with another's
    within figure.
    """
    names = list(next(iter(results.values()))["per_model"])
    out = {}
    for model in names:
        cross = results["cross"]["per_model"][model]["auc"]
        within = [v["per_model"][model]["auc"] for v in results.values() if v["kind"] == "within"]
        null = [v["per_model"][model]["auc"] for v in results.values() if v["kind"] == "null"]
        out[model] = {
            "cross_auc": cross,
            "within_auc_max": round(max(within), 4) if within else None,
            "gap": round(cross - max(within), 4) if within else None,
            "null_auc_max": round(max(null), 4) if null else None,
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
        "per_model": summarise_models(results),
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
        # The stump is the capacity floor, not one of the seven; folding it
        # into the range would make every row look as though some protocol
        # model had barely separated the boundary.
        aucs = [value["per_model"][m]["auc"] for m in PROTOCOL_MODELS]
        logger.info(
            "  %-12s   %-16s lgbm %.4f  7 models %.4f-%.4f  stump %.4f   %s",
            "",
            key,
            value["auc"],
            min(aucs),
            max(aucs),
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
        "per_model": verdict_per_model(rows),
    }


def verdict_per_model(rows: list[dict]) -> dict[str, dict]:
    """The same rule, applied to each model's own numbers.

    Reported alongside the headline verdict, never in place of it. The rule was
    pre-registered against a single model and that verdict is what the report
    quotes; running it seven more times afterwards is a robustness check, and
    presenting the friendliest of the eight as *the* answer would be exactly
    the freedom the pre-registration exists to remove.
    """
    if not rows:
        return {}
    out = {}
    for model in next(iter(rows)).get("per_model", {}):
        judged = [
            r["per_model"][model]
            for r in rows
            if r["per_model"][model]["cross_auc"] >= SEPARABLE
            and r["per_model"][model]["gap"] is not None
        ]
        if not judged:
            out[model] = {"verdict": "inconclusive", "n_classes_judged": 0}
            continue
        confirmed = all(j["gap"] >= 0.10 and j["within_auc_max"] < 0.95 for j in judged)
        refuted = sum(j["within_auc_max"] >= SEPARABLE for j in judged) > len(judged) / 2
        out[model] = {
            "verdict": (
                "fingerprint confirmed"
                if confirmed
                else "fingerprint refuted" if refuted else "partial"
            ),
            "n_classes_judged": len(judged),
            "min_gap": round(min(j["gap"] for j in judged), 4),
            "max_within_auc": round(max(j["within_auc_max"] for j in judged), 4),
            "max_null_auc": round(
                max(j["null_auc_max"] for j in judged if j["null_auc_max"] is not None),
                4,
            ),
        }
    return out


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
    logger.info("")
    logger.info("%-20s %-22s %9s %9s %9s", "model", "verdict", "min gap", "within", "null")
    for model, v in decided.get("per_model", {}).items():
        if v.get("n_classes_judged"):
            logger.info(
                "%-20s %-22s %9.4f %9.4f %9.4f",
                model,
                v["verdict"],
                v["min_gap"],
                v["max_within_auc"],
                v["max_null_auc"],
            )

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
