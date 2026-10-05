"""Run the four train/test dataset combinations under the shared protocol.

Protocol: 77 shared features, 7 shared classes, 300,000 rows per dataset,
train 70 / test 30, repeated over five seeds. Both split modes
are run -- ``random`` because it is the protocol's own and makes our numbers
comparable with the advisor's, ``chronological`` because it is the one that does
not let a model see the back half of an attack window while being scored on the
front half.

Per (mode, seed) a model is trained on each dataset's train split and scored on
*both* datasets' test splits, which yields all four combinations from two fits
rather than four:

    2017 -> 2017    within        2017 -> 2018    transfer
    2018 -> 2018    within        2018 -> 2017    transfer

**No hyper-parameter tuning.** Every model uses its library defaults, identical
on both sides. The gap being measured here is 0.4-0.6 macro-F1; tuning moves
these models by 0.01-0.05. Importing the 2018 tuned parameters would have been
worse than useless -- they were searched against 15 classes on one dataset, so
applying them to both sides would hand 2018 an advantage that looks like a
dataset effect.

The preprocessor is fitted on the training dataset only and applied unchanged
to both test sets. That is what transfer means: the target is scored using the
source's statistics, with nothing adapted to it.

Writes to results/crossdataset/<run-name>/. Reads nothing it can write.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.crossdataset.labels import SHARED_CLASSES  # noqa: E402
from src.crossdataset.loaders import Corpus, load_ids2017, load_ids2018  # noqa: E402
from src.crossdataset.splits import (  # noqa: E402
    TEST_FRAC,
    TRAIN_FRAC,
    Partition,
    chronological_split,
    random_split,
)
from src.ids2018.models import build_model, fit_model  # noqa: E402
from src.ids2018.preprocessing import Ids2018Preprocessor  # noqa: E402

logger = logging.getLogger("crossdataset")

DEFAULT_MODELS = [
    "lightgbm",
    "xgboost",
    "catboost",
    "random_forest",
    "mlp",
    "logistic_regression",
    "stacking",
]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
#: A class needs this many test rows before its F1 describes the model rather
#: than the handful of rows that happened to land in test.
MEASURABLE_MIN = 30


def seed_model(model, seed: int):
    """Point every random_state inside ``model`` at ``seed``.

    ``build_model`` bakes in ``RANDOM_STATE = 42`` from the project config, so
    without this the protocol's five seeds would vary only the split. Under the
    chronological mode -- which is deterministic and ignores the seed by design
    -- that left five identical runs reported as five samples, with a standard
    deviation of exactly zero standing in for "we never actually varied
    anything".

    ``get_params(deep=True)`` reaches nested estimators, so stacking's base
    learners are reseeded too; catboost spells it ``random_seed``.
    """
    keys = [
        k for k in model.get_params(deep=True)
        if k.split("__")[-1] in ("random_state", "random_seed")
    ]
    if keys:
        model.set_params(**dict.fromkeys(keys, seed))
    return model


def partition(corpus: Corpus, mode: str, seed: int) -> Partition:
    if mode == "random":
        return random_split(corpus.y, seed=seed)
    return chronological_split(corpus.y, corpus.order_key, corpus.group, seed=seed)


def score(y_true: pd.Series, y_pred: np.ndarray, encoder: LabelEncoder) -> dict:
    truth = encoder.inverse_transform(encoder.transform(y_true))
    pred = encoder.inverse_transform(pred_codes := np.asarray(y_pred))
    del pred_codes
    present = [c for c in SHARED_CLASSES if c in set(truth)]
    rep = classification_report(
        truth, pred, labels=present, output_dict=True, zero_division=0
    )
    per_class = {c: rep[c]["f1-score"] for c in present}
    support = {c: int(rep[c]["support"]) for c in present}
    measurable = [c for c in present if support[c] >= MEASURABLE_MIN]

    majority = pd.Series(["Benign"] * len(truth))
    out = {
        "accuracy": round(float((truth == pred).mean()), 4),
        "f1_macro_all": round(
            f1_score(truth, pred, labels=present, average="macro", zero_division=0), 4
        ),
        "f1_macro_measurable": round(
            float(np.mean([per_class[c] for c in measurable])), 4
        ),
        "baseline_accuracy": round(float((truth == majority.to_numpy()).mean()), 4),
        "baseline_f1_macro": round(
            f1_score(
                truth, majority, labels=present, average="macro", zero_division=0
            ),
            4,
        ),
        "n_test": len(truth),
    }
    out.update({f"f1__{c}": round(per_class[c], 4) for c in present})
    out.update({f"n__{c}": support[c] for c in present})
    return out


def resolve_dropped(args) -> list[str]:
    """The top-K ranked features, expanded to whole duplicate groups.

    Expansion is the point. Dropping ``Total Backward Packets`` while leaving
    ``Subflow Bwd Packets`` in place removes a name, not a measurement: the two
    hold identical values on every row of both corpora, so the model reads the
    signal off the survivor and the ablation tests nothing.
    """
    import pandas as pd

    audit = (PROJECT_ROOT / "results" / "crossdataset" / "feature_mapping_audit"
             / "findings.json")
    groups = json.loads(audit.read_text(encoding="utf-8"))["duplicate_groups"]
    member = {c: g for g in groups for c in g}

    ranked = pd.read_csv(args.ranking)["feature"].tolist()[: args.drop_top_k]
    out: list[str] = []
    for feature in [*ranked, *args.also_drop]:
        for column in member.get(feature, [feature]):
            if column not in out:
                out.append(column)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", default="protocol_v1")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--modes", nargs="+", default=["random", "chronological"])
    ap.add_argument("--accelerator", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--target", type=int, default=300_000)
    ap.add_argument(
        "--no-dst-port", action="store_true",
        help="drop Destination Port from the shared schema (76 features)",
    )
    ap.add_argument(
        "--drop-top-k", type=int, default=0,
        help="ablation: drop the K features ranked most dataset-identifying by "
             "adversarial validation, together with their exact duplicates",
    )
    ap.add_argument(
        "--ranking", type=Path,
        default=PROJECT_ROOT / "results" / "crossdataset" / "adversarial_validation"
        / "ranking.csv",
        help="feature ranking that --drop-top-k reads",
    )
    ap.add_argument(
        "--also-drop", nargs="*", default=[],
        help="extra columns to drop alongside the top-K, for near-duplicates that "
             "would otherwise carry the dropped signal onward",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="load both corpora, write corpora.json and stop before training -- "
             "the quickest way to check the data path and the loader guards",
    )
    args = ap.parse_args(argv)

    out_dir = PROJECT_ROOT / "results" / "crossdataset" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "run.log", encoding="utf-8"),
        ],
    )

    keep_port = not args.no_dst_port
    logger.info("loading corpora (Destination Port %s)", "kept" if keep_port else "dropped")
    corpora = {
        "ids2017": load_ids2017(keep_dst_port=keep_port, target=args.target),
        "ids2018": load_ids2018(keep_dst_port=keep_port, target=args.target),
    }
    dropped: list[str] = []
    if args.drop_top_k or args.also_drop:
        dropped = resolve_dropped(args)
        for corpus in corpora.values():
            corpus.X.drop(columns=dropped, inplace=True, errors="ignore")
        logger.info(
            "ablation: dropped %d of %d feature(s), %d remain -- %s",
            len(dropped), len(dropped) + corpora["ids2017"].X.shape[1],
            corpora["ids2017"].X.shape[1], ", ".join(dropped),
        )
        (out_dir / "dropped_features.json").write_text(
            json.dumps({"drop_top_k": args.drop_top_k, "dropped": dropped}, indent=2),
            encoding="utf-8",
        )

    (out_dir / "corpora.json").write_text(
        json.dumps({k: v.as_dict() for k, v in corpora.items()}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "split.json").write_text(
        json.dumps(
            {"shape": f"{TRAIN_FRAC:.0%}/{TEST_FRAC:.0%}".replace("%", ""), "modes": args.modes},
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.dry_run:
        for name, corpus in corpora.items():
            logger.info(
                "%s: %s rows from a corpus of %s, %d features, ordered by %s",
                name, f"{len(corpus.X):,}",
                f"{corpus.cleaning.rows_in:,}", corpus.X.shape[1], corpus.order_basis,
            )
        logger.info("--dry-run set: corpora written to %s, stopping before training",
                    out_dir / "corpora.json")
        return 0

    # One encoder over the shared vocabulary, so a class always gets the same
    # code no matter which dataset trained the model.
    encoder = LabelEncoder().fit(np.array(SHARED_CLASSES))

    rows: list[dict] = []
    started = time.perf_counter()
    for mode in args.modes:
        parts = {n: partition(c, mode, 0) if mode == "chronological" else None
                 for n, c in corpora.items()}
        for seed in args.seeds:
            if mode == "random":
                parts = {n: partition(c, mode, seed) for n, c in corpora.items()}
            for train_name, train_corpus in corpora.items():
                p = parts[train_name]
                pre = Ids2018Preprocessor()
                Xt = pre.fit_transform(train_corpus.X.iloc[p.train])
                yt = encoder.transform(train_corpus.y.iloc[p.train])

                for model_name in args.models:
                    t0 = time.perf_counter()
                    model = seed_model(build_model(model_name, args.accelerator), seed)
                    try:
                        model = fit_model(model_name, model, Xt, yt)
                    except Exception as exc:  # noqa: BLE001 - GPU OOM etc.
                        logger.warning(
                            "%s failed on %s (%s); retrying on CPU: %s",
                            model_name, args.accelerator, train_name, exc,
                        )
                        model = fit_model(
                            model_name, seed_model(build_model(model_name, "cpu"), seed), Xt, yt
                        )
                    fit_secs = time.perf_counter() - t0

                    for test_name, test_corpus in corpora.items():
                        tp = parts[test_name]
                        Xe = pre.transform(test_corpus.X.iloc[tp.test])
                        pred = model.predict(Xe)
                        row = {
                            "mode": mode,
                            "seed": seed,
                            "train": train_name,
                            "test": test_name,
                            "transfer": train_name != test_name,
                            "model": model_name,
                            "fit_seconds": round(fit_secs, 1),
                            **score(test_corpus.y.iloc[tp.test], pred, encoder),
                        }
                        rows.append(row)
                        logger.info(
                            "%s seed=%d %s->%s %-20s macro=%.4f measurable=%.4f "
                            "acc=%.4f (base %.4f)",
                            mode, seed, train_name[-4:], test_name[-4:], model_name,
                            row["f1_macro_all"], row["f1_macro_measurable"],
                            row["accuracy"], row["baseline_f1_macro"],
                        )
                    del model
                    pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)

    elapsed = (time.perf_counter() - started) / 60
    logger.info("done: %d result rows in %.1f min -> %s", len(rows), elapsed, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
