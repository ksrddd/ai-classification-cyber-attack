"""SHAP for a CSE-CIC-IDS2018 bundle, for the dashboard's Explainability page.

The IDS2018 pipeline has no explain stage of its own. This rebuilds the
bundle's locked test split (the same way ``src.ids2018.analyze`` does), checks
it against the confusion matrix training wrote, and runs the project's SHAP
analyzer on each saved model. Output lands in
``results/ids2018/<bundle>/shap/<model>/``, where ``/api/shap?bundle=`` reads it.

How the numbers are made, and why:

- Rows are sampled per class (``--per-class`` each, all rows of a smaller
  class), and each class is explained on its own rows only. A random sample
  is ~85% Benign and holds almost no rows of the rare attacks.
- Importances are also given as a share of the model's total mean |SHAP|.
  Raw values are log-odds for XGBoost/LightGBM/CatBoost, probability for
  RandomForest/MLP and logit for LR, so only shares compare across models.
- LR uses the exact LinearExplainer. MLP uses KernelExplainer with
  nsamples="auto" and a k-means background, not 100 coalitions on 69 features.
- A class with fewer than ``--per-class`` test rows is topped up with
  *supplementary* rows: flows of that class in the raw corpus that the
  nested 500k sample never drew, minus exact duplicates of any sampled row,
  so no model has seen them. They are counted apart from the locked test
  rows (``per_class_pool``) and never enter any metric.
- Each class also records how much probability the model gives the class on
  its own rows. A class the model rarely predicts has near-zero attributions,
  and an unstable ranking there means "nothing to explain", not noise.
- Tree explainers run on ``--jobs`` row chunks in parallel; the values are
  identical to a serial run.
- Every model is run on ``--seeds`` different samples. The JSON keeps the
  mean and SD of each share, the pairwise Spearman rho of the full rankings
  and the Jaccard overlap of the top 10, so an unstable ranking is visible.

Stacking is skipped by default: KernelExplainer over its three base models
takes hours.

Run::

    python scripts/shap_ids2018.py --bundle 500k_temporal_tuned_fair
    python scripts/shap_ids2018.py --bundle 300k_temporal_tuned --model catboost --force
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.shap_analyzer import ShapResult, analyze_model  # noqa: E402
from src.ids2018.analyze import _check_against_saved, rebuild_test_split  # noqa: E402
from src.ids2018 import data_loader as dl  # noqa: E402
from src.ids2018.config import CACHE_DIR, MODELS_DIR, OUTPUT_DIR, sample_cache_path  # noqa: E402
from src.ids2018.preprocessing import split_features_labels  # noqa: E402
from src.ids2018.train_ids2018 import discover_csvs  # noqa: E402

logger = logging.getLogger("shap_ids2018")

MODELS = ("catboost", "lightgbm", "xgboost", "random_forest", "logistic_regression", "mlp")
TOP_K = 15
STABLE_TOP = 10


def _share(v: np.ndarray) -> np.ndarray:
    total = v.sum()
    return v / total if total > 0 else v


def _stability(vectors: list[np.ndarray]) -> dict:
    """Pairwise agreement between seeds' full importance vectors.

    Rho is None when a vector is constant (e.g. all-zero SHAP on a class's
    single row), where rank correlation is undefined; NaN is not valid JSON.
    """
    tops = [set(np.argsort(v)[::-1][:STABLE_TOP]) for v in vectors]
    rho = [spearmanr(a, b).statistic for a, b in combinations(vectors, 2)]
    jac = [len(a & b) / len(a | b) for a, b in combinations(tops, 2)]
    defined = [r for r in rho if np.isfinite(r)]
    return {
        "spearman_rho_min": float(np.min(defined)) if len(defined) == len(rho) else None,
        "spearman_rho_mean": float(np.mean(defined)) if len(defined) == len(rho) else None,
        f"top{STABLE_TOP}_jaccard_min": float(np.min(jac)),
        f"top{STABLE_TOP}_jaccard_mean": float(np.mean(jac)),
    }


def _ranked(vectors: list[np.ndarray], features: list[str], in_all_top: set[int]) -> list:
    """[feature, mean share, sd share, in top-10 on every seed] by mean share."""
    shares = np.array([_share(v) for v in vectors])
    mean = shares.mean(axis=0)
    sd = shares.std(axis=0, ddof=1) if len(vectors) > 1 else np.zeros_like(mean)
    order = np.argsort(mean)[::-1][:TOP_K]
    return [[features[i], float(mean[i]), float(sd[i]), i in in_all_top] for i in order]


def _all_top(vectors: list[np.ndarray]) -> set[int]:
    return set.intersection(*(set(np.argsort(v)[::-1][:STABLE_TOP]) for v in vectors))


def _row_keys(X: pd.DataFrame) -> pd.Index:
    return pd.Index(pd.util.hash_pandas_object(X.astype("float64"), index=False))


def supplementary_rows(bundle: str, models_dir: Path, need: dict[str, int], cap: int):
    """Unseen raw-corpus rows for the classes in ``need``, preprocessed.

    Returns (X, y_encoded) or None. Cached per bundle under data/ids2018/.
    """
    meta = json.loads((models_dir / "metadata.json").read_text(encoding="utf-8"))
    cache = CACHE_DIR / f"shap_supplement_{bundle}_{cap}.parquet"
    if cache.is_file():
        raw = pd.read_parquet(cache)
    else:
        files = discover_csvs(Path(meta["raw_dir"]))
        index = dl.load_index(CACHE_DIR / "corpus_label_index.npz", files) or dl.scan_labels(files)
        drawn = dl.plan_nested_sample(index, int(meta["sample_size"]),
                                      int(meta["min_per_class"]), int(meta["random_state"]))
        codes = index.codes
        unseen = np.ones(len(codes), bool)
        unseen[drawn] = False
        rng = np.random.default_rng(0)
        wanted = []
        for code, name in enumerate(index.classes):
            if str(name) in need:
                pos = np.flatnonzero(unseen & (codes == code))
                # Oversample before dedup so the cap survives dropped duplicates.
                wanted.append(rng.permutation(pos)[: cap * 3])
        if not wanted:
            return None
        raw = dl.materialize_sample(index, np.sort(np.concatenate(wanted)))
        raw.to_parquet(cache)

    sample = pd.read_parquet(sample_cache_path(int(meta["sample_size"])))
    X_seen, _ = split_features_labels(sample, keep_dst_port=bool(meta.get("keep_dst_port")))
    X_new, y_new = split_features_labels(raw, keep_dst_port=bool(meta.get("keep_dst_port")))
    X_new = X_new[X_seen.columns]
    fresh = ~_row_keys(X_new).isin(_row_keys(X_seen))
    logger.info("Supplementary rows: %d raw, %d after dropping duplicates of sampled rows",
                len(X_new), int(fresh.sum()))
    X_new, y_new = X_new[fresh], y_new[fresh]
    keep = np.concatenate([np.flatnonzero(y_new.values == c)[:cap] for c in need])
    pre = joblib.load(models_dir / "preprocessor.joblib")
    enc = joblib.load(models_dir / "label_encoder.joblib")
    return pre.transform(X_new.iloc[keep]), enc.transform(y_new.iloc[keep].astype(str))


def own_class_confidence(clf, X: pd.DataFrame, y: np.ndarray, classes: list[str]) -> dict:
    """Mean predicted probability of the true class, and recall, per class."""
    proba = clf.predict_proba(X.values)
    out = {}
    for ci, cls in enumerate(classes):
        rows = y == ci
        if rows.any():
            out[cls] = {
                "mean_proba": float(proba[rows, ci].mean()),
                "recall": float((proba[rows].argmax(axis=1) == ci).mean()),
            }
    return out


def aggregate(runs: list[ShapResult], features: list[str], seeds: list[int], per_class: int) -> dict:
    """One dashboard JSON from several seeds' results of one model."""
    first = runs[0]
    overall = [r.importance_overall for r in runs]
    raw_mean = np.mean(overall, axis=0)
    out = {
        # Kept for older readers: raw mean |SHAP| averaged over seeds.
        "overall": [[features[i], float(raw_mean[i])] for i in np.argsort(raw_mean)[::-1][:TOP_K]],
        "per_class": {},
        "overall_share": _ranked(overall, features, _all_top(overall)),
        "per_class_share": {},
        "per_class_n": first.per_class_n,
        "units": first.units,
        "explainer": first.explainer_kind,
        "seeds": seeds,
        "per_class_samples": per_class,
        "stability": {"overall": _stability(overall), "per_class": {}},
    }
    for cls in first.importance_per_class:
        vecs = [r.importance_per_class[cls] for r in runs if cls in r.importance_per_class]
        m = np.mean(vecs, axis=0)
        out["per_class"][cls] = [[features[i], float(m[i])] for i in np.argsort(m)[::-1][:TOP_K]]
        out["per_class_share"][cls] = _ranked(vecs, features, _all_top(vecs))
        out["stability"]["per_class"][cls] = _stability(vecs)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bundle", required=True, help="e.g. 500k_temporal_tuned_fair")
    parser.add_argument("--model", action="append",
                        help="Model to explain; repeatable. Default: all but stacking.")
    parser.add_argument("--per-class", type=int, default=50,
                        help="Rows sampled per class per seed (all rows of a smaller class)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--supplement-cap", type=int, default=200,
                        help="Max unseen raw rows added per short class (0 = none)")
    parser.add_argument("--kernel-nsamples", default="auto",
                        help='KernelExplainer coalitions per row (MLP): "auto" or an int')
    parser.add_argument("--jobs", type=int, default=8, help="Parallel row chunks for trees")
    parser.add_argument("--force", action="store_true",
                        help="Recompute models that already have stability output")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    models_dir, results_dir = MODELS_DIR / args.bundle, OUTPUT_DIR / args.bundle
    out_dir = results_dir / "shap"

    X_test, y_test, classes = rebuild_test_split(args.bundle, results_dir, models_dir)
    features = json.loads((models_dir / "metadata.json").read_text(encoding="utf-8"))["feature_names"]
    X = pd.DataFrame(np.asarray(X_test, dtype=np.float32), columns=features)

    # Attributions on a split that differs from the scored one would explain
    # rows the reported metrics never saw.
    reference = joblib.load(models_dir / "catboost.joblib")
    predicted = np.ravel(reference.predict(X_test))
    _check_against_saved(
        "catboost", confusion_matrix(y_test, predicted, labels=range(len(classes))), results_dir
    )
    logger.info("Rebuilt split matches the stored catboost confusion matrix")

    counts = np.bincount(y_test, minlength=len(classes))
    need = {c: int(n) for c, n in zip(classes, counts) if n < args.per_class}
    y_all = np.asarray(y_test)
    pool = {c: {"test": int(n), "supplementary": 0} for c, n in zip(classes, counts)}
    if need and args.supplement_cap:
        extra = supplementary_rows(args.bundle, models_dir, need, args.supplement_cap)
        if extra is not None:
            X_extra, y_extra = extra
            X = pd.concat([X, pd.DataFrame(np.asarray(X_extra, dtype=np.float32), columns=features)],
                          ignore_index=True)
            y_all = np.concatenate([y_all, y_extra])
            for ci, n in zip(*np.unique(y_extra, return_counts=True)):
                pool[classes[ci]]["supplementary"] = int(n)
    logger.info("Class pools: %s", {c: v for c, v in pool.items() if c in need})
    nsamples = args.kernel_nsamples if args.kernel_nsamples == "auto" else int(args.kernel_nsamples)

    for name in args.model or MODELS:
        target = out_dir / name / "top_features.json"
        if target.is_file() and "stability" in json.loads(target.read_text(encoding="utf-8")) \
                and not args.force:
            logger.info("%s: already has multi-seed output, skipping (use --force)", name)
            continue
        clf = Pipeline([("clf", joblib.load(models_dir / f"{name}.joblib"))])
        runs = []
        for k, seed in enumerate(args.seeds):
            started = time.perf_counter()
            runs.append(analyze_model(
                clf, X, classes, model_name=name, random_state=seed,
                y=y_all, per_class_samples=args.per_class,
                background_samples=200, background_kmeans=20,
                kernel_nsamples=nsamples, kernel_l1_reg=False,
                linear_explainer=True, n_jobs=args.jobs,
                save_dir=out_dir / name, write_plots=k == 0, write_json=False,
            ))
            logger.info("%s seed %d: %.0fs", name, seed, time.perf_counter() - started)
        payload = aggregate(runs, features, args.seeds, args.per_class)
        payload["per_class_pool"] = pool
        payload["per_class_confidence"] = own_class_confidence(clf, X, y_all, classes)
        payload["kernel_nsamples"] = nsamples if payload["explainer"] == "kernel" else None
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("%s: overall stability %s", name, payload["stability"]["overall"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
