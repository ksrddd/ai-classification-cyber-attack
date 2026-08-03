"""Post-training analysis for a CSE-CIC-IDS2018 bundle.

Run::

    python -m src.ids2018.analyze --bundle 300k_temporal_tuned

Why this exists
---------------
``train_ids2018`` writes per-model ``metrics.json``, a confusion matrix and a
per-class report. The dashboard reads four more files that nothing in the
repository produced -- the ones under ``results/ids2018/300k`` were generated
ad hoc and committed, so a newly trained bundle had no way to acquire them and
rendered every operational number as absent:

    extended_metrics.csv          mcc, false-alarm rate, missed attacks, cost
    significance_f1_macro_ci.csv  bootstrap CI on macro F1
    significance_mcnemar.csv      leader vs each rival, paired
    per_class_f1_matrix.csv       per-class F1 across models, with support

False-alarm rate and missed attacks are the two numbers PRODUCT.md names as the
operationally meaningful ones, so a bundle without them cannot take part in the
comparison this project is built around.

Rebuilding the predictions
--------------------------
Confidence intervals and McNemar both need *per-row* predictions, which training
does not persist -- only aggregates. They are recovered rather than approximated:
the temporal split is deterministic and RNG-free, the preprocessor and label
encoder are saved next to the models, so the exact test split can be
reconstructed and re-scored.

That reconstruction is verified, not assumed. :func:`_check_against_saved` re-derives
each model's confusion matrix and compares it cell-for-cell with the one training
wrote. If the split were rebuilt even slightly differently every downstream number
here would be wrong in a way no reader could detect, so a mismatch aborts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef

from src.ids2018.config import MODELS_DIR, OUTPUT_DIR, RANDOM_STATE, TIMESTAMP_COL, sample_cache_path
from src.ids2018.preprocessing import split_features_labels, stratified_train_test_split
from src.ids2018.temporal_split import temporal_train_test_split

logger = logging.getLogger("ids2018.analyze")

BENIGN = "Benign"
DEFAULT_BOOTSTRAP = 1000


# ----------------------------------------------------------------------
# Split reconstruction
# ----------------------------------------------------------------------
def rebuild_test_split(bundle: str, results_dir: Path, models_dir: Path):
    """Reproduce the exact (X_test, y_test) the bundle was scored on."""
    meta = json.loads((models_dir / "metadata.json").read_text(encoding="utf-8"))
    sample = pd.read_parquet(sample_cache_path(int(meta["sample_size"])))

    manifest_path = results_dir / "split_manifest.json"
    temporal = manifest_path.is_file()

    X_df, y_series = split_features_labels(
        sample,
        keep_dst_port=bool(meta.get("keep_dst_port")),
        keep_timestamp=temporal,
    )

    if temporal:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _, X_test_df, _, y_test_s = temporal_train_test_split(
            X_df,
            y_series,
            X_df[TIMESTAMP_COL],
            test_size=float(manifest["test_size"]),
            min_test_per_class=int(manifest["min_test_per_class"]),
        )
        X_test_df = X_test_df.drop(columns=[TIMESTAMP_COL])
    else:
        _, X_test_df, _, y_test_s = stratified_train_test_split(
            X_df,
            y_series,
            test_size=float(meta["test_size"]),
            random_state=int(meta["random_state"]),
        )

    pre = joblib.load(models_dir / "preprocessor.joblib")
    encoder = joblib.load(models_dir / "label_encoder.joblib")
    X_test = pre.transform(X_test_df)
    y_test = encoder.transform(y_test_s.astype(str))
    class_names = [str(c) for c in encoder.classes_]

    logger.info("Rebuilt test split: %s rows, %d classes", f"{len(y_test):,}", len(class_names))
    return X_test, y_test, class_names


def _check_against_saved(name: str, cm_new: np.ndarray, results_dir: Path) -> None:
    """Abort unless the re-derived confusion matrix matches the stored one."""
    stored_path = results_dir / name / "confusion_matrix.csv"
    if not stored_path.is_file():
        raise FileNotFoundError(f"{stored_path} is missing; cannot verify the rebuild")
    stored = pd.read_csv(stored_path, index_col=0).to_numpy()
    if stored.shape != cm_new.shape or not np.array_equal(stored, cm_new):
        raise RuntimeError(
            f"{name}: re-derived confusion matrix does not match the one training "
            "wrote. The test split was not reconstructed identically, so every "
            "confidence interval and significance test below would be wrong."
        )


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def binary_view(cm: np.ndarray, benign_index: int) -> dict[str, float]:
    """Collapse the multiclass matrix to Attack-vs-Benign.

    This is the SOC-facing view: a false positive is benign traffic that raised
    an alarm, a false negative is an attack that was let through. Which specific
    attack class a flow was misassigned to does not matter here -- both stay
    inside the Attack side and are not counted as errors by this view.
    """
    total = cm.sum()
    tn = float(cm[benign_index, benign_index])
    fp = float(cm[benign_index].sum() - tn)
    fn = float(cm[:, benign_index].sum() - tn)
    tp = float(total - tn - fp - fn)
    return {
        "binary_fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
        "binary_recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "false_alarms_fp": int(fp),
        "missed_attacks_fn": int(fn),
    }


def bootstrap_f1(
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Bootstrap macro-F1 for every model over one shared set of resamples.

    Sharing the resample indices across models is what makes the leader-minus-
    rival differences paired: with independent draws per model the difference of
    two CIs would mix sampling noise from two different resamplings and the
    interval on the gap would be far too wide.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    out = {name: np.empty(n_boot) for name in preds}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        truth = y_true[idx]
        for name, pred in preds.items():
            out[name][b] = f1_score(truth, pred[idx], average="macro", zero_division=0)
    return out


# ----------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------
def write_extended_metrics(
    rows: list[dict], results_dir: Path
) -> pd.DataFrame:
    frame = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    frame.to_csv(results_dir / "extended_metrics.csv", index=False, encoding="utf-8")
    logger.info("Wrote %s", results_dir / "extended_metrics.csv")
    return frame


def write_ci(
    boot: dict[str, np.ndarray], point: dict[str, dict], results_dir: Path
) -> pd.DataFrame:
    rows = []
    for name, samples in boot.items():
        lo, hi = np.percentile(samples, [2.5, 97.5])
        rows.append({
            "model": name,
            "f1_macro": point[name]["f1_macro"],
            "acc": point[name]["accuracy"],
            "ci_lo": float(lo),
            "ci_hi": float(hi),
        })
    frame = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    frame.to_csv(results_dir / "significance_f1_macro_ci.csv", index=False, encoding="utf-8")
    logger.info("Wrote %s", results_dir / "significance_f1_macro_ci.csv")
    return frame


def write_mcnemar(
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    boot: dict[str, np.ndarray],
    point: dict[str, dict],
    results_dir: Path,
) -> pd.DataFrame:
    """Leader against every rival, on the same rows.

    McNemar is the right test here rather than a two-sample comparison: both
    models were evaluated on the identical test flows, so the pairing carries
    real information and only the flows they disagree on are evidence.
    ``binomtest`` gives the exact test, which matters because the discordant
    counts are small -- a chi-square approximation is unreliable below ~25.
    """
    leader = max(point, key=lambda m: point[m]["f1_macro"])
    correct = {name: (pred == y_true) for name, pred in preds.items()}

    rows = []
    for name in preds:
        if name == leader:
            continue
        b = int(np.sum(correct[leader] & ~correct[name]))
        c = int(np.sum(~correct[leader] & correct[name]))
        p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else 1.0
        diff = boot[leader] - boot[name]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        rows.append({
            "rival": name,
            "leader_only_right": b,
            "rival_only_right": c,
            "disagree": b + c,
            "mcnemar_p": float(p),
            "d_f1": point[leader]["f1_macro"] - point[name]["f1_macro"],
            "d_ci_lo": float(lo),
            "d_ci_hi": float(hi),
        })

    frame = pd.DataFrame(rows).sort_values("d_f1")
    frame.to_csv(results_dir / "significance_mcnemar.csv", index=False, encoding="utf-8")
    logger.info("Wrote %s (leader: %s)", results_dir / "significance_mcnemar.csv", leader)
    return frame


def write_per_class_matrix(
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    class_names: list[str],
    results_dir: Path,
) -> pd.DataFrame:
    """Per-class F1 for every model, with the support that produced it.

    Support is a column, not a footnote: three of these classes have under five
    test flows, and an F1 of 1.000 on one flow has to arrive with the 1 attached.
    """
    data: dict[str, list[float]] = {}
    for name, pred in preds.items():
        data[name] = f1_score(
            y_true, pred, average=None,
            labels=np.arange(len(class_names)), zero_division=0,
        ).tolist()

    support = np.bincount(y_true, minlength=len(class_names))
    frame = pd.DataFrame(data, index=class_names)
    frame = frame[sorted(frame.columns)]
    frame.insert(len(frame.columns), "support", support)
    frame.index.name = "class"
    frame.to_csv(results_dir / "per_class_f1_matrix.csv", encoding="utf-8")
    logger.info("Wrote %s", results_dir / "per_class_f1_matrix.csv")
    return frame


# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="analyze",
        description="Derive the dashboard's analysis files for a trained 2018 bundle",
    )
    parser.add_argument("--bundle", required=True, help="e.g. 300k_temporal_tuned")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    results_dir = OUTPUT_DIR / args.bundle
    models_dir = MODELS_DIR / args.bundle
    if not results_dir.is_dir():
        raise SystemExit(f"No such bundle: {results_dir}")

    X_test, y_test, class_names = rebuild_test_split(args.bundle, results_dir, models_dir)
    benign_index = class_names.index(BENIGN)

    preds: dict[str, np.ndarray] = {}
    point: dict[str, dict] = {}
    rows: list[dict] = []

    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        name = model_dir.name
        artifact = models_dir / f"{name}.joblib"
        if not (model_dir / "metrics.json").is_file() or not artifact.is_file():
            logger.warning("Skipping %s: missing metrics.json or %s", name, artifact.name)
            continue

        metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
        model = joblib.load(artifact)
        y_pred = np.asarray(model.predict(X_test)).ravel().astype(int)
        del model

        cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))
        _check_against_saved(name, cm, results_dir)

        preds[name] = y_pred
        point[name] = metrics

        predict_s = float(metrics.get("predict_seconds") or 0.0)
        rows.append({
            "model": name,
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "recall_macro": metrics["recall_macro"],
            "precision_macro": metrics["precision_macro"],
            "bal_acc": metrics["balanced_accuracy"],
            "mcc": float(matthews_corrcoef(y_test, y_pred)),
            **binary_view(cm, benign_index),
            "train_s": metrics.get("train_seconds"),
            "predict_s": predict_s,
            "flows_per_sec": (len(y_test) / predict_s) if predict_s else None,
            "artifact_mb": artifact.stat().st_size / 1e6,
        })
        logger.info("  %-20s verified against saved confusion matrix", name)

    if not preds:
        raise SystemExit("No models could be loaded; nothing to analyse")

    logger.info("Bootstrapping macro F1 (%d resamples, shared across models)...", args.bootstrap)
    boot = bootstrap_f1(y_test, preds, n_boot=args.bootstrap, seed=args.seed)

    write_extended_metrics(rows, results_dir)
    write_ci(boot, point, results_dir)
    write_mcnemar(y_test, preds, boot, point, results_dir)
    write_per_class_matrix(y_test, preds, class_names, results_dir)
    logger.info("Analysis complete for %s", args.bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
