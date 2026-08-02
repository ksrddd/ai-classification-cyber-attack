"""End-to-end training driver for CSE-CIC-IDS2018.

Run::

    python -m src.ids2018.train_ids2018 --raw-dir D:/CSE-CIC-IDS2018

Stages, in order:

    1. Two-pass stratified sample of 300,000 rows from the ~13M-row corpus
       (cached to Parquet, so only the first run pays the scan cost).
    2. Drop leakage/identifier columns, split off the target.
    3. Stratified 70/30 train/test split -> 210,000 / 90,000 rows.
    4. Fit the preprocessor on train only; transform both splits.
    5. Train and evaluate all seven models.
    6. Write per-model artefacts and one comparison table.

``--dry-run`` executes stages 1-4 and stops before any model is trained,
which is the quickest way to verify the data path on a new machine.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

from src.ids2018.config import (
    CACHE_DIR,
    CHUNKSIZE,
    CLASS_WEIGHTING,
    INDEX_CACHE,
    MIN_PER_CLASS,
    MODEL_NAMES,
    MODELS_DIR,
    OUTPUT_DIR,
    RANDOM_STATE,
    RAW_DIR,
    SAMPLE_SIZE,
    SAMPLING_MODE,
    TEST_SIZE,
    sample_cache_path,
    size_tag,
)
from src.ids2018.data_loader import (
    load_index,
    materialize_sample,
    plan_sample,
    save_index,
    scan_labels,
)
from src.ids2018.evaluate import (
    build_comparison_table,
    confusion_frame,
    evaluate_model,
    per_class_report,
    save_comparison_table,
    save_model_artifacts,
)
from src.ids2018.models import build_model, fit_model, save_model, set_class_weighting
from src.ids2018.preprocessing import (
    Ids2018Preprocessor,
    encode_labels,
    split_features_labels,
    stratified_train_test_split,
)

logger = logging.getLogger("ids2018")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train_ids2018",
        description="Stratified sampling + 7-model benchmark on CSE-CIC-IDS2018",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="directory of the daily CSVs")
    data.add_argument("--sample-size", type=int, default=SAMPLE_SIZE, help="total rows to sample")
    data.add_argument("--test-size", type=float, default=TEST_SIZE, help="test fraction")
    data.add_argument("--chunksize", type=int, default=CHUNKSIZE, help="rows read per chunk")
    data.add_argument(
        "--min-per-class",
        type=int,
        default=MIN_PER_CLASS,
        help="floor on sampled rows per class; 0 = strictly proportional",
    )
    data.add_argument(
        "--sampling",
        choices=["nested", "independent"],
        default=SAMPLING_MODE,
        help=(
            "'nested' makes smaller samples exact subsets of larger ones "
            "(use for a 300k/500k/1M ladder); 'independent' draws each size "
            "separately with StratifiedShuffleSplit"
        ),
    )
    data.add_argument(
        "--keep-dst-port", action="store_true", help="keep 'Dst Port' as a feature"
    )
    data.add_argument(
        "--drop-duplicates",
        action="store_true",
        help="drop exact duplicate flows after sampling (changes the final row count)",
    )
    data.add_argument(
        "--rebuild-sample",
        action="store_true",
        help="ignore the Parquet cache and re-extract the sample",
    )
    data.add_argument(
        "--sample-cache",
        type=Path,
        default=None,
        help="Parquet cache path (default: data/ids2018/sample_<size>.parquet)",
    )
    data.add_argument(
        "--index-cache",
        type=Path,
        default=INDEX_CACHE,
        help="cached pass-1 label index, shared by every sample size",
    )

    train = parser.add_argument_group("training")
    train.add_argument(
        "--models",
        nargs="+",
        default=MODEL_NAMES,
        choices=MODEL_NAMES,
        metavar="NAME",
        help=f"subset of models to train ({', '.join(MODEL_NAMES)})",
    )
    train.add_argument(
        "--scaler", choices=["standard", "robust"], default="standard", help="feature scaler"
    )
    train.add_argument(
        "--class-weighting",
        choices=["none", "balanced"],
        default=CLASS_WEIGHTING,
        help="'none' gives all seven models identical treatment (the only fair "
        "comparison, since MLPClassifier cannot be weighted at all); 'balanced' "
        "raises macro recall on rare attacks but leaves MLP incomparable",
    )
    train.add_argument(
        "--accelerator",
        choices=["cpu", "gpu"],
        default="cpu",
        help="'gpu' moves XGBoost, CatBoost and the stacking ensemble's XGBoost "
        "base learner onto CUDA; every other model is CPU-only",
    )
    train.add_argument(
        "--gpu-devices", default="0", help="CUDA device ids, e.g. '0' or '0,1'"
    )
    train.add_argument("--seed", type=int, default=RANDOM_STATE, help="random seed")
    train.add_argument(
        "--dry-run", action="store_true", help="prepare the data then exit without training"
    )

    out = parser.add_argument_group("output")
    out.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="metrics + plots (default: results/ids2018/<size>)",
    )
    out.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="serialised models (default: models/ids2018/<size>)",
    )

    args = parser.parse_args(argv)

    # Every default path is namespaced by sample size, so a 500k run never
    # overwrites the 300k results sitting next to it.
    tag = size_tag(args.sample_size)
    if args.sample_cache is None:
        args.sample_cache = sample_cache_path(args.sample_size)
    if args.output_dir is None:
        args.output_dir = OUTPUT_DIR / tag
    if args.models_dir is None:
        args.models_dir = MODELS_DIR / tag
    return args


def setup_logging(output_dir: Path) -> None:
    """Log to stdout and to a run log inside the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "train_ids2018.log", encoding="utf-8"),
        ],
    )


# ----------------------------------------------------------------------
# Stage 1 -- sampling
# ----------------------------------------------------------------------
def discover_csvs(raw_dir: Path) -> list[Path]:
    """Return the daily CSVs in deterministic (chronological) order.

    Ordering matters: the row positions produced by pass 1 are only valid
    for pass 2 if both passes walk the files in the same sequence.
    """
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    logger.info("Found %d CSV file(s) in %s", len(files), raw_dir)
    return files


def load_or_build_sample(args: argparse.Namespace) -> pd.DataFrame:
    """Return the 300k stratified sample, reusing the Parquet cache if valid."""
    cache = args.sample_cache

    if cache.exists() and not args.rebuild_sample:
        sample = pd.read_parquet(cache)
        if len(sample) == args.sample_size:
            logger.info("Reusing cached sample: %s (%s rows)", cache, f"{len(sample):,}")
            return sample
        logger.warning(
            "Cached sample has %s rows but %s were requested -- rebuilding",
            f"{len(sample):,}",
            f"{args.sample_size:,}",
        )

    files = discover_csvs(args.raw_dir)

    # Pass 1 is size-independent, so the 300k / 500k / 1M runs share one
    # cached label index and only the first of them pays the CSV scan.
    index = load_index(args.index_cache, files)
    if index is None:
        index = scan_labels(files, chunksize=args.chunksize)
        save_index(index, args.index_cache)

    selected = plan_sample(
        index,
        n_samples=args.sample_size,
        mode=args.sampling,
        min_per_class=args.min_per_class,
        random_state=args.seed,
    )
    sample = materialize_sample(index, selected, chunksize=args.chunksize)

    # The label index for the whole corpus is no longer needed and is the
    # single largest object still alive at this point.
    del index, selected
    gc.collect()

    cache.parent.mkdir(parents=True, exist_ok=True)
    # Parquet keeps dtypes and is ~10x smaller than CSV for this data.
    sample.to_parquet(cache, index=False)
    logger.info("Sample cached to %s", cache)
    return sample


# ----------------------------------------------------------------------
# Stage 2-4 -- data preparation
# ----------------------------------------------------------------------
def prepare_data(sample: pd.DataFrame, args: argparse.Namespace):
    """Clean, split and scale. Returns arrays ready for ``fit``."""
    if args.drop_duplicates:
        before = len(sample)
        sample = sample.drop_duplicates().reset_index(drop=True)
        logger.info("Dropped %s duplicate flow(s)", f"{before - len(sample):,}")

    X_df, y_series = split_features_labels(sample, keep_dst_port=args.keep_dst_port)
    logger.info("Feature matrix before cleaning: %s x %d", f"{len(X_df):,}", X_df.shape[1])

    X_train_df, X_test_df, y_train_s, y_test_s = stratified_train_test_split(
        X_df, y_series, test_size=args.test_size, random_state=args.seed
    )
    del X_df, y_series
    gc.collect()

    # Fit on train only -- medians, constant-column detection and scaler
    # statistics must never see the test split.
    pre = Ids2018Preprocessor(scaler_kind=args.scaler)
    X_train = pre.fit_transform(X_train_df)
    X_test = pre.transform(X_test_df)
    del X_train_df, X_test_df
    gc.collect()

    y_train, y_test, encoder = encode_labels(y_train_s, y_test_s)

    logger.info(
        "Ready: X_train %s, X_test %s (float32, %.1f MB total)",
        X_train.shape,
        X_test.shape,
        (X_train.nbytes + X_test.nbytes) / 1e6,
    )
    return X_train, X_test, y_train, y_test, pre, encoder


def save_preprocessing_artifacts(
    pre: Ids2018Preprocessor, encoder, args: argparse.Namespace
) -> None:
    """Persist everything needed to score new traffic with these models."""
    args.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, args.models_dir / "preprocessor.joblib", compress=3)
    joblib.dump(encoder, args.models_dir / "label_encoder.joblib", compress=3)

    metadata = {
        "dataset": "CSE-CIC-IDS2018",
        "raw_dir": str(args.raw_dir),
        "sample_size": args.sample_size,
        "sampling_mode": args.sampling,
        "min_per_class": args.min_per_class,
        "test_size": args.test_size,
        "random_state": args.seed,
        "scaler": args.scaler,
        "class_weighting": args.class_weighting,
        "accelerator": args.accelerator,
        "gpu_devices": args.gpu_devices if args.accelerator == "gpu" else None,
        "keep_dst_port": args.keep_dst_port,
        "n_features": len(pre.feature_names),
        "feature_names": pre.feature_names,
        "dropped_columns": pre.dropped_columns,
        "classes": list(encoder.classes_),
    }
    (args.models_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Preprocessor, encoder and metadata saved to %s", args.models_dir)


# ----------------------------------------------------------------------
# Stage 5 -- training loop
# ----------------------------------------------------------------------
def verify_accelerator(args: argparse.Namespace) -> None:
    """Prove the CUDA path works before spending an hour on it.

    A CUDA build that imports but cannot run -- wrong driver, no visible
    device -- would otherwise fall back to the CPU silently, and every
    timing in the comparison table would be mislabelled ``gpu``. Reuses the
    project's existing hardware probe: it is dataset-agnostic, so sharing it
    does not compromise this package's isolation from the 2017 pipeline.
    """
    if args.accelerator != "gpu":
        return

    from src.training.gpu import run_gpu_acceptance

    logger.info("Verifying CUDA on device(s) %s ...", args.gpu_devices)
    result = run_gpu_acceptance(args.gpu_devices)
    for check, outcome in result["checks"].items():
        logger.info("  %-18s %s | %s", check, "PASS" if outcome["passed"] else "FAIL", outcome["detail"])

    if not result["passed"]:
        raise RuntimeError(
            "GPU acceptance failed -- refusing to run with --accelerator gpu and report "
            "CPU timings as GPU. Fix the CUDA setup or drop the flag."
        )


def run_training(X_train, X_test, y_train, y_test, class_names, args) -> list[dict]:
    """Train, evaluate and persist each requested model."""
    set_class_weighting(args.class_weighting)
    all_metrics: list[dict] = []

    for i, name in enumerate(args.models, start=1):
        logger.info("-" * 70)
        logger.info("[%d/%d] %s", i, len(args.models), name)

        model = None
        try:
            model = build_model(name, accelerator=args.accelerator, gpu_devices=args.gpu_devices)

            t0 = time.perf_counter()
            model = fit_model(name, model, X_train, y_train)
            train_seconds = time.perf_counter() - t0

            t0 = time.perf_counter()
            y_pred = model.predict(X_test)
            predict_seconds = time.perf_counter() - t0
            # CatBoost returns an (n, 1) column of class indices.
            y_pred = y_pred.ravel().astype(int)

            metrics = evaluate_model(
                name,
                y_test,
                y_pred,
                train_seconds=train_seconds,
                predict_seconds=predict_seconds,
            )
            save_model_artifacts(
                name,
                metrics,
                confusion_frame(y_test, y_pred, class_names),
                per_class_report(y_test, y_pred, class_names),
                args.output_dir,
            )
            save_model(name, model, args.models_dir)
            all_metrics.append(metrics)

        except Exception:
            # One failing model should not discard the other six results.
            logger.exception("%s failed -- continuing with the remaining models", name)
        finally:
            # Ensembles hold several fitted trees; release before the next run.
            model = None
            gc.collect()

    return all_metrics


# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.output_dir)

    logger.info("=" * 70)
    logger.info("CSE-CIC-IDS2018 — stratified sampling + 7-model benchmark")
    logger.info(
        "sample=%s rows | sampling=%s | seed=%d | split=%.0f/%.0f | weighting=%s | accelerator=%s",
        f"{args.sample_size:,}",
        args.sampling,
        args.seed,
        100 * (1 - args.test_size),
        100 * args.test_size,
        args.class_weighting,
        args.accelerator,
    )
    logger.info("results -> %s | models -> %s", args.output_dir, args.models_dir)
    logger.info("=" * 70)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Checked up front: failing on CUDA after a 10-minute sampling pass
    # wastes the run. Skipped for --dry-run, which never trains anything.
    if not args.dry_run:
        verify_accelerator(args)

    started = time.perf_counter()
    sample = load_or_build_sample(args)
    X_train, X_test, y_train, y_test, pre, encoder = prepare_data(sample, args)
    del sample
    gc.collect()

    save_preprocessing_artifacts(pre, encoder, args)

    if args.dry_run:
        logger.info("--dry-run set: data prepared, stopping before training")
        return 0

    class_names = [str(c) for c in encoder.classes_]
    all_metrics = run_training(X_train, X_test, y_train, y_test, class_names, args)

    if not all_metrics:
        logger.error("Every model failed -- no comparison table to write")
        return 1

    logger.info("=" * 70)
    save_comparison_table(build_comparison_table(all_metrics, args.output_dir), args.output_dir)
    logger.info("Total wall time: %.1f min", (time.perf_counter() - started) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
