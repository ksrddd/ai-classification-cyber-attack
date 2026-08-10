"""Backfill per-class split sizes into older CSE-CIC-IDS2018 bundles.

Run::

    python -m scripts.backfill_ids2018_counts --bundle 300k 500k
    python -m scripts.backfill_ids2018_counts --all --dry-run

Why this exists
---------------
``train_ids2018`` now records ``per_class_n_train`` / ``per_class_n_test`` in
``metadata.json``, but bundles trained before that carry neither. The temporal
ones are unaffected -- their ``split_manifest.json`` always stated both sides --
so in practice this is about the random stratified bundles, whose train counts
existed nowhere on disk and rendered as absent in the dashboard.

Recovering rather than guessing
-------------------------------
Nothing here is inferred from the 70/30 ratio. The random split is deterministic
given the sample and the seed, both of which the bundle records, so the original
split is *rebuilt* and its classes counted -- the same technique
``src/ids2018/analyze.py`` uses to recover per-row predictions.

A rebuild that is merely plausible would be worse than no backfill at all: the
numbers would look measured and sit in the same table as the ones that are. So
the rebuilt *test* side is compared class-for-class against the ``support``
column of a per-class report written at training time, and a single mismatch
aborts that bundle without touching its metadata. The train side is only trusted
because the test side beside it reproduced exactly.

What it will not do
-------------------
It never writes ``split_manifest.json``. Presence of that file is how
``analyze.py`` decides a bundle is temporal; dropping one into a random bundle
would silently reroute every later analysis through the wrong splitter.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ids2018.config import MODELS_DIR, OUTPUT_DIR, sample_cache_path  # noqa: E402
from src.ids2018.preprocessing import (  # noqa: E402
    split_features_labels,
    stratified_train_test_split,
)

logger = logging.getLogger("ids2018.backfill")


class BackfillError(RuntimeError):
    """Raised when a bundle cannot be backfilled safely."""


def _load_metadata(bundle: str) -> tuple[Path, dict]:
    path = MODELS_DIR / bundle / "metadata.json"
    if not path.is_file():
        raise BackfillError(f"no metadata.json at {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _recorded_test_supports(results_dir: Path) -> dict[str, int]:
    """Per-class test support as training recorded it, from any model."""
    for report in sorted(results_dir.glob("*/per_class_report.csv")):
        df = pd.read_csv(report)
        label_col = df.columns[0]
        skip = {"accuracy", "macro avg", "weighted avg"}
        return {
            str(row[label_col]): int(row["support"])
            for _, row in df.iterrows()
            if str(row[label_col]) not in skip
        }
    raise BackfillError(f"no per_class_report.csv under {results_dir}")


def rebuild_counts(bundle: str, meta: dict) -> tuple[dict[str, int], dict[str, int]]:
    """Rebuild the bundle's split and count both sides per class."""
    if (OUTPUT_DIR / bundle / "split_manifest.json").is_file():
        raise BackfillError(
            "bundle is temporal and already states both sides in split_manifest.json"
        )

    sample_size = meta.get("sample_size")
    if not sample_size:
        raise BackfillError("metadata records no sample_size")

    cache = sample_cache_path(int(sample_size))
    if not cache.is_file():
        raise BackfillError(
            f"sample cache {cache} is absent -- rebuild it with "
            f"`--sample-size {sample_size} --dry-run` before backfilling"
        )

    sample = pd.read_parquet(cache)
    _, y_series = split_features_labels(
        sample, keep_dst_port=bool(meta.get("keep_dst_port")), keep_timestamp=False
    )
    # The splitter is stratified on y alone, so a single column stands in for
    # the feature matrix and the 77-column frame never has to be materialised.
    placeholder = pd.DataFrame({"_": range(len(y_series))}, index=y_series.index)
    _, _, y_train, y_test = stratified_train_test_split(
        placeholder,
        y_series,
        test_size=float(meta["test_size"]),
        random_state=int(meta["random_state"]),
    )
    to_counts = lambda s: {str(k): int(v) for k, v in s.value_counts().items()}  # noqa: E731
    return to_counts(y_train), to_counts(y_test)


def backfill(bundle: str, *, dry_run: bool = False) -> bool:
    """Return True if the bundle was updated (or would be, under --dry-run)."""
    meta_path, meta = _load_metadata(bundle)

    if meta.get("per_class_n_train"):
        logger.info("%-26s already records per-class counts -- skipped", bundle)
        return False

    train_counts, test_counts = rebuild_counts(bundle, meta)
    recorded = _recorded_test_supports(OUTPUT_DIR / bundle)

    if test_counts != recorded:
        only_rebuilt = {k: v for k, v in test_counts.items() if recorded.get(k) != v}
        raise BackfillError(
            f"rebuilt test split does not match the one training scored: {only_rebuilt}. "
            "The train counts derived from it would be wrong in a way no reader "
            "could detect, so nothing was written."
        )

    logger.info(
        "%-26s verified: %d classes, train %s / test %s",
        bundle,
        len(train_counts),
        f"{sum(train_counts.values()):,}",
        f"{sum(test_counts.values()):,}",
    )
    if dry_run:
        return True

    meta["per_class_n_train"] = train_counts
    meta["per_class_n_test"] = test_counts
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("%-26s written to %s", "", meta_path)
    return True


def _discover() -> list[str]:
    if not OUTPUT_DIR.is_dir():
        return []
    return [
        p.name
        for p in sorted(OUTPUT_DIR.iterdir())
        if p.is_dir() and not (p / "split_manifest.json").is_file()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="backfill_ids2018_counts",
        description="Recover per-class split sizes for pre-existing 2018 bundles",
    )
    parser.add_argument("--bundle", nargs="+", help="bundle names, e.g. 300k 500k")
    parser.add_argument(
        "--all", action="store_true", help="every non-temporal bundle under results/ids2018"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="verify and report without writing metadata"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)-7s | %(message)s", stream=sys.stdout
    )

    bundles = args.bundle or (_discover() if args.all else None)
    if not bundles:
        parser.error("pass --bundle NAME [NAME ...] or --all")

    failures = 0
    for bundle in bundles:
        try:
            backfill(bundle, dry_run=args.dry_run)
        except BackfillError as exc:
            logger.error("%-26s %s", bundle, exc)
            failures += 1

    if args.dry_run:
        logger.info("dry run -- no metadata was modified")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
