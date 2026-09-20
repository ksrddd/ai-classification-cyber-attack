"""Build a full-corpus CSE-CIC-IDS2018 parquet for the cross-dataset protocol.

Why this exists
---------------
The cross-dataset comparison must put 2017 and 2018 through the *same* steps in
the *same order*: clean first, sample second, both starting from the whole
corpus. 2017 already does that -- :mod:`src.crossdataset.loaders` reads
``data/processed/cicids2017_clean.parquet`` (2.4M rows) and the shared
:func:`src.crossdataset.cleaning.clean` is a verified no-op on it.

2018 did not. ``load_ids2018`` read ``data/ids2018/sample_300k.parquet``, which
``src/ids2018/train_ids2018.py`` had already stratified down from ~13M rows
*before* any cleaning, then cleaned again inside the loader -- so 2018 reached
300k by sample->clean while 2017 reached it by clean->sample, and the 2018 side
lost ~37k rows (mostly post-sample duplicates) that skewed its class balance.

This script removes that asymmetry by materialising the entire 2018 corpus,
renamed into the canonical vocabulary and passed through the shared cleaning
rules once, so the cross-dataset loader can then do clean->sample from the full
corpus exactly as it does for 2017.

Pipeline
--------
1. Stream every ``*-2018.csv`` in ``--raw-dir`` in chunks. Per chunk:
     * keep only the 77 canonical-schema source columns plus ``Label`` and
       ``Timestamp`` (``02-20-2018.csv`` ships four extra identifier columns;
       they are simply not selected),
     * drop embedded CSV header lines (``Label == "label"``) and empty labels,
     * coerce feature columns to float32, normalise label spelling.
   Chunks are appended to an intermediate raw parquet via ``ParquetWriter`` so
   peak memory is one chunk.
2. Load the intermediate frame once and apply :func:`clean` (numeric coercion,
   +/-inf -> NaN, drop non-finite rows, drop exact duplicate feature+label
   rows) -- the identical rule set 2017 passed through.
3. Write ``cicids2018_clean.parquet`` and a ``.meta.json`` sidecar recording
   the source fingerprints, the row count entering cleaning (so it can be
   compared against 2017's in ``corpora.json``), the CleaningReport, and the
   per-shared-class counts.

Run::

    python scripts/build_ids2018_full.py --raw-dir data/raw

Memory: step 2 holds the whole ~13M x 77 float32 frame (~4 GB) plus the
transient dedup buffer. Comfortable on 32 GB, tight on 16 GB -- lower
``--chunksize`` only affects step 1.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.crossdataset.cleaning import clean  # noqa: E402
from src.crossdataset.labels import map_to_shared  # noqa: E402
from src.crossdataset.schema import IDS2018_TO_CANONICAL  # noqa: E402
from src.ids2018.config import CHUNKSIZE, EMBEDDED_HEADER_TOKEN  # noqa: E402
from src.ids2018.data_loader import normalise_labels  # noqa: E402

logger = logging.getLogger("build_ids2018_full")

LABEL_COL = "Label"
TIMESTAMP_COL = "Timestamp"

#: Raw 2018 column spellings the canonical schema is built from. These are the
#: keys ``src.crossdataset.schema.to_canonical`` renames, so the parquet must
#: carry every one of them or the loader raises.
FEATURE_COLS: list[str] = list(IDS2018_TO_CANONICAL)
WANTED_COLS: set[str] = {*FEATURE_COLS, LABEL_COL, TIMESTAMP_COL}

DEFAULT_OUT = PROJECT_ROOT / "data" / "ids2018" / "cicids2018_clean.parquet"


# ----------------------------------------------------------------------
def discover_csvs(raw_dir: Path) -> list[Path]:
    """The 2018 daily CSVs in chronological (filename) order.

    Filenames are ``MM-DD-2018.csv`` and February sorts before March, so a
    plain sort is already chronological -- the same order pass 1 and pass 2
    of the sampler rely on.
    """
    files = sorted(raw_dir.glob("*-2018.csv"))
    if not files:
        raise FileNotFoundError(f"No *-2018.csv files in {raw_dir}")
    logger.info("Found %d CSV file(s):", len(files))
    for f in files:
        logger.info("  %-16s %8.1f MB", f.name, f.stat().st_size / 1e6)
    return files


def _fingerprints(files: list[Path]) -> list[str]:
    return [f"{p.name}:{p.stat().st_size}:{int(p.stat().st_mtime)}" for p in files]


def _arrow_schema() -> pa.Schema:
    """float32 for every feature, string for label and timestamp."""
    fields = [(c, pa.float32()) for c in FEATURE_COLS]
    fields += [(LABEL_COL, pa.string()), (TIMESTAMP_COL, pa.string())]
    return pa.schema(fields)


def _prepare_chunk(chunk: pd.DataFrame) -> pd.DataFrame | None:
    """One raw chunk -> canonical-column, header-free, float32 feature frame."""
    missing = WANTED_COLS - set(chunk.columns)
    if missing:
        raise KeyError(f"chunk missing expected column(s): {sorted(missing)}")

    df = chunk.loc[:, [*FEATURE_COLS, LABEL_COL, TIMESTAMP_COL]].copy()

    label = df[LABEL_COL].astype("string").str.strip()
    keep = label.notna() & (label.str.lower() != EMBEDDED_HEADER_TOKEN) & (label != "")
    df = df.loc[keep]
    if df.empty:
        return None

    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    df[LABEL_COL] = normalise_labels(df[LABEL_COL]).astype("string")
    df[TIMESTAMP_COL] = df[TIMESTAMP_COL].astype("string")
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------
def stage1_stream_raw(files: list[Path], raw_out: Path, chunksize: int) -> dict[str, int]:
    """Stream the CSVs into one uncleaned parquet. Returns rows kept per file."""
    logger.info("Stage 1/2: streaming raw CSVs -> %s", raw_out)
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    schema = _arrow_schema()
    per_file: dict[str, int] = {}

    with pq.ParquetWriter(raw_out, schema, compression="zstd") as writer:
        for path in files:
            kept = 0
            reader = pd.read_csv(
                path,
                dtype=str,
                usecols=lambda c: c in WANTED_COLS,
                chunksize=chunksize,
                engine="c",
                skip_blank_lines=False,
                on_bad_lines="skip",
            )
            for chunk in reader:
                prepared = _prepare_chunk(chunk)
                if prepared is None:
                    continue
                writer.write_table(pa.Table.from_pandas(prepared, schema=schema, preserve_index=False))
                kept += len(prepared)
            per_file[path.name] = kept
            logger.info("  %-16s %10s rows kept", path.name, f"{kept:,}")

    total = sum(per_file.values())
    logger.info("Stage 1 complete: %s rows across %d file(s)", f"{total:,}", len(files))
    return per_file


def stage2_clean(raw_out: Path, final_out: Path) -> tuple[int, dict]:
    """Apply the shared cleaning rules once. Returns (rows_in, CleaningReport dict)."""
    logger.info("Stage 2/2: loading %s and applying shared clean()", raw_out)
    df = pd.read_parquet(raw_out)
    rows_in = len(df)

    cleaned, report = clean(df, FEATURE_COLS, LABEL_COL)
    del df

    final_out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(final_out, index=False, compression="zstd")
    logger.info(
        "Stage 2 complete: %s -> %s rows (%.2f%% kept) -> %s",
        f"{rows_in:,}",
        f"{len(cleaned):,}",
        100 * report.kept_fraction,
        final_out,
    )

    shared_counts = (
        map_to_shared(cleaned[LABEL_COL], "ids2018")
        .value_counts()
        .sort_values(ascending=False)
        .to_dict()
    )
    return rows_in, {"report": report.as_dict(), "shared_class_counts": {k: int(v) for k, v in shared_counts.items()}}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return None


# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw",
                    help="directory holding the *-2018.csv daily captures")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="destination for the cleaned full-corpus parquet")
    ap.add_argument("--chunksize", type=int, default=CHUNKSIZE,
                    help="rows per read chunk in stage 1")
    ap.add_argument("--keep-raw", action="store_true",
                    help="keep the uncleaned intermediate parquet instead of deleting it")
    ap.add_argument("--stage1-only", action="store_true",
                    help="stream the raw parquet and stop before cleaning")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    started = time.perf_counter()
    files = discover_csvs(args.raw_dir)
    raw_out = args.out.with_suffix(".raw.parquet")

    per_file = stage1_stream_raw(files, raw_out, args.chunksize)

    if args.stage1_only:
        logger.info("--stage1-only set: wrote %s, stopping", raw_out)
        return 0

    rows_in, clean_info = stage2_clean(raw_out, args.out)

    meta = {
        "dataset": "CSE-CIC-IDS2018",
        "purpose": "cross-dataset protocol: full corpus, canonical schema, shared cleaning",
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "source_dir": str(args.raw_dir),
        "source_fingerprints": _fingerprints(files),
        "rows_kept_per_file": per_file,
        # The row count entering clean(). Compare against ids2017 in
        # results/crossdataset/<run>/corpora.json: both must read as the whole
        # corpus, not a pre-sample, for the comparison to be under one protocol.
        "rows_entering_clean": rows_in,
        "n_feature_columns": len(FEATURE_COLS),
        **clean_info,
    }
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", meta_path)

    if not args.keep_raw:
        raw_out.unlink(missing_ok=True)
        logger.info("Removed intermediate %s (pass --keep-raw to retain it)", raw_out)

    logger.info("Done in %.1f min", (time.perf_counter() - started) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
