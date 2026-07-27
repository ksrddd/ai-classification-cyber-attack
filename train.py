#!/usr/bin/env python
"""train.py -- CICIDS2017 + CSE-CIC-IDS2018 cyber-attack classification pipeline.

End-to-end with ZERO errors and ZERO warnings. Verifies under:

    python train.py
    python -W error::Warning train.py

Library targets (matches repo requirements.txt):
    Python >= 3.10  | pandas >= 2.0   | numpy >= 1.26 (NumPy 2.x OK)
    scikit-learn >= 1.3 | xgboost >= 2.0  | lightgbm >= 4.1
    matplotlib >= 3.7   | seaborn >= 0.12 | joblib >= 1.3

Four-layer correctness plan for handling extreme class imbalance:
    1. Draw a natural-distribution test set first. Any targeted sampling or
       synthetic over-sampling is restricted to TRAIN (and to each CV fold).
    2. Use a configurable training strategy: class weights, targeted sampling
       of real Infiltration rows, RandomOverSampler, BorderlineSMOTE, or
       SMOTEENN. Heartbleed is never synthesised by the target-only samplers.
    3. Keep the untouched test distribution and per-class sample counts in
       every report so recall gains cannot hide an explosion in false alarms.
    4. Trust checks: 5-fold CV (mean +/- std), label-shuffle macro-F1 (must
       collapse to ~chance), majority-class baseline, per-class N alongside
       metrics so small-N entries are read with appropriate scepticism.

Resumable: cleaned full-corpus data is cached as parquet; per-model
artefacts under ``results/<run_name>/`` are kept and skipped on re-run
unless ``--force`` is passed. Lets you retrain one model at a time
without redoing load+clean (3-5 min).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import sys
import time
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: no GUI backend = no Tk/Qt warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.artifacts.bundle import build_bundle_manifest, write_bundle_manifest
from src.artifacts.paths import result_run_dir
from src.artifacts.publish import load_ranking_policy, rank_models
from src.data.temporal_split import (
    is_temporal_manifest,
    load_temporal_manifest,
    temporal_source_split,
    validate_capture_chronology,
    verify_split_against_manifest,
)
from src.training.checkpoints import checkpoint_matches, load_checkpoint, write_checkpoint
from src.utils.io import json_dumps_strict

# ---------------------------------------------------------------------------
# CONFIG -- one dict, one source of truth.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

CONFIG: dict[str, Any] = {
    # -- Data ------------------------------------------------------------
    "raw_dir":            ROOT / "data" / "raw",
    "processed_dir":      ROOT / "data" / "processed",
    "csv_glob":           "*.csv",
    "csv_encoding":       "latin-1",
    "label_column":       "Label",
    "split_manifest":     None,
    # Columns carried alongside the features but never fed to a model.
    # "_row_hash" is no longer written -- it belonged to the retired
    # source-held-out protocol -- but stays listed so a cache built before
    # that removal still has the column excluded from the feature matrix.
    "metadata_columns":   ("dataset_id", "source_file", "capture_window",
                           "_row_hash", "_row_index"),
    "schema_drop_columns": ("Protocol",),

    # -- Reproducibility -------------------------------------------------
    "random_state":       42,

    # -- Composite subsampling (Layer 1) ---------------------------------
    # Classes with <= rare_threshold rows: keep ALL.
    # Classes with >  rare_threshold rows: stratified within remaining budget.
    # 5000 captures Heartbleed (11), Infiltration (36), Bot (~1956),
    # Web Attack (~2143), and leaves Brute Force (~13k) plus the four big
    # classes (BENIGN, DoS, DDoS, PortScan) to be subsampled.
    "subsample_n":        300_000,
    "rare_threshold":     5_000,

    # -- Split (Layer 2) -------------------------------------------------
    "test_size":          0.30,
    "min_test_per_class": 3,    # Heartbleed (11) -> 7 train / 4 test

    # -- Modeling (Layer 3) ---------------------------------------------
    "models":             (
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost",
        "mlp",
        "logistic_regression",
        "stacking",
    ),
    # Macro-F1 weights every attack family equally, so tuning cannot buy a
    # headline score by serving BENIGN well. Tuning against a single class's
    # F2 is unsound on a 2017-only corpus, where Infiltration has 36 rows.
    "primary_metric":     "f1_macro",
    "rf_class_weight":    "balanced_subsample",
    # ``class_weight`` treats every model identically and is the fair default
    # for a model comparison. ``targeted`` uses additional REAL rows of
    # target_class; the other sampler strategies run inside ImbPipeline and
    # therefore only see each training/CV fold. target_ratio means
    # target / majority after sampling.
    "imbalance_strategy": "class_weight",
    "target_class":       "Infiltration",
    "target_ratio":       1.00,
    # The delivery protocol uses a 70/30 train/test split with no calibration
    # partition. Set this explicitly above zero only for a separately approved
    # experiment; the locked test set must never select a threshold.
    "threshold_validation_size": 0.0,
    "target_max_fpr":      0.02,
    "accelerator":         "cpu",  # cpu | gpu; GPU affects XGBoost/CatBoost only
    "gpu_devices":         "0",

    # -- Reporting ------------------------------------------------------
    # A class needs at least this many test rows before its per-class recall
    # is treated as a stable estimate rather than an anecdote. Drives
    # f1_macro_reportable and the granularity warnings in the report.
    "reportable_min_test": 30,
    # Inference-cost measurement (dimension 4 of the evaluation standard).
    "latency_batch_size": 1_000,
    "latency_repeats":    30,

    # -- Trust checks (Layer 4) -----------------------------------------
    "cv_splits":          5,
    "label_shuffle_check":     True,
    "near_perfect_threshold":  0.99,

    # -- Hyperparameter search ------------------------------------------
    # HP search runs on a smaller subset of TRAIN (RAM + speed); best
    # estimator is then refit on the FULL train inside the search.
    "hp_search":             True,
    "hp_search_n_iter":      8,
    "hp_search_subsample":   80_000,
    "hp_search_jobs":        1,        # tree models saturate cores per fit

    # -- Outputs --------------------------------------------------------
    "results_root":       ROOT / "results",
    "run_name":           "latest",    # stable name -> resumable across runs

    # -- Leaky columns to drop if present (other CICIDS forks ship them) -
    "leaky_columns": (
        "Flow ID",
        "Source IP",       "Src IP",
        "Destination IP",  "Dst IP",
        "Source Port",     "Src Port",
        "Timestamp",
    ),

    # -- Smoke-test override (set via --smoke). Tiny budget for the
    # `python -W error::Warning train.py --smoke` clean-run proof.
    "smoke_subsample_n":  10_000,
}

IMBALANCE_STRATEGIES = (
    "class_weight",
    "targeted",
    "random_over",
    "borderline_smote",
    "smoteenn",
)
IMBALANCE_PROTOCOL_VERSION = 6  # strict row budget + verified target ratio
# 4: temporal split protocol, macro-F1 tuning objective, balanced HP grids,
#    split identity folded into the artifact-reuse signature.
TRAINING_PROTOCOL_VERSION = 4

# Objectives the hyperparameter search may optimise. Validated rather than
# forwarded to sklearn verbatim, so a typo fails immediately and every model
# in a comparison is provably tuned against the same objective.
PRIMARY_METRICS = (
    "f1_macro",
    "f1_weighted",
    "balanced_accuracy",
    "accuracy",
    "target_f2",
)

# Ordering key for the temporal split. Holds the row's position in its source
# CSV *before* any row is dropped by cleaning. The CICIDS2017 export stores
# flows in capture order, so this is a chronological ordering -- an assumption
# that ``src.data.temporal_split.validate_capture_chronology`` proves against
# the published CIC attack schedule rather than assuming.
ROW_INDEX_COLUMN = "_row_index"

# Only these three consult ``accelerator``: XGBoost takes device="cuda",
# CatBoost takes task_type="GPU", and stacking embeds an XGBoost base learner.
# RandomForest / LightGBM / MLP / LogisticRegression build byte-identical
# pipelines either way, so their artifacts must NOT be invalidated by a
# CPU->GPU switch -- otherwise flipping the flag throws away hours of finished
# work that would be recomputed to exactly the same result.
GPU_CAPABLE_MODELS = frozenset({"xgboost", "catboost", "stacking"})

DATASET_ID = "CICIDS2017"
CLEAN_CACHE_NAME = "cicids2017_clean.parquet"

# The eight CICIDS2017 capture files, in the order CIC recorded them. Every
# attack class lives in exactly one of these, which is why the split has to be
# chronological rather than source-held -- see src/data/temporal_split.py.
CICIDS2017_CAPTURES = (
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
)

MODEL_ALIASES = {
    "rf": "random_forest",
    "xgb": "xgboost",
    "lgbm": "lightgbm",
    "cat": "catboost",
    "nn": "mlp",
    "lr": "logistic_regression",
    "stack": "stacking",
}


# RAM presets -- override CONFIG by setting these via --preset NAME on the
# CLI. Lets a teammate with more RAM use the same script without editing
# CONFIG (and lets us pin the recommended values per RAM tier).
#
# Sizing rationale:
#   8gb : 300k subsample fits comfortably; HP search on 80k subset so a
#         single fit stays under ~2 min on a modern CPU.
#   16gb: 800k subsample captures more of BENIGN diversity without
#         pressuring page cache; HP search budget +50%.
#   32gb: subsample_n=None uses the full ~2.5M-row corpus; n_iter and
#         search subset both grow so HP search actually explores the
#         space rather than just doing a few random draws. hp_search_jobs
#         goes to 2 so two CV folds run in parallel (RAM headroom for
#         duplicate working sets of tree ensembles).
RAM_PRESETS: dict[str, dict[str, Any]] = {
    "8gb": {
        "subsample_n":          300_000,
        "hp_search_subsample":   80_000,
        "hp_search_n_iter":           8,
        "hp_search_jobs":             1,
    },
    "16gb": {
        "subsample_n":        1_500_000,    # maximum safe limit for 16GB RAM systems
        "hp_search_subsample":  150_000,
        "hp_search_n_iter":          12,
        "hp_search_jobs":             1,
    },
    "32gb": {
        "subsample_n":        3_000_000,    # safe limit for XGBoost/RF on 32GB RAM
        "hp_search_subsample":  200_000,
        "hp_search_n_iter":          20,
        "hp_search_jobs":             2,
    },
    "full": {
        "subsample_n":             None,    # use every row in the cleaned cache
        "hp_search_subsample":  300_000,
        "hp_search_n_iter":          20,
        "hp_search_jobs":             1,
    },
}


# ---------------------------------------------------------------------------
# Logging + warning hygiene
# ---------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    """Single root logger. Captures Python warnings to the same stream so
    that under ``-W error::Warning`` any warning is immediately visible
    AND is promoted to an exception by the interpreter."""
    log = logging.getLogger("train")
    log.setLevel(logging.INFO)
    # Keep this module's handler from propagating to main.py's root handler.
    log.propagate = False
    if log.handlers:
        return log
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    log.addHandler(handler)
    logging.captureWarnings(True)
    warning_log = logging.getLogger("py.warnings")
    warning_log.addHandler(handler)
    warning_log.propagate = False
    return log


LOG = setup_logging()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Label normalization (15 raw labels -> 10 canonical families)
# ---------------------------------------------------------------------------
def normalize_label(value: object) -> str:
    """Canonicalize a raw CICIDS2017/CSE-CIC-IDS2018 label.

    Strips 0x96 (Windows-1252 en-dash byte embedded in Web Attack labels),
    collapses whitespace, lowercases for matching. Maps to one of:
    BENIGN, DoS, DDoS, PortScan, Bot, Web Attack, Brute Force,
    Infiltration, Heartbleed, Other.

    Supports both CICIDS2017 and CIC-IDS2018 label schemes.
    Rows where the label cell literally equals 'Label' (corrupted header
    rows found in some 2018 CSVs) are mapped to 'Other' and later dropped.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Other"
    s = str(value)
    cleaned = "".join(ch if 0x20 <= ord(ch) < 0x7f else " " for ch in s)
    key = " ".join(cleaned.split()).lower()

    # ----- BENIGN / normal traffic ----------------------------------------
    if key in {"benign", "normal"}:
        return "BENIGN"

    # ----- DoS ---------------------------------------------------------------
    # CICIDS2017
    if key in {"dos hulk", "dos goldeneye", "dos slowloris", "dos slowhttptest"}:
        return "DoS"
    # CIC-IDS2018 (prefix "dos attacks-")
    if key.startswith("dos attacks-"):
        return "DoS"

    # ----- DDoS --------------------------------------------------------------
    # CICIDS2017
    if key == "ddos":
        return "DDoS"
    # CIC-IDS2018 variants
    if key in {
        "ddos attacks-loic-http",
        "ddos attack-hoic",
        "ddos attack-loic-udp",
        "ddos attack-loic-http",
    }:
        return "DDoS"
    if key.startswith("ddos"):
        return "DDoS"

    # ----- PortScan ----------------------------------------------------------
    if key == "portscan":
        return "PortScan"

    # ----- Bot ---------------------------------------------------------------
    if key == "bot":
        return "Bot"

    # ----- Web Attack --------------------------------------------------------
    # CICIDS2017
    if key.startswith("web attack"):
        return "Web Attack"
    # CIC-IDS2018
    if key in {
        "brute force -web",
        "brute force -xss",
        "sql injection",
    }:
        return "Web Attack"

    # ----- Brute Force -------------------------------------------------------
    # CICIDS2017
    if key in {"ftp-patator", "ssh-patator"}:
        return "Brute Force"
    # CIC-IDS2018
    if key in {"ftp-bruteforce", "ssh-bruteforce"}:
        return "Brute Force"
    if key.startswith("brute force"):
        return "Brute Force"

    # ----- Infiltration ------------------------------------------------------
    # CICIDS2017 + CIC-IDS2018 (typo: 'Infilteration')
    if key in {"infiltration", "infilteration"}:
        return "Infiltration"

    # ----- Heartbleed --------------------------------------------------------
    if key == "heartbleed":
        return "Heartbleed"

    # ----- corrupted header rows ('label') & anything unmapped → Other ------
    return "Other"


# ---------------------------------------------------------------------------
# Data loading + cleaning (RAM-efficient: clean per CSV, then concat)
# ---------------------------------------------------------------------------


# Some CICIDS2017 redistributions ship the packet-rate columns under the
# abbreviated CICFlowMeter names. Normalising them keeps the feature matrix
# identical across distributions, so a model trained on one can score the
# other; without it the run silently produces differently-named features.
_RATE_COLUMN_ALIASES: dict[str, str] = {
    "Fwd Pkts/s": "Fwd Packets/s",
    "Bwd Pkts/s": "Bwd Packets/s",
}


def _clean_one_frame(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Clean one CICIDS2017 CSV: strip column names, drop leaky/duplicate
    columns, normalise labels, replace Inf, drop NaN rows.

    Cleaning per CSV rather than on the concatenated corpus keeps the
    memory peak bounded.
    """
    # Strip whitespace from column names (CICIDS2017 has leading spaces).
    df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)

    # Record the raw CSV row number BEFORE any row is dropped. Every later
    # step in this function preserves relative order, so this column keeps the
    # capture ordering that the temporal split needs. It is metadata: excluded
    # from feature_cols below and from CONFIG["metadata_columns"] downstream.
    df[ROW_INDEX_COLUMN] = np.arange(len(df), dtype=np.int64)

    df = df.rename(columns=_RATE_COLUMN_ALIASES)

    if "Fwd Header Length.1" in df.columns:
        df = df.drop(columns=["Fwd Header Length.1"])

    leaky = [c for c in cfg["leaky_columns"] if c in df.columns]
    if leaky:
        df = df.drop(columns=leaky)

    # Protocol exists only in the 2018 export; drop it before concat so
    # missingness cannot become a hidden dataset-source indicator.
    schema_only = [c for c in cfg.get("schema_drop_columns", ()) if c in df.columns]
    if schema_only:
        df = df.drop(columns=schema_only)

    label_col = cfg["label_column"]
    if label_col not in df.columns:
        raise ValueError(
            f"Required label column {label_col!r} missing. "
            f"Got first 8 cols: {list(df.columns)[:8]}"
        )

    df = df.copy()

    # Drop corrupted header rows (some 2018 CSVs have 'Label' as a data row).
    df = df[df[label_col].astype(str).str.strip() != "Label"].reset_index(drop=True)

    df[label_col] = df[label_col].map(normalize_label).astype("string")

    # Drop rows mapped to 'Other' — these are either header artifacts or
    # unknown attack types that can't be reliably classified.
    df = df[df[label_col] != "Other"].reset_index(drop=True)

    # Metadata columns (currently _row_index) must not be coerced or cast to
    # float32 with the features -- the row number has to stay exact int64.
    metadata_cols = set(cfg.get("metadata_columns", ()))
    feature_cols = [
        c for c in df.columns if c != label_col and c not in metadata_cols
    ]
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    bad = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise ValueError(
            f"Non-numeric feature columns after cleanup: {bad}."
        )

    # Inf -> NaN via numpy mask. Avoids pandas 2.x downcasting FutureWarning
    # that DataFrame.replace([inf,-inf], nan) emits.
    arr = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    arr[~np.isfinite(arr)] = np.nan
    df[feature_cols] = arr

    df = df.dropna(subset=feature_cols + [label_col]).reset_index(drop=True)
    return df


def resolve_cache_path(cfg: dict[str, Any]) -> Path:
    """Path of the cleaned CICIDS2017 parquet cache."""
    return Path(cfg["processed_dir"]) / CLEAN_CACHE_NAME


def load_and_clean_cached(cfg: dict[str, Any], force: bool = False) -> pd.DataFrame:
    """Load, clean and cache the CICIDS2017 corpus.

    Cleaning runs per CSV to bound the memory peak, then the frames are
    concatenated and cached as parquet so later runs skip the whole step.
    Pass ``force`` (``--refresh-cache``) to rebuild.
    """
    cache = resolve_cache_path(cfg)
    if not force and cache.exists():
        LOG.info("Loading cleaned corpus from cache: %s", cache)
        df = pd.read_parquet(cache)
        cache_has_schema_only = any(
            c in df.columns for c in cfg.get("schema_drop_columns", ())
        )
        # A cache holding the abbreviated names predates alias normalisation,
        # so its feature columns no longer match what a fresh build produces.
        cache_has_aliases = any(c in df.columns for c in _RATE_COLUMN_ALIASES)
        cache_feature_cols = [
            c for c in df.columns
            if c != cfg["label_column"] and c not in cfg["metadata_columns"]
        ]
        cache_has_missing = bool(df[cache_feature_cols].isna().any().any())
        if ROW_INDEX_COLUMN not in df.columns:
            LOG.info("Cached corpus predates %s (the temporal ordering key); "
                     "rebuilding.", ROW_INDEX_COLUMN)
        elif cache_has_aliases:
            LOG.info("Cached corpus uses pre-normalisation rate column names; "
                     "rebuilding.")
        elif cache_has_schema_only or cache_has_missing:
            LOG.info("Cached corpus has stale schema or missing feature values; "
                     "rebuilding.")
        else:
            LOG.info("  -> %d rows x %d cols (from cache)", *df.shape)
            return df

    raw_dir = Path(cfg["raw_dir"])
    paths = sorted(raw_dir.glob(cfg["csv_glob"]))
    if not paths:
        raise FileNotFoundError(
            f"No CSV files found under {raw_dir}. Extract the CICIDS2017 "
            "MachineLearningCSV archive into data/raw/ (see README section 7)."
        )
    missing = sorted(set(CICIDS2017_CAPTURES) - {p.name for p in paths})
    if missing:
        raise FileNotFoundError(
            f"data/raw/ is missing {len(missing)} CICIDS2017 capture file(s): "
            f"{missing}. Every class lives in exactly one capture, so a missing "
            "file silently removes a whole attack class from the corpus."
        )
    paths = [p for p in paths if p.name in set(CICIDS2017_CAPTURES)]

    LOG.info("Loading %d CSV file(s) from %s (cleaning per-file to bound RAM)",
             len(paths), raw_dir)
    cleaned: list[pd.DataFrame] = []
    n_raw_total = 0
    for p in paths:
        # low_memory=False forces a single-pass dtype inference per column,
        # eliminating pandas DtypeWarning on Flow Bytes/s & Flow Packets/s.
        # latin-1 tolerates 0x96 byte in Web Attack labels.
        raw = pd.read_csv(p, low_memory=False, encoding=cfg["csv_encoding"])
        n_raw_total += len(raw)
        clean = _clean_one_frame(raw, cfg)
        # Preserve provenance outside the model feature matrix.  The cleaned
        # row number is stable for an unchanged source file and is sufficient
        # to derive deterministic quotas without introducing an RNG.
        clean["source_file"] = p.name
        clean["dataset_id"] = DATASET_ID
        clean["capture_window"] = p.stem
        LOG.info("  %s -> raw %d -> clean %d", p.name, len(raw), len(clean))
        del raw
        cleaned.append(clean)

    df = pd.concat(cleaned, axis=0, ignore_index=True)
    del cleaned
    LOG.info("Concatenated cleaned frames: %d rows", len(df))

    feature_cols = [
        c for c in df.columns
        if c != cfg["label_column"] and c not in cfg["metadata_columns"]
    ]
    missing = df[feature_cols].isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        raise ValueError(
            "Schema alignment failed: feature columns still contain missing values "
            f"after per-file cleaning: {missing.head(10).to_dict()}"
        )

    n_before = len(df)
    dedup_columns = [c for c in df.columns if c not in cfg.get("metadata_columns", ())]
    df = df.drop_duplicates(subset=dedup_columns).reset_index(drop=True)
    if len(df) != n_before:
        LOG.info("Dropped %d exact-duplicate rows (cross-CSV dedup)",
                 n_before - len(df))

    LOG.info("Cleaning summary: %d raw -> %d kept (%.2f%%)",
             n_raw_total, len(df), 100 * len(df) / max(n_raw_total, 1))

    cache.parent.mkdir(parents=True, exist_ok=True)
    temp_cache = cache.with_name(f"{cache.stem}.tmp{cache.suffix}")
    try:
        df.to_parquet(temp_cache, index=False)
        temp_cache.replace(cache)
    finally:
        if temp_cache.exists():
            temp_cache.unlink()
    LOG.info("Cached cleaned corpus -> %s", cache)
    return df


def preprocess_cache(*, force: bool = False) -> dict[str, Any]:
    """Build or validate the CICIDS2017 cache that ``train.py`` consumes."""
    cfg = dict(CONFIG)
    set_seeds(cfg["random_state"])
    df = load_and_clean_cached(cfg, force=force)
    feature_cols = [
        c for c in df.columns
        if c != cfg["label_column"] and c not in cfg["metadata_columns"]
    ]
    summary = {
        "cache_path": str(resolve_cache_path(cfg)),
        "dataset_id": DATASET_ID,
        "rows": int(len(df)),
        "features": int(len(feature_cols)),
        "sources": sorted(str(s) for s in df["source_file"].unique()),
        "label_distribution": {
            str(label): int(count)
            for label, count in df[cfg["label_column"]].value_counts().items()
        },
    }
    LOG.info("Canonical preprocessing complete: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Composite subsampling (Layer 1)
# ---------------------------------------------------------------------------
def stratified_split_min_test(
    df: pd.DataFrame,
    label_col: str,
    test_size: float,
    min_test_per_class: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-class manual split guaranteeing each class has at least
    ``min_test_per_class`` rows in test (and at least 1 in train).

    sklearn's stratified split only preserves proportions; for tiny
    classes (Heartbleed=11) proportional could leave only 2 in test,
    which makes per-class recall round to {0, 50%, 100%}. We force a
    minimum so the recall metric has at least 3-bin granularity.
    """
    rng = np.random.default_rng(random_state)
    train_idx: list[int] = []
    test_idx:  list[int] = []
    for _cls, group in df.groupby(label_col, sort=False, observed=True):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        # Want at least min_test_per_class in test, and at least 1 in train.
        proportional = int(round(n * test_size))
        test_n = max(min_test_per_class, proportional)
        test_n = min(test_n, n - 1) if n > 1 else 0
        test_idx.extend(idx[:test_n].tolist())
        train_idx.extend(idx[test_n:].tolist())

    train_idx_arr = np.asarray(train_idx, dtype=np.int64)
    test_idx_arr = np.asarray(test_idx, dtype=np.int64)
    rng.shuffle(train_idx_arr)
    rng.shuffle(test_idx_arr)

    train_df = df.iloc[train_idx_arr].reset_index(drop=True)
    test_df  = df.iloc[test_idx_arr ].reset_index(drop=True)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Models (Layer 3): class-weight aware classifiers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TargetClassRatio:
    """Pick a target-only over-sampling count for multiclass samplers."""

    target_class: int
    target_ratio: float

    def __call__(self, y: np.ndarray) -> dict[int, int]:
        values, counts = np.unique(np.asarray(y), return_counts=True)
        by_class = {
            int(cls): int(n) for cls, n in zip(values, counts, strict=True)
        }
        if self.target_class not in by_class:
            raise ValueError(
                f"encoded target class {self.target_class} is absent from this fold"
            )
        if not 0.0 < self.target_ratio <= 1.0:
            raise ValueError("target_ratio must be in the interval (0, 1]")
        majority_n = max(by_class.values())
        current = by_class[self.target_class]
        desired = max(current, int(np.ceil(majority_n * self.target_ratio)))
        return {self.target_class: desired}


class BalancedXGBClassifier(XGBClassifier):
    """XGBClassifier that auto-applies class-weight-balanced sample_weight.

    XGBoost has no native ``class_weight`` arg; the canonical workaround
    is to compute per-sample weights from ``y`` and pass them via
    ``fit(sample_weight=...)``. By doing it inside .fit() we get the
    right behaviour automatically when sklearn's CV passes per-fold y --
    each fold computes its own per-fold sample_weight.
    """

    def fit(self, X, y, sample_weight=None, **kwargs):
        if sample_weight is None:
            sample_weight = compute_sample_weight("balanced", y)
        return super().fit(X, y, sample_weight=sample_weight, **kwargs)


class BalancedMLPClassifier(MLPClassifier):
    """MLP with balanced per-row weights computed inside each fit/CV fold."""

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = compute_sample_weight("balanced", y)
        return super().fit(X, y, sample_weight=sample_weight)


class FlatCatBoostClassifier(CatBoostClassifier):
    """Normalize CatBoost's multiclass ``(n, 1)`` labels to sklearn ``(n,)``."""

    def predict(self, X, **kwargs):
        return np.asarray(super().predict(X, **kwargs)).reshape(-1)


class _LGBMNoFeatureNamesCheck(LGBMClassifier):
    """LightGBM that disables sklearn's predict-time feature-name check.

    LightGBM auto-assigns synthetic ``Column_N`` feature names during fit
    on a numpy array, then sklearn's predict-time check complains that
    the same numpy input "doesn't have valid feature names". The check
    is meaningless here because the upstream StandardScaler preserves
    column ORDER (which is what actually matters). Deleting the captured
    ``feature_names_in_`` after fit silences the warning without changing
    predictions.
    """

    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
        if hasattr(self, "feature_names_in_"):
            with suppress(AttributeError):
                object.__delattr__(self, "feature_names_in_")
        return self

    def _restore_feature_names(self, X):
        if not hasattr(X, "columns") and hasattr(self, "feature_names_in_"):
            return pd.DataFrame(X, columns=list(self.feature_names_in_))
        return X

    def predict(self, X, *args, **kwargs):
        return super().predict(self._restore_feature_names(X), *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        return super().predict_proba(
            self._restore_feature_names(X), *args, **kwargs
        )


class TargetThresholdPipeline(ImbPipeline):
    """Pipeline whose multiclass prediction honors a target-class threshold.

    Hyperparameter search and calibration leave ``target_threshold`` unset,
    which preserves the estimator's native argmax prediction. After a
    train-only calibration split selects a threshold, setting it makes both
    batch inference and dashboard predictions use the FN-aware decision rule.
    """

    def __init__(
        self,
        steps,
        *,
        target_class_index: int | None = None,
        target_threshold: float | None = None,
        transform_input=None,
        memory=None,
        verbose: bool = False,
    ):
        self.target_class_index = target_class_index
        self.target_threshold = target_threshold
        super().__init__(
            steps,
            transform_input=transform_input,
            memory=memory,
            verbose=verbose,
        )

    def predict(self, X, **params):
        if self.target_class_index is None or self.target_threshold is None:
            return super().predict(X, **params)
        probabilities = np.asarray(self.predict_proba(X, **params))
        target_idx = int(self.target_class_index)
        if not 0 <= target_idx < probabilities.shape[1]:
            raise ValueError(
                f"target_class_index={target_idx} is outside probability columns"
            )
        non_target = probabilities.copy()
        non_target[:, target_idx] = -np.inf
        predictions = np.argmax(non_target, axis=1)
        predictions[probabilities[:, target_idx] >= self.target_threshold] = (
            target_idx
        )
        return predictions


def target_fbeta_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target_class: int,
    beta: float = 2.0,
) -> float:
    """One-vs-rest F-beta for the class whose false negatives matter most."""
    return float(fbeta_score(
        np.asarray(y_true) == target_class,
        np.asarray(y_pred) == target_class,
        beta=beta,
        zero_division=0,
    ))


def search_scorer(metric: str, target_class_index: int):
    """Scorer for hyperparameter search.

    Every model in a comparison must be tuned against the same objective, so
    the metric name is validated here rather than forwarded verbatim into
    sklearn, where a typo would only surface deep inside the search.
    """
    if metric not in PRIMARY_METRICS:
        raise ValueError(
            f"Unsupported primary_metric {metric!r}. "
            f"Choices: {', '.join(sorted(PRIMARY_METRICS))}"
        )
    if metric == "target_f2":
        return make_scorer(
            target_fbeta_score,
            target_class=target_class_index,
            beta=2.0,
        )
    return metric


@dataclass(frozen=True)
class ThresholdCalibration:
    threshold: float
    recall: float
    precision: float
    f2: float
    false_positive_rate: float
    false_positives: int
    false_negatives: int


def calibrate_target_threshold(
    y_true: np.ndarray,
    target_probabilities: np.ndarray,
    *,
    target_class_index: int,
    max_false_positive_rate: float,
) -> ThresholdCalibration:
    """Choose the best target F1 under a validation FPR ceiling.

    F1 is used at the decision-threshold stage because it penalizes both
    target false positives and false negatives.  Hyperparameter search may
    still use target-F2 to make the representation sensitive to the rare
    class, while this final calibration avoids spending the entire FPR
    budget merely to maximize recall.
    """
    if not 0.0 <= max_false_positive_rate <= 1.0:
        raise ValueError("max_false_positive_rate must be in [0, 1]")
    y_binary = np.asarray(y_true) == target_class_index
    probabilities = np.asarray(target_probabilities, dtype=float)
    if probabilities.ndim != 1 or len(probabilities) != len(y_binary):
        raise ValueError("target_probabilities must be 1-D and align with y_true")
    if not y_binary.any() or y_binary.all():
        raise ValueError("threshold calibration needs target and non-target rows")

    fpr, recall, thresholds = roc_curve(
        y_binary,
        probabilities,
        drop_intermediate=False,
    )
    candidates = np.flatnonzero(
        (fpr <= max_false_positive_rate)
        & np.isfinite(thresholds)
        & (thresholds >= 0.0)
        & (thresholds <= 1.0)
    )
    n_positive = int(y_binary.sum())
    n_negative = len(y_binary) - n_positive

    def balanced_rank(idx: int) -> tuple[float, float, float, float]:
        tp = recall[idx] * n_positive
        fp = fpr[idx] * n_negative
        precision = tp / max(tp + fp, np.finfo(float).eps)
        target_f1 = (
            2.0 * precision * recall[idx]
            / max(precision + recall[idx], np.finfo(float).eps)
        )
        return (
            float(target_f1),
            float(precision),
            float(recall[idx]),
            float(thresholds[idx]),
        )

    if len(candidates) == 0:
        # A threshold that violates the declared FPR policy is not a valid
        # operating point.  Failing calibration keeps promotion from silently
        # turning a security constraint into a best-effort suggestion.
        raise ValueError(
            f"No finite target threshold satisfies max FPR "
            f"{max_false_positive_rate:.4f}"
        )
    best = max(candidates, key=balanced_rank)
    threshold = float(thresholds[best])
    predicted = probabilities >= threshold
    fp = int(np.sum(predicted & ~y_binary))
    fn = int(np.sum(~predicted & y_binary))
    tn = int(np.sum(~predicted & ~y_binary))
    return ThresholdCalibration(
        threshold=threshold,
        recall=float(recall_score(y_binary, predicted, zero_division=0)),
        precision=float(precision_score(y_binary, predicted, zero_division=0)),
        f2=float(fbeta_score(
            y_binary,
            predicted,
            beta=2.0,
            zero_division=0,
        )),
        false_positive_rate=float(fp / max(fp + tn, 1)),
        false_positives=fp,
        false_negatives=fn,
    )


def build_pipeline(
    model_name: str,
    n_classes: int,
    random_state: int,
    *,
    rf_class_weight: str | None = "balanced_subsample",
    imbalance_strategy: str = "class_weight",
    target_class_index: int | None = None,
    target_ratio: float = 0.20,
    accelerator: str = "cpu",
    gpu_devices: str = "0",
) -> TargetThresholdPipeline:
    """Return an unfitted, leakage-safe scaler/sampler/classifier pipeline."""
    valid_strategies = {
        "class_weight", "targeted", "random_over",
        "borderline_smote", "smoteenn",
    }
    if imbalance_strategy not in valid_strategies:
        raise ValueError(
            f"unknown imbalance_strategy={imbalance_strategy!r}; "
            f"choose from {sorted(valid_strategies)}"
        )
    if accelerator not in {"cpu", "gpu"}:
        raise ValueError("accelerator must be 'cpu' or 'gpu'")
    use_class_weights = imbalance_strategy == "class_weight"
    use_gpu = accelerator == "gpu"

    if model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            # 'balanced_subsample' recomputes weights per bootstrap sample
            # -- more accurate than 'balanced' on extreme imbalance.
            class_weight=rf_class_weight if use_class_weights else None,
            n_jobs=-1,
            random_state=random_state,
        )
    elif model_name == "xgboost":
        # XGBoost 2.x: no `use_label_encoder`. We set objective + eval_metric
        # explicitly so behaviour is pinned across minor versions and no
        # deprecation warnings fire.
        xgb_class = BalancedXGBClassifier if use_class_weights else XGBClassifier
        clf = xgb_class(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            device="cuda" if use_gpu else "cpu",
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            num_class=n_classes if n_classes > 2 else None,
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            n_jobs=-1,
            random_state=random_state,
        )
    elif model_name == "lightgbm":
        # verbosity=-1 silences LightGBM's C++ stdout chatter.
        # force_col_wise stops the "guessing column-wise vs row-wise"
        # info log. class_weight='balanced' uses sklearn's heuristic.
        # Subclass disables sklearn's predict-time feature-name check
        # (see _LGBMNoFeatureNamesCheck docstring above).
        # min_child_samples=1: LightGBM's leaf-creation threshold is on the
        # RAW row count, not the weighted one. With Heartbleed (8 train rows
        # -> 1-2 per CV fold), the default 20 prevents the model from EVER
        # creating a leaf for the rare class, and CV f1_macro collapses on
        # those folds (verified: prior run gave per-fold scores
        # [0.18, 0.09, 0.99, 0.34, 0.99] -- bimodal). XGBoost avoids this
        # via min_child_weight (weighted hessian); RF via min_samples_leaf=1.
        # We mirror that here so LightGBM also learns from very small leaves.
        # bagging_fraction=1.0 + feature_fraction=1.0 + bagging_freq=0:
        # disable LightGBM's random row/column sub-sampling. With Heartbleed
        # at 1-2 rows per CV fold, a 10% bagging dropout has a non-trivial
        # chance of removing the rare-class rows from any given tree's
        # training set, which is what drove the prior bimodal CV pattern.
        # RF and XGB both use full-data per tree in our config; mirror that.
        clf = _LGBMNoFeatureNamesCheck(
            n_estimators=600,
            num_leaves=127,
            learning_rate=0.05,
            feature_fraction=1.0,
            bagging_fraction=1.0,
            bagging_freq=0,
            min_child_samples=1,
            min_split_gain=0.0,
            class_weight="balanced" if use_class_weights else None,
            n_jobs=-1,
            random_state=random_state,
            verbosity=-1,
            force_col_wise=True,
        )
    elif model_name == "catboost":
        cat_params: dict[str, Any] = {
            "iterations": 400,
            "depth": 8,
            "learning_rate": 0.05,
            "thread_count": -1,
            "random_seed": random_state,
            "verbose": 0,
            "allow_writing_files": False,
        }
        if use_class_weights:
            cat_params["auto_class_weights"] = "Balanced"
        if use_gpu:
            cat_params.update({"task_type": "GPU", "devices": gpu_devices})
        clf = FlatCatBoostClassifier(**cat_params)
    elif model_name == "mlp":
        mlp_class = BalancedMLPClassifier if use_class_weights else MLPClassifier
        clf = mlp_class(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            learning_rate_init=0.001,
            alpha=0.0001,
            random_state=random_state,
        )
    elif model_name == "logistic_regression":
        clf = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            class_weight="balanced" if use_class_weights else None,
            max_iter=2_000,
            tol=0.001,
            random_state=random_state,
        )
    elif model_name == "stacking":
        stack_xgb_class = (
            BalancedXGBClassifier if use_class_weights else XGBClassifier
        )
        base_estimators = [
            ("lgbm", _LGBMNoFeatureNamesCheck(
                n_estimators=300,
                num_leaves=63,
                learning_rate=0.05,
                feature_fraction=0.9,
                bagging_fraction=0.9,
                bagging_freq=5,
                class_weight="balanced" if use_class_weights else None,
                n_jobs=-1,
                verbosity=-1,
                random_state=random_state,
            )),
            ("xgb", stack_xgb_class(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                device="cuda" if use_gpu else "cpu",
                objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                num_class=n_classes if n_classes > 2 else None,
                eval_metric="mlogloss" if n_classes > 2 else "logloss",
                n_jobs=-1,
                random_state=random_state,
            )),
            ("rf", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced" if use_class_weights else None,
                n_jobs=-1,
                random_state=random_state,
            )),
        ]
        meta = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1_000,
            class_weight="balanced" if use_class_weights else None,
            random_state=random_state,
        )
        clf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta,
            cv=5,
            stack_method="predict_proba",
            n_jobs=1,
            passthrough=False,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")
    # Imputation remains inside the fitted pipeline so an uploaded/inference
    # row with a missing value is handled using train-only statistics.
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ]
    if imbalance_strategy in {"random_over", "borderline_smote", "smoteenn"}:
        if target_class_index is None:
            raise ValueError(
                f"{imbalance_strategy} requires target_class_index"
            )
        sampling_strategy = TargetClassRatio(target_class_index, target_ratio)
        if imbalance_strategy == "random_over":
            sampler = RandomOverSampler(
                sampling_strategy=sampling_strategy,
                random_state=random_state,
            )
        elif imbalance_strategy == "borderline_smote":
            sampler = BorderlineSMOTE(
                sampling_strategy=sampling_strategy,
                random_state=random_state,
                k_neighbors=5,
            )
        else:
            sampler = SMOTEENN(
                smote=SMOTE(
                    sampling_strategy=sampling_strategy,
                    random_state=random_state,
                    k_neighbors=5,
                ),
                random_state=random_state,
            )
        steps.append(("sampler", sampler))
    steps.append(("clf", clf))
    pipe = TargetThresholdPipeline(
        steps,
        target_class_index=target_class_index,
    )
    # Make StandardScaler output a DataFrame (preserving feature names) so
    # the downstream classifier sees the same X shape on fit and predict.
    # LightGBM 4.x sets feature_names_in_ from DataFrame columns at fit; if
    # predict receives a numpy array, sklearn emits the "X does not have
    # valid feature names" UserWarning. set_output("pandas") fixes this end
    # to end and is the sklearn 1.3+ recommended way.
    # StackingClassifier fits its meta-learner on an ndarray but, with pandas
    # output enabled, predicts on a named DataFrame in sklearn 1.8. Keep the
    # stacking path consistently ndarray to avoid feature-name drift warnings.
    if model_name != "stacking":
        pipe.set_output(transform="pandas")
    return pipe


# Every tunable model gets a search space of exactly this size, so that a
# shared ``hp_search_n_iter`` means the same fraction of the space is explored
# for each one. Equal trial counts over unequal spaces is not a fair
# comparison: the old grids ranged from 8 combinations (logistic regression,
# fully enumerated after 8 draws) to 256 (XGBoost, 8% explored at n_iter=20).
HP_SEARCH_SPACE_SIZE = 144

# Models whose search space is deliberately empty, with the reason. Recorded
# in the metrics JSON and the report rather than left silent.
HP_UNTUNED_MODELS = {
    "stacking": (
        "StackingClassifier refits every base estimator for each candidate, so "
        "a comparable search would cost n_iter x the whole ensemble. It is "
        "reported as untuned rather than given a smaller, unfair budget."
    ),
}


def hp_grids(model_name: str) -> dict[str, list]:
    """Randomized-search distribution for one model.

    All grids enumerate :data:`HP_SEARCH_SPACE_SIZE` combinations. None of them
    touches ``class_weight``: that is owned by ``--imbalance-strategy`` and must
    stay identical across models, or the imbalance treatment stops being a
    controlled variable.
    """
    if model_name == "random_forest":
        return {
            "clf__n_estimators":      [200, 300, 400, 500],
            "clf__max_depth":         [None, 20, 30, 40],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf":  [1, 2, 4],
        }
    if model_name == "xgboost":
        return {
            "clf__n_estimators":  [200, 400, 600, 800],
            "clf__max_depth":     [6, 8, 10],
            "clf__learning_rate": [0.03, 0.05, 0.1, 0.15],
            "clf__subsample":     [0.8, 0.9, 1.0],
        }
    if model_name == "lightgbm":
        # min_child_samples grid: 1-10 only. Anything >=10 starves the
        # Heartbleed-class leaf on per-fold CV (7 train rows -> 1-2 per
        # fold). See build_pipeline() docstring for the diagnosis.
        return {
            "clf__n_estimators":       [400, 600, 800, 1000],
            "clf__num_leaves":         [63, 127, 255],
            "clf__learning_rate":      [0.03, 0.05, 0.1, 0.15],
            "clf__min_child_samples":  [1, 2, 5],
        }
    if model_name == "catboost":
        return {
            "clf__iterations":    [300, 400, 500, 600],
            "clf__depth":         [6, 8, 10],
            "clf__learning_rate": [0.03, 0.05, 0.1, 0.15],
            "clf__l2_leaf_reg":   [1, 3, 5],
        }
    if model_name == "mlp":
        return {
            "clf__hidden_layer_sizes": [
                (128, 64), (256, 128, 64), (256, 128, 64, 32), (512, 256, 128),
            ],
            "clf__alpha": [0.00001, 0.0001, 0.001],
            "clf__learning_rate_init": [0.0005, 0.001, 0.003, 0.005],
            "clf__batch_size": [256, 512, 1024],
        }
    if model_name == "logistic_regression":
        # No max_iter here, deliberately. It is the optimiser's budget, not a
        # property of the model: including values below the configured budget
        # makes the search compare under-converged fits against converged ones,
        # spends trials on results that would never be deployed, and emits
        # ConvergenceWarning -- which this project promotes to an exception.
        # The space is filled out along C instead, which is the parameter that
        # actually governs a logistic regression.
        return {
            "clf__C": [
                0.001, 0.002, 0.003, 0.005,
                0.01, 0.02, 0.03, 0.05,
                0.1, 0.2, 0.3, 0.5,
                1.0, 2.0, 3.0, 5.0,
                10.0, 20.0, 30.0, 50.0,
                100.0, 200.0, 300.0, 500.0,
            ],
            # All at or above the sklearn default (1e-4): a looser tolerance
            # stops earlier, so none of these needs more iterations than the
            # configured budget already allows.
            "clf__tol": [0.0001, 0.0005, 0.001],
            "clf__fit_intercept": [True, False],
        }
    return {}


def hp_search_space_size(model_name: str) -> int:
    """Number of distinct configurations in a model's search space."""
    size = 1
    for values in hp_grids(model_name).values():
        size *= len(values)
    return size if hp_grids(model_name) else 0


def hp_grid_fingerprint(model_name: str) -> str:
    """Content hash of a model's search space.

    Part of artifact identity: the space size alone cannot detect a grid whose
    *values* changed, so without this an edited grid would silently reuse a
    model tuned against the old one.
    """
    grid = {key: list(values) for key, values in sorted(hp_grids(model_name).items())}
    encoded = json.dumps(grid, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


# Keys produced by extended_metrics_from_confusion, and by
# measure_inference_cost + the efficiency block around it. Declared explicitly
# so the EvalResult <-> JSON round trip is exact and testable, instead of the
# mapping being retyped at each of the three sites that persist a result.
EXTENDED_METRIC_KEYS = (
    "precision_macro", "recall_macro", "precision_weighted", "recall_weighted",
    "f1_macro_reportable", "reportable_min_test", "reportable_classes",
    "per_class_precision", "per_class_recall", "per_class_f1",
    "per_class_fpr", "per_class_fnr", "per_class_support",
    "per_class_false_positives", "per_class_false_negatives",
    "binary_precision", "binary_recall", "binary_f1",
    "binary_fpr", "binary_fnr",
    "binary_false_positives", "binary_false_negatives",
    "binary_benign_support", "binary_attack_support",
    "mcc",
)

EFFICIENCY_KEYS = (
    "fit_seconds", "model_size_mb",
    "predict_batch_size", "predict_repeats",
    "predict_latency_p50_ms", "predict_latency_p95_ms",
    "throughput_flows_per_sec",
    "predict_device", "predict_moved_from_gpu",
    "process_rss_mb", "fit_rss_delta_mb",
)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    model:               str
    accuracy:            float
    balanced_accuracy:   float
    f1_macro:            float
    f1_weighted:         float
    per_class:           pd.DataFrame   = field(repr=False)
    confusion:           np.ndarray     = field(repr=False)
    cv_scores:           list[float]    = field(default_factory=list)
    cv_mean:             float | None   = 0.0
    cv_std:              float | None   = 0.0
    shuffle_accuracy:    float | None   = None
    shuffle_f1_macro:    float | None   = None
    majority_baseline:   float          = 0.0
    best_params:         dict[str, Any] = field(default_factory=dict)
    target_threshold:    float | None   = None
    target_precision:    float          = 0.0
    target_recall:       float          = 0.0
    target_f1:           float          = 0.0
    target_f2:           float          = 0.0
    target_fpr:          float          = 0.0
    target_false_positives: int         = 0
    target_false_negatives: int         = 0
    target_to_benign_fn: int            = 0
    calibration_recall:  float | None   = None
    calibration_fpr:     float | None   = None
    # Dimensions 1-2 of the evaluation standard beyond the headline four.
    extended:            dict[str, Any] = field(default_factory=dict, repr=False)
    # Dimension 4: computational efficiency.
    efficiency:          dict[str, Any] = field(default_factory=dict, repr=False)
    hp_space_size:       int            = 0
    hp_tuned:            bool           = False


def evaluate(model: Pipeline, X_test: np.ndarray, y_test: np.ndarray,
             class_names: list[str]):
    y_pred = model.predict(X_test)
    labels = list(range(len(class_names)))
    acc  = accuracy_score(y_test, y_pred)
    # Average over the same label set as f1_macro. These two are compared
    # against each other in the report, so they must share a denominator;
    # averaging balanced accuracy over np.unique(y_test) while dividing
    # f1_macro by n_classes made them incomparable whenever a class was
    # missing from the test set. Under the temporal protocol every class is
    # present on both sides, so this changes nothing there -- it removes a
    # trap for corpora where that does not hold.
    bacc = recall_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)
    f1m = f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)
    f1w = f1_score(y_test, y_pred, labels=labels, average="weighted", zero_division=0)
    # labels=range(n) ensures every class appears in the report; otherwise
    # missing classes trigger UndefinedMetricWarning.
    report_dict = classification_report(
        y_test, y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    per_class = pd.DataFrame(report_dict).transpose().round(4)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    return acc, bacc, f1m, f1w, per_class, cm


def target_metrics_from_confusion(
    cm: np.ndarray,
    *,
    target_class_index: int,
    benign_class_index: int,
) -> dict[str, float | int]:
    tp = int(cm[target_class_index, target_class_index])
    fn = int(cm[target_class_index, :].sum() - tp)
    fp = int(cm[:, target_class_index].sum() - tp)
    tn = int(cm.sum() - tp - fn - fp)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (
        2.0 * precision * recall
        / max(precision + recall, np.finfo(float).eps)
    )
    beta_sq = 4.0
    f2 = (
        (1.0 + beta_sq) * precision * recall
        / max(beta_sq * precision + recall, np.finfo(float).eps)
    )
    return {
        "target_precision": float(precision),
        "target_recall": float(recall),
        "target_f1": float(f1),
        "target_f2": float(f2),
        "target_fpr": float(fp / max(fp + tn, 1)),
        "target_false_positives": fp,
        "target_false_negatives": fn,
        "target_to_benign_fn": int(cm[target_class_index, benign_class_index]),
    }


def extended_metrics_from_confusion(
    cm: np.ndarray,
    class_names: list[str],
    *,
    benign_class_index: int,
    reportable_min_test: int,
) -> dict[str, Any]:
    """Metrics the evaluation standard requires that headline numbers omit.

    Everything is derived from the confusion matrix ``evaluate`` already
    returns, so no second pass over the test set is needed and the numbers
    cannot drift from the reported matrix.

    Covers, per the evaluation standard:

    * macro and weighted **Precision** and **Recall** (only per-class values
      existed before, inside the classification report CSV);
    * per-class one-vs-rest **FPR** and **FNR** -- previously computed for the
      single configured target class only;
    * an Attack-vs-BENIGN **binary view**, which is what makes the alert
      fatigue argument concrete: one FPR over the whole BENIGN population;
    * ``f1_macro_reportable``, macro-F1 restricted to classes with enough test
      rows to be a stable estimate. On a 2017-only corpus Heartbleed (4 test
      rows) and Infiltration (11) each carry 1/9 of plain macro-F1 while being
      statistical noise, so both numbers are reported side by side rather than
      one being quietly substituted for the other.
    * Matthews correlation coefficient, a single imbalance-robust summary.
    """
    cm = np.asarray(cm, dtype=np.float64)
    total = cm.sum()
    support = cm.sum(axis=1)                      # true rows per class
    predicted = cm.sum(axis=0)                    # predicted per class
    tp = np.diag(cm)
    fn = support - tp
    fp = predicted - tp
    tn = total - tp - fn - fp

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
        recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
        f1 = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros_like(tp), where=(precision + recall) > 0)
        fpr = np.divide(fp, fp + tn, out=np.zeros_like(tp), where=(fp + tn) > 0)
        fnr = np.divide(fn, support, out=np.zeros_like(tp), where=support > 0)

    # Weighted averages use true support, matching sklearn's convention.
    weights = support / total if total else np.zeros_like(support)

    reportable = support >= reportable_min_test
    f1_macro_reportable = float(f1[reportable].mean()) if reportable.any() else 0.0

    # Attack vs BENIGN. "Positive" is Attack, so a false positive is a BENIGN
    # flow raised as an alert and a false negative is a missed attack.
    attack = np.ones(len(class_names), dtype=bool)
    attack[benign_class_index] = False
    benign_total = float(support[benign_class_index])
    attack_total = float(support[attack].sum())
    # BENIGN rows predicted as any attack class.
    benign_as_attack = float(cm[benign_class_index, attack].sum())
    # Attack rows predicted as BENIGN.
    attack_as_benign = float(cm[np.ix_(attack, [benign_class_index])].sum())
    attack_as_attack = float(cm[np.ix_(attack, attack)].sum())

    binary_fpr = benign_as_attack / benign_total if benign_total else 0.0
    binary_fnr = attack_as_benign / attack_total if attack_total else 0.0
    binary_recall = attack_as_attack / attack_total if attack_total else 0.0
    binary_predicted_attack = attack_as_attack + benign_as_attack
    binary_precision = (
        attack_as_attack / binary_predicted_attack if binary_predicted_attack else 0.0
    )
    binary_f1 = (
        2 * binary_precision * binary_recall / (binary_precision + binary_recall)
        if (binary_precision + binary_recall) > 0 else 0.0
    )

    # Matthews correlation coefficient, multiclass form (Gorodkin 2004).
    correct = float(tp.sum())
    sum_pk_tk = float((predicted * support).sum())
    sum_tk_sq = float((support ** 2).sum())
    sum_pk_sq = float((predicted ** 2).sum())
    mcc_denominator = np.sqrt(
        (total ** 2 - sum_pk_sq) * (total ** 2 - sum_tk_sq)
    )
    mcc = (
        float((correct * total - sum_pk_tk) / mcc_denominator)
        if mcc_denominator > 0 else 0.0
    )

    def by_class(values: np.ndarray) -> dict[str, float]:
        return {name: float(value) for name, value in zip(class_names, values, strict=True)}

    return {
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "precision_weighted": float((precision * weights).sum()),
        "recall_weighted": float((recall * weights).sum()),
        "f1_macro_reportable": f1_macro_reportable,
        "reportable_min_test": int(reportable_min_test),
        "reportable_classes": [
            name for name, keep in zip(class_names, reportable, strict=True) if keep
        ],
        "per_class_precision": by_class(precision),
        "per_class_recall": by_class(recall),
        "per_class_f1": by_class(f1),
        "per_class_fpr": by_class(fpr),
        "per_class_fnr": by_class(fnr),
        "per_class_support": {
            name: int(value) for name, value in zip(class_names, support, strict=True)
        },
        "per_class_false_positives": {
            name: int(value) for name, value in zip(class_names, fp, strict=True)
        },
        "per_class_false_negatives": {
            name: int(value) for name, value in zip(class_names, fn, strict=True)
        },
        "binary_precision": binary_precision,
        "binary_recall": binary_recall,
        "binary_f1": binary_f1,
        "binary_fpr": binary_fpr,
        "binary_fnr": binary_fnr,
        "binary_false_positives": int(benign_as_attack),
        "binary_false_negatives": int(attack_as_benign),
        "binary_benign_support": int(benign_total),
        "binary_attack_support": int(attack_total),
        "mcc": mcc,
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str],
                          out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar=True, ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


@contextmanager
def _inference_on_cpu(model: Pipeline):
    """Temporarily move any CUDA-resident estimator to CPU for timing.

    The Deployment ranking is decided on ``predict_latency_p95_ms``. If the two
    GPU-capable models were timed on a GPU while the other five were timed on a
    CPU, that ranking would be measuring hardware rather than models, and the
    comparison would be meaningless.

    Only XGBoost exposes a ``device`` parameter that survives fitting;
    CatBoost already predicts on CPU regardless of ``task_type="GPU"``. The
    sweep over ``get_params`` also catches the XGBoost nested inside the
    stacking ensemble. Restored afterwards, and the artifact on disk is written
    before this runs, so the saved model is never affected either way.
    """
    try:
        params = model.get_params(deep=True)
        cuda_params = {
            name: value
            for name, value in params.items()
            if name.endswith("device") and value == "cuda"
        }
    except (AttributeError, TypeError):
        params, cuda_params = {}, {}

    def _set_booster_device(device: str) -> None:
        """Move the fitted Booster too.

        ``set_params`` only updates the sklearn wrapper. The fitted Booster
        keeps its own device, and predicting across the mismatch makes
        XGBoost fall back to a DMatrix copy *and emit a UserWarning* -- which
        this project promotes to an exception under ``-W error::Warning``.
        """
        for name in cuda_params:
            owner = params.get(name.removesuffix("device").removesuffix("__"))
            getter = getattr(owner, "get_booster", None)
            if getter is None:
                continue
            with suppress(Exception):
                getter().set_param({"device": device})

    if cuda_params:
        model.set_params(**dict.fromkeys(cuda_params, "cpu"))
        _set_booster_device("cpu")
    try:
        yield bool(cuda_params)
    finally:
        if cuda_params:
            model.set_params(**cuda_params)
            _set_booster_device("cuda")


def measure_inference_cost(
    model: Pipeline,
    X_test: pd.DataFrame,
    *,
    batch_size: int = 1_000,
    repeats: int = 30,
    random_state: int = 42,
    force_cpu: bool = True,
) -> dict[str, float]:
    """Per-batch prediction latency and throughput for one fitted pipeline.

    Detection quality is only half of what an IDS is judged on -- a model that
    cannot keep up with the link is not deployable regardless of its F1. This
    reports the tail (p95) alongside the median, because an IDS that is usually
    fast and occasionally slow still drops flows.

    Batches are drawn from random offsets so the measurement is not dominated
    by whichever class happens to sit at the head of the test set, and one
    warm-up batch is discarded to exclude lazy allocation inside the estimator.

    ``force_cpu`` keeps every model on the same hardware so the numbers stay
    comparable across a mixed CPU/GPU run.
    """
    n_rows = len(X_test)
    if n_rows == 0 or repeats < 1:
        return {}
    batch_size = min(batch_size, n_rows)
    rng = np.random.default_rng(random_state)

    context = _inference_on_cpu(model) if force_cpu else nullcontext(False)
    with context as moved_from_gpu:
        # Warm-up: the first predict pays one-off allocation costs, and after a
        # device switch it also pays the transfer.
        model.predict(X_test.iloc[:batch_size])

        durations: list[float] = []
        for _ in range(repeats):
            start = int(rng.integers(0, max(n_rows - batch_size, 0) + 1))
            batch = X_test.iloc[start:start + batch_size]
            t_start = time.perf_counter()
            model.predict(batch)
            durations.append(time.perf_counter() - t_start)

    seconds = np.asarray(durations, dtype=np.float64)
    median = float(np.median(seconds))
    return {
        "predict_batch_size": int(batch_size),
        "predict_repeats": int(repeats),
        "predict_latency_p50_ms": median * 1_000.0,
        "predict_latency_p95_ms": float(np.percentile(seconds, 95)) * 1_000.0,
        "throughput_flows_per_sec": (batch_size / median) if median > 0 else 0.0,
        # Recorded so the report can state where the timing was taken rather
        # than asserting "on CPU" and hoping.
        "predict_device": "cpu" if force_cpu else "as-trained",
        "predict_moved_from_gpu": bool(moved_from_gpu),
    }


def peak_rss_mb() -> float | None:
    """Resident set size of this process in MiB, or ``None`` if unavailable.

    ``tracemalloc`` is deliberately not used: it only sees Python-level
    allocations and would miss essentially all of the memory a tree ensemble
    or a numpy feature matrix occupies, understating the figure badly enough
    to be misleading.
    """
    try:
        import psutil
    except ImportError:
        return None
    return float(psutil.Process().memory_info().rss) / (1024.0 ** 2)


def hardware_profile(cfg: dict[str, Any]) -> dict[str, Any]:
    """Machine identity, so latency figures stay interpretable later."""
    profile: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "logical_cores": os.cpu_count(),
        "accelerator": cfg.get("accelerator"),
    }
    try:
        import psutil

        profile["logical_cores"] = psutil.cpu_count(logical=True)
        profile["physical_cores"] = psutil.cpu_count(logical=False)
        profile["total_ram_gb"] = round(psutil.virtual_memory().total / (1024.0 ** 3), 2)
    except ImportError:
        profile["note"] = "psutil unavailable; core/RAM detail omitted"
    return profile


def cross_validate_clean(
    estimator: Pipeline, X: np.ndarray, y: np.ndarray, cv: Any,
    *, groups: np.ndarray | None = None,
) -> tuple[list[float], float, float]:
    """Macro-F1 CV; source-held runs never mix one capture across folds."""
    labels = np.unique(y)
    scorer = make_scorer(f1_score, labels=labels, average="macro", zero_division=0)
    scores = cross_val_score(
        clone(estimator), X, y, cv=cv, groups=groups, scoring=scorer, n_jobs=1,
    )
    return list(scores), float(scores.mean()), float(scores.std())


def label_shuffle_sanity(estimator: Pipeline, X: pd.DataFrame, y: np.ndarray,
                         random_state: int,
                         min_test_per_class: int) -> tuple[float, float]:
    """Refit on label-shuffled data; macro-F1 must collapse to ~chance.

    Why macro-F1, not accuracy: in extreme imbalance, accuracy under shuffle
    just tracks the majority-class rate (uninformative). Macro-F1 averages
    per-class F1, so collapsing to ~1/n_classes proves the pipeline is NOT
    learning ANY class structure (i.e. not leaking labels through features
    or scaler state).
    """
    rng = np.random.default_rng(random_state)

    # Subsample X and y to at most 100k rows using stratified sampling to guarantee all classes survive
    if len(y) > 100_000:
        classes = np.unique(y)
        budget_per_class = max(1, 100_000 // len(classes))
        keep_idx = []
        for cls in classes:
            idx = np.flatnonzero(y == cls)
            take = min(budget_per_class, len(idx))
            keep_idx.extend(rng.choice(idx, size=take, replace=False).tolist())
        keep_idx = np.asarray(keep_idx, dtype=np.int64)
        X = X.iloc[keep_idx].reset_index(drop=True)
        y = y[keep_idx]

    y_shuf = y.copy()
    rng.shuffle(y_shuf)
    # Use the same min-test stratified split so tiny classes survive.
    tmp = pd.DataFrame({"_idx": np.arange(len(y_shuf)), "_y": y_shuf})
    train_df, test_df = stratified_split_min_test(
        tmp, "_y", test_size=0.25,
        min_test_per_class=min_test_per_class,
        random_state=random_state,
    )
    tr = train_df["_idx"].to_numpy()
    te = test_df["_idx"].to_numpy()
    est = clone(estimator)
    est.fit(X.iloc[tr].reset_index(drop=True), y_shuf[tr])
    y_pred = est.predict(X.iloc[te].reset_index(drop=True))
    return (
        float(accuracy_score(y_shuf[te], y_pred)),
        float(f1_score(y_shuf[te], y_pred, average="macro", zero_division=0)),
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _fmt(value: Any, places: int = 4) -> str:
    """Format a metric for a markdown cell, or ``n/a`` when absent."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return str(value)


def _ranking_section(results: list[EvalResult], cfg: dict[str, Any]) -> list[str]:
    """Three rankings, each with the declared rule that produced it.

    A single "best model" hides the trade-off the reader actually cares about.
    The policy lives in ``configs/ranking_policy.json`` and is applied to the
    same numbers reported above, so a reader can re-derive every winner.
    """
    lines = ["## Model rankings -- three axes", ""]
    payload = [_eval_result_to_dict(r, cfg) for r in results]
    try:
        rankings = rank_models(payload, load_ranking_policy())
    except (OSError, ValueError, KeyError) as exc:
        lines.append(f"> Ranking policy could not be applied: {exc}")
        lines.append("")
        return lines

    lines.append("No single ranking is authoritative. A model that detects the "
                 "most attacks may raise too many alerts to be usable, and the "
                 "fastest model is rarely the most accurate, so all three are "
                 "published side by side with the rule that produced each.")
    lines.append("")
    lines.append("| ranking | winner | decided by | rule |")
    lines.append("|---|---|---:|---|")
    for entry in rankings.values():
        winner = entry.get("model") or "none"
        value = _fmt(entry.get("objective_value"))
        lines.append(
            f"| **{entry['label']}** | `{winner}` | "
            f"{entry.get('objective', '')} = {value} | {entry.get('rule', '')} |"
        )
    lines.append("")
    for entry in rankings.values():
        if entry.get("status") == "conditional_no_model_meets_constraints":
            lines.append(
                f"> **{entry['label']}**: no model satisfied every constraint, "
                "so the winner shown is the unconstrained best. Treat it as "
                "provisional."
            )
        for reason in entry.get("excluded", []):
            lines.append(f"> {entry['label']} excluded {reason}")
    lines.append("")
    return lines


def _consequences_section(
    results: list[EvalResult],
    cfg: dict[str, Any],
    class_names: list[str],
    per_class_n_test: dict[str, int],
    per_class_n_train: dict[str, int],
) -> list[str]:
    """The four consequences, argued from this run's own numbers.

    Written from the measured confusion matrices rather than as boilerplate,
    so the claims are checkable against the tables above.
    """
    lines = ["## Consequences", ""]
    scored = [r for r in results if r.extended]
    if not scored:
        lines.append("> No per-class detail available in this run.")
        lines.append("")
        return lines

    best = max(scored, key=lambda r: r.f1_macro)
    e = best.extended

    # 1. A false negative costs more than a false positive.
    lines.append("**1. A false negative is more damaging than a false positive.**")
    worst_fn = max(
        scored, key=lambda r: r.extended.get("binary_false_negatives", 0)
    )
    lines.append(
        f"Across this test set the models miss between "
        f"{min(r.extended.get('binary_false_negatives', 0) for r in scored):,} and "
        f"{worst_fn.extended.get('binary_false_negatives', 0):,} attack flows. "
        "Every missed flow is an intrusion that reached the network with no "
        "alert raised, whereas every false positive costs an analyst a few "
        "minutes. That asymmetry is why the Security-focused ranking maximises "
        "recall rather than accuracy."
    )
    lines.append("")

    # 2. False positives cause alert fatigue.
    lines.append("**2. False positives cause alert fatigue.**")
    benign_support = e.get("binary_benign_support", 0) or 0
    fp = e.get("binary_false_positives", 0)
    rate = e.get("binary_fpr", 0.0)
    try:
        fpr_cap = float(load_ranking_policy()["max_binary_fpr"])
        cap_text = (
            f" The Security ranking therefore caps binary FPR at "
            f"{fpr_cap:.1%} rather than leaving it free."
        )
    except (OSError, ValueError, KeyError):
        cap_text = ""
    lines.append(
        f"`{best.model}` has a binary FPR of {rate:.5f}. That sounds "
        f"negligible, but against {benign_support:,} benign test flows it is "
        f"**{fp:,} false alerts**. An analyst who cannot triage that volume "
        "starts ignoring the queue, at which point the true positives are "
        "missed too -- so an unbounded FPR destroys detection indirectly."
        + cap_text
    )
    lines.append("")

    # 3. Class imbalance makes accuracy misleading.
    lines.append("**3. Class imbalance makes accuracy misleading.**")
    recalls = {k: v for k, v in (e.get("per_class_recall") or {}).items()
               if k != "BENIGN"}
    if recalls:
        weakest = min(recalls, key=recalls.get)
        weakest_recall = recalls[weakest]
        missed = e.get("per_class_false_negatives", {}).get(weakest, 0)
        support = per_class_n_test.get(weakest, 0)
        lines.append(
            f"`{best.model}` reports accuracy {best.accuracy:.4f} against a "
            f"majority-class baseline of {best.majority_baseline:.4f} -- a lift "
            f"of only {best.accuracy - best.majority_baseline:+.4f}. Its "
            f"weakest attack class is **{weakest}**, at recall "
            f"{weakest_recall:.4f}: {missed:,} of {support:,} test flows "
            "missed. Accuracy hides this completely because the class is a "
            "rounding error in the total, which is exactly why macro-F1 and "
            "per-class recall decide this comparison."
        )
    lines.append("")

    # 4. Speed matters in an IDS.
    lines.append("**4. Inference speed is a deployment constraint, not a detail.**")
    timed = [r for r in results if r.efficiency.get("predict_latency_p95_ms")]
    if timed:
        fastest = min(timed, key=lambda r: r.efficiency["predict_latency_p95_ms"])
        slowest = max(timed, key=lambda r: r.efficiency["predict_latency_p95_ms"])
        ratio = (
            slowest.efficiency["predict_latency_p95_ms"]
            / max(fastest.efficiency["predict_latency_p95_ms"], 1e-9)
        )
        lines.append(
            f"p95 latency spans {fastest.efficiency['predict_latency_p95_ms']:.2f} ms "
            f"(`{fastest.model}`) to {slowest.efficiency['predict_latency_p95_ms']:.2f} ms "
            f"(`{slowest.model}`) per {cfg['latency_batch_size']:,}-flow batch -- "
            f"a {ratio:,.1f}x spread, or "
            f"{fastest.efficiency.get('throughput_flows_per_sec', 0):,.0f} versus "
            f"{slowest.efficiency.get('throughput_flows_per_sec', 0):,.0f} flows/s. "
            f"Artifact size ranges {min(r.efficiency.get('model_size_mb', 0) for r in timed):,.1f} MB "
            f"to {max(r.efficiency.get('model_size_mb', 0) for r in timed):,.1f} MB. "
            "A sensor that cannot keep pace with the link drops flows, and a "
            "dropped flow is a false negative by another name."
        )
    else:
        lines.append(
            "No latency was measured in this run (all models were reused), so "
            "this dimension cannot be argued from the current numbers."
        )
    lines.append("")
    return lines


def write_report(results: list[EvalResult], outdir: Path, cfg: dict[str, Any],
                 n_classes: int, class_names: list[str],
                 per_class_n_test: dict[str, int],
                 per_class_n_train: dict[str, int]) -> Path:
    chance = 1.0 / max(n_classes, 1)
    corpus = DATASET_ID
    protocol = cfg.get("split_protocol", "unknown")
    lines: list[str] = []
    lines.append(f"# {corpus} -- training run `{cfg['run_name']}`")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(f"- Corpus: {corpus}")
    lines.append(f"- Split protocol: `{protocol}`")
    if str(protocol).startswith("cicids2017_temporal"):
        lines.append(
            "  - Chronological 70/30 inside every (capture file, class) group: "
            "the earliest 70% of each class's flows train, the latest 30% test. "
            "No test flow precedes the training flows of its own class."
        )
        lines.append(
            "  - Ordering key: original CSV row position. The 2017 export has no "
            "`Timestamp` column, so row order is used and is verified against "
            "the published CIC attack schedule before every run."
        )
        lines.append(
            "  - Source holdout is impossible on this corpus: each attack class "
            "occurs in exactly one capture file, so holding out a file would "
            "leave its class with no training rows."
        )
        lines.append(
            "  - CV uses StratifiedKFold, not GroupKFold by source: with one "
            "capture per class, source grouping would place an entire class in "
            "a single fold."
        )
    else:
        lines.append(f"- subsample_n: {cfg['subsample_n']!r}")
        lines.append(f"- rare_threshold (keep-all-rows below this): "
                     f"{cfg['rare_threshold']}")
    lines.append(f"- test_size: {cfg['test_size']}, "
                 f"min_test_per_class: {cfg['min_test_per_class']}")
    lines.append(
        f"- imbalance_strategy: {cfg['imbalance_strategy']} "
        "(identical for every model, so imbalance handling stays a "
        "controlled variable)"
    )
    lines.append("- resampling scope: TRAIN/CV folds only; test distribution untouched")
    if cfg["cv_check"]:
        lines.append(f"- CV: {cfg['cv_splits']}-fold, scored on macro-F1")
    else:
        lines.append("- CV: skipped")
    lines.append("")
    lines.append("### Tuning fairness")
    lines.append("")
    if cfg["hp_search"]:
        lines.append(
            f"Every tunable model is searched with the same method "
            f"(RandomizedSearchCV), the same objective "
            f"(`{cfg['primary_metric']}`), the same budget "
            f"(n_iter={cfg['hp_search_n_iter']} on a "
            f"{cfg['hp_search_subsample']:,}-row train subset), and a search "
            f"space of exactly {HP_SEARCH_SPACE_SIZE} configurations -- so the "
            f"same fraction ({cfg['hp_search_n_iter'] / HP_SEARCH_SPACE_SIZE:.1%}) "
            "of each space is explored. Equal trial counts over unequal spaces "
            "would not be a fair comparison."
        )
    else:
        lines.append("Hyperparameter search was skipped for this run; every "
                     "model uses its configured defaults.")
    for name, reason in HP_UNTUNED_MODELS.items():
        if name in cfg["models"]:
            lines.append("")
            lines.append(f"> `{name}` is **not** tuned. {reason}")
    lines.append("")
    lines.append(f"- Random state: {cfg['random_state']}")
    lines.append(f"- Classes ({n_classes}): {', '.join(class_names)}")
    lines.append("")

    lines.append("## Per-class sample sizes (REPORT-CRITICAL)")
    lines.append("")
    lines.append("| class | n_train | n_test | granularity warning |")
    lines.append("|---|---|---|---|")
    for cls in class_names:
        n_tr = per_class_n_train.get(cls, 0)
        n_te = per_class_n_test.get(cls, 0)
        warn = ""
        if n_te < 10:
            warn = "indicative only -- small N"
        if n_te < 5:
            warn = "very low confidence -- treat as anecdote"
        lines.append(f"| {cls} | {n_tr} | {n_te} | {warn} |")
    lines.append("")
    lines.append("> Per-class recall for any class with `n_test < 10` should "
                 "be read as an upper-bound estimate, not a stable metric. "
                 "This is most visible for classes with only a handful of "
                 "rows after subsampling, especially Heartbleed.")
    lines.append("")

    # ---- Dimension 1: Classification Performance --------------------
    lines.append("## Dimension 1 -- Classification Performance")
    lines.append("")
    lines.append("Macro-F1 and per-class recall decide this comparison, not "
                 "accuracy. With BENIGN at "
                 f"{max(per_class_n_test.values()) / max(sum(per_class_n_test.values()), 1):.1%} "
                 "of the test set, a model that predicted BENIGN and nothing "
                 "else would already score that as accuracy while detecting "
                 "no attacks at all.")
    lines.append("")
    lines.append("| model | accuracy | balanced acc | macro P | macro R | "
                 "f1_macro | f1_macro (reportable) | f1_weighted | wtd P | "
                 "wtd R | MCC |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(results, key=lambda item: -item.f1_macro):
        e = r.extended
        lines.append(
            f"| {r.model} | {r.accuracy:.4f} | {r.balanced_accuracy:.4f} | "
            f"{_fmt(e.get('precision_macro'))} | {_fmt(e.get('recall_macro'))} | "
            f"{r.f1_macro:.4f} | {_fmt(e.get('f1_macro_reportable'))} | "
            f"{r.f1_weighted:.4f} | {_fmt(e.get('precision_weighted'))} | "
            f"{_fmt(e.get('recall_weighted'))} | {_fmt(e.get('mcc'))} |"
        )
    lines.append("")
    reportable = next(
        (r.extended.get("reportable_classes") for r in results
         if r.extended.get("reportable_classes")), None,
    )
    if reportable is not None:
        excluded = [c for c in class_names if c not in reportable]
        lines.append(
            f"> `f1_macro (reportable)` averages only the {len(reportable)} "
            f"classes with at least {cfg['reportable_min_test']} test rows"
            + (f", excluding {', '.join(excluded)}. " if excluded else ". ")
            + "Both numbers are shown because plain macro-F1 gives a class "
            "with a handful of test rows the same weight as one with "
            "hundreds of thousands; neither number alone tells the whole "
            "story."
        )
        lines.append("")

    lines.append("### Trust checks")
    lines.append("")
    lines.append("| model | CV f1_macro (mean +/- std) | majority baseline acc "
                 "| shuffled-labels f1_macro |")
    lines.append("|---|---|---:|---:|")
    for r in results:
        shuf = (f"{r.shuffle_f1_macro:.4f}"
                if r.shuffle_f1_macro is not None else "skipped")
        cv_text = (
            f"{r.cv_mean:.4f} +/- {r.cv_std:.4f}"
            if r.cv_mean is not None and r.cv_std is not None else "skipped"
        )
        lines.append(
            f"| {r.model} | {cv_text} | {r.majority_baseline:.4f} | {shuf} |"
        )
    lines.append("")

    # ---- Dimension 2: Attack Detection Ability ----------------------
    lines.append("## Dimension 2 -- Attack Detection Ability")
    lines.append("")
    lines.append("Attack-vs-BENIGN view. A false positive is a benign flow "
                 "raised as an alert; a false negative is an attack that "
                 "reached the network unnoticed.")
    lines.append("")
    lines.append("| model | binary recall (DR) | binary precision | binary F1 "
                 "| FPR | FNR | false alerts | attacks missed |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(results, key=lambda item: -(item.extended.get("binary_recall") or 0.0)):
        e = r.extended
        if not e:
            continue
        lines.append(
            f"| {r.model} | {_fmt(e.get('binary_recall'))} | "
            f"{_fmt(e.get('binary_precision'))} | {_fmt(e.get('binary_f1'))} | "
            f"{_fmt(e.get('binary_fpr'), 5)} | {_fmt(e.get('binary_fnr'), 5)} | "
            f"{e.get('binary_false_positives', 0):,} | "
            f"{e.get('binary_false_negatives', 0):,} |"
        )
    lines.append("")

    lines.append("### Per-class recall (detection rate)")
    lines.append("")
    header = "| model | " + " | ".join(class_names) + " |"
    lines.append(header)
    lines.append("|---" * (len(class_names) + 1) + "|")
    for r in results:
        recalls = r.extended.get("per_class_recall") or {}
        if not recalls:
            continue
        cells = " | ".join(_fmt(recalls.get(cls)) for cls in class_names)
        lines.append(f"| {r.model} | {cells} |")
    lines.append("")

    lines.append("### Per-class false-negative rate")
    lines.append("")
    lines.append(header)
    lines.append("|---" * (len(class_names) + 1) + "|")
    for r in results:
        fnrs = r.extended.get("per_class_fnr") or {}
        if not fnrs:
            continue
        cells = " | ".join(_fmt(fnrs.get(cls)) for cls in class_names)
        lines.append(f"| {r.model} | {cells} |")
    lines.append("")

    lines.append("### Per-class false-positive rate")
    lines.append("")
    lines.append(header)
    lines.append("|---" * (len(class_names) + 1) + "|")
    for r in results:
        fprs = r.extended.get("per_class_fpr") or {}
        if not fprs:
            continue
        cells = " | ".join(_fmt(fprs.get(cls), 5) for cls in class_names)
        lines.append(f"| {r.model} | {cells} |")
    lines.append("")
    lines.append("Confusion matrices: `<model>_confusion_matrix.png`; "
                 "full per-class precision/recall/F1: `<model>_per_class.csv`.")
    lines.append("")

    # ---- Dimension 3: Operational Impact ----------------------------
    lines.append("## Dimension 3 -- Operational Impact")
    lines.append("")
    lines.append("What the error rates cost an analyst on this test set "
                 f"({sum(per_class_n_test.values()):,} flows, of which "
                 f"{per_class_n_test.get('BENIGN', 0):,} are BENIGN).")
    lines.append("")
    lines.append("| model | false alerts | alerts per 10k benign flows | "
                 "attacks missed | worst-detected class | its recall |")
    lines.append("|---|---:|---:|---:|---|---:|")
    for r in sorted(results, key=lambda item: item.extended.get("binary_false_negatives", 0)):
        e = r.extended
        if not e:
            continue
        benign_support = e.get("binary_benign_support", 0) or 0
        fp = e.get("binary_false_positives", 0)
        per_10k = (fp / benign_support * 10_000) if benign_support else 0.0
        recalls = e.get("per_class_recall") or {}
        attack_recalls = {k: v for k, v in recalls.items() if k != "BENIGN"}
        worst = min(attack_recalls, key=attack_recalls.get) if attack_recalls else "-"
        lines.append(
            f"| {r.model} | {fp:,} | {per_10k:,.1f} | "
            f"{e.get('binary_false_negatives', 0):,} | {worst} | "
            f"{_fmt(attack_recalls.get(worst))} |"
        )
    lines.append("")

    # ---- Dimension 4: Computational Efficiency ----------------------
    lines.append("## Dimension 4 -- Computational Efficiency")
    lines.append("")
    lines.append("Inference is measured on CPU for **every** model in batches "
                 f"of {cfg['latency_batch_size']:,} flows, "
                 f"{cfg['latency_repeats']} repeats from random offsets, after "
                 "one discarded warm-up batch. p95 matters as much as the "
                 "median: an IDS that is usually fast and occasionally slow "
                 "still drops flows.")
    moved = sorted(r.model for r in results
                   if r.efficiency.get("predict_moved_from_gpu"))
    if moved:
        lines.append("")
        lines.append(
            f"> {', '.join(moved)} trained on GPU and were moved to CPU for "
            "timing. Otherwise this table would rank hardware rather than "
            "models -- and the Deployment ranking is decided on p95 latency."
        )
    lines.append("")
    lines.append("| model | trained on | fit time | p50 latency | p95 latency "
                 "| throughput | model size |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in sorted(results, key=lambda item: (
        item.efficiency.get("predict_latency_p95_ms") or float("inf")
    )):
        eff = r.efficiency
        if not eff:
            continue
        fit = eff.get("fit_seconds")
        lines.append(
            f"| {r.model} | {effective_accelerator(cfg, r.model)} | "
            f"{f'{fit:,.1f}s' if fit is not None else 'resumed'} | "
            f"{_fmt(eff.get('predict_latency_p50_ms'), 2)} ms | "
            f"{_fmt(eff.get('predict_latency_p95_ms'), 2)} ms | "
            f"{eff.get('throughput_flows_per_sec', 0):,.0f} flows/s | "
            f"{eff.get('model_size_mb', 0):,.1f} MB |"
        )
    lines.append("")
    if not any(r.efficiency for r in results):
        lines.append("> No efficiency figures in this run -- every model was "
                     "reused from a previous run's artifacts.")
        lines.append("")

    lines.append(f"## {cfg['target_class']} false-negative metrics")
    lines.append("")
    lines.append(
        "| model | threshold | precision | recall | F2 | FPR | FN | "
        "FN to BENIGN | FP |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(results, key=lambda item: item.target_false_negatives):
        threshold = (
            f"{r.target_threshold:.6f}"
            if r.target_threshold is not None
            else "native"
        )
        lines.append(
            f"| {r.model} | {threshold} | {r.target_precision:.4f} | "
            f"{r.target_recall:.4f} | {r.target_f2:.4f} | "
            f"{r.target_fpr:.4f} | {r.target_false_negatives} | "
            f"{r.target_to_benign_fn} | {r.target_false_positives} |"
        )
    lines.append("")

    lines.append("## Verdict on accuracy")
    lines.append("")
    for r in results:
        verdict: list[str] = []
        if r.accuracy >= cfg["near_perfect_threshold"]:
            verdict.append(
                f"`{r.model}` reports test accuracy {r.accuracy:.4f} "
                f"(>= {cfg['near_perfect_threshold']}). For CICIDS-style "
                "flow features this can be plausible because the dataset is "
                "highly separable for tree ensembles. Trust checks:"
            )
        else:
            verdict.append(f"`{r.model}` test accuracy {r.accuracy:.4f} -- "
                           "in the expected range.")
        lift = r.accuracy - r.majority_baseline
        verdict.append(
            f"  - majority-class baseline accuracy = {r.majority_baseline:.4f}. "
            f"Model lift = {lift:+.4f}. macro_f1 = {r.f1_macro:.4f} is the "
            "load-bearing number on this imbalanced dataset."
        )
        if r.cv_mean is None or r.cv_std is None:
            verdict.append(
                "  - CV trust check was skipped for this run to keep full-corpus "
                "training practical. Use a subsampled run with CV when you need "
                "fold-stability evidence."
            )
        else:
            n_folds = len(r.cv_scores) if r.cv_scores else cfg["cv_splits"]
            # std interpretation: <0.02 very stable, <0.05 stable, <0.10 acceptable,
            # >=0.10 unstable -- minority class likely missing some folds.
            if r.cv_std < 0.02:
                std_msg = "small std confirms result is not a single-lucky-split fluke"
            elif r.cv_std < 0.05:
                std_msg = "std is acceptable; result is reasonably stable"
            elif r.cv_std < 0.10:
                std_msg = ("std is moderate; the test score is real but CV folds "
                           "vary -- treat the headline as best-case rather than mean")
            else:
                std_msg = ("**UNSTABLE** (std >= 0.10). Likely cause: some CV folds "
                           "contained too few rows of a minority class (Heartbleed "
                           "has only 8 train rows total). The test-set number is "
                           "still valid (verified by the shuffled-labels check) but "
                           "this model has high variance across splits and may not "
                           "generalise well to new minority-class instances")
            verdict.append(
                f"  - {n_folds}-fold CV f1_macro = {r.cv_mean:.4f} +/- {r.cv_std:.4f} "
                f"({std_msg})."
            )
        if r.shuffle_f1_macro is not None:
            verdict.append(
                f"  - shuffled-labels f1_macro = {r.shuffle_f1_macro:.4f} "
                f"(chance level = {chance:.4f}). Collapse to ~chance confirms "
                "the pipeline is NOT leaking labels through preprocessing."
            )
        lines.extend("- " + v for v in verdict)
        lines.append("")

    lines.extend(_ranking_section(results, cfg))
    lines.extend(_consequences_section(results, cfg, class_names,
                                       per_class_n_test, per_class_n_train))

    lines.append("## Top weaknesses + concrete improvements")
    lines.append("")
    rare = [c for c in class_names if per_class_n_test.get(c, 0) < cfg["reportable_min_test"]]
    if rare:
        detail = ", ".join(
            f"{c} ({per_class_n_train.get(c, 0)} train / {per_class_n_test.get(c, 0)} test)"
            for c in rare
        )
        lines.append(
            f"1. **Minority-class metric variance** -- {detail}. A per-class "
            "recall computed from single-digit test rows moves in steps of "
            "tens of percent, so it is an anecdote, not an estimate. This "
            "report therefore publishes `f1_macro (reportable)` alongside "
            "plain macro-F1 and flags every affected class in the sample-size "
            "table above."
        )
    else:
        lines.append(
            f"1. **Minority-class metric variance** -- every class has at "
            f"least {cfg['reportable_min_test']} test rows in this run, so no "
            "per-class metric is sample-size limited."
        )
    lines.append(
        "2. **Single-capture attack families** -- in CICIDS2017 each attack "
        "class occurs in exactly one capture file, so a model sees only one "
        "instance of each attack campaign. The chronological split prevents "
        "leakage within a capture but cannot show whether a model generalises "
        "to a *different* execution of the same attack. Improvement: validate "
        "the champion against CSE-CIC-IDS2018 as an unseen-campaign test."
    )
    lines.append(
        "3. **CICIDS labelling noise** -- labels are assigned per attack "
        "window, not per flow, so benign flows inside an attack window can "
        "carry an attack label. This inflates every model's apparent "
        "performance equally, so the ranking stays valid while the absolute "
        "numbers should be read as an upper bound."
    )
    lines.append(
        "4. **Flow-completion ordering** -- CICFlowMeter emits a record when a "
        "flow terminates, so the chronological order is by completion, not by "
        "start. Long-lived attacks (DoS GoldenEye) therefore trail past their "
        "attack window. This is the correct ordering for an IDS -- a flow's "
        "features only exist once it completes -- but it is not the same as "
        "ordering by attack time."
    )
    lines.append("")

    lines.append("## Verifying the clean run")
    lines.append("")
    lines.append("```")
    lines.append("python -W error::Warning train.py")
    lines.append("```")
    lines.append("")
    lines.append("Any warning becomes a hard exception. Exit code 0 = clean.")
    lines.append("")

    report_path = outdir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv: list[str]) -> dict[str, Any]:
    """Tiny arg-parser for training experiments."""
    args = {"smoke": False, "force": False, "models": None,
            "run_name": None, "preset": None, "refresh_cache": False,
            "refresh_plots": False,
            "skip_hp": False, "primary_metric": None,
            "reuse_best_params": False,
            "rf_class_weight": None, "skip_cv": False,
            "skip_label_shuffle": False,
            "imbalance_strategy": None, "target_class": None,
            "target_ratio": None, "target_max_fpr": None,
            "threshold_validation_size": None, "split_manifest": None}
    it = iter(argv)
    for tok in it:
        if tok == "--smoke":
            args["smoke"] = True
        elif tok == "--force":
            args["force"] = True
        elif tok == "--refresh-cache":
            args["refresh_cache"] = True
        elif tok == "--refresh-plots":
            args["refresh_plots"] = True
        elif tok == "--skip-hp":
            args["skip_hp"] = True
        elif tok == "--reuse-best-params":
            args["reuse_best_params"] = True
        elif tok == "--skip-cv":
            args["skip_cv"] = True
        elif tok == "--skip-label-shuffle":
            args["skip_label_shuffle"] = True
        elif tok == "--models":
            requested = tuple(part.strip() for part in next(it).split(","))
            if requested == ("all",):
                args["models"] = CONFIG["models"]
            else:
                canonical = tuple(MODEL_ALIASES.get(name, name) for name in requested)
                unknown = sorted(set(canonical) - set(CONFIG["models"]))
                if unknown:
                    raise SystemExit(
                        f"unknown model(s): {unknown}. Choices: {list(CONFIG['models'])}"
                    )
                args["models"] = canonical
        elif tok == "--run-name":
            args["run_name"] = next(it)
        elif tok == "--split-manifest":
            args["split_manifest"] = Path(next(it)).resolve()
        elif tok == "--accelerator":
            value = next(it).lower()
            if value not in {"cpu", "gpu"}:
                raise SystemExit("--accelerator must be cpu or gpu")
            args["accelerator"] = value
        elif tok == "--gpu-devices":
            args["gpu_devices"] = next(it)
        elif tok == "--primary-metric":
            value = next(it)
            if value not in PRIMARY_METRICS:
                raise SystemExit(
                    f"--primary-metric must be one of: "
                    f"{', '.join(sorted(PRIMARY_METRICS))}"
                )
            args["primary_metric"] = value
        elif tok == "--imbalance-strategy":
            value = next(it).lower()
            if value not in IMBALANCE_STRATEGIES:
                raise SystemExit(
                    "--imbalance-strategy must be one of: "
                    + ", ".join(IMBALANCE_STRATEGIES)
                )
            args["imbalance_strategy"] = value
        elif tok == "--target-class":
            args["target_class"] = next(it)
        elif tok == "--target-ratio":
            try:
                value = float(next(it))
            except ValueError as exc:
                raise SystemExit("--target-ratio must be a number") from exc
            if not 0.0 < value <= 1.0:
                raise SystemExit("--target-ratio must be in the interval (0, 1]")
            args["target_ratio"] = value
        elif tok == "--target-max-fpr":
            try:
                value = float(next(it))
            except ValueError as exc:
                raise SystemExit("--target-max-fpr must be a number") from exc
            if not 0.0 <= value <= 1.0:
                raise SystemExit("--target-max-fpr must be in the interval [0, 1]")
            args["target_max_fpr"] = value
        elif tok == "--threshold-validation-size":
            try:
                value = float(next(it))
            except ValueError as exc:
                raise SystemExit(
                    "--threshold-validation-size must be a number"
                ) from exc
            if not 0.0 <= value < 0.5:
                raise SystemExit(
                    "--threshold-validation-size must be in the interval [0, 0.5)"
                )
            args["threshold_validation_size"] = value
        elif tok == "--rf-class-weight":
            value = next(it).lower()
            choices = {"none": None, "balanced": "balanced",
                       "balanced_subsample": "balanced_subsample"}
            if value not in choices:
                raise SystemExit(
                    "--rf-class-weight must be one of: "
                    "none, balanced, balanced_subsample"
                )
            args["rf_class_weight"] = choices[value]
        elif tok == "--preset":
            name = next(it).lower()
            if name not in RAM_PRESETS:
                raise SystemExit(
                    f"unknown preset {name!r}. "
                    f"Choices: {sorted(RAM_PRESETS)}"
                )
            args["preset"] = name
        elif tok in ("-h", "--help"):
            print("usage: train.py [--smoke] [--force] [--refresh-cache] "
                  "[--refresh-plots] "
                  "[--skip-hp] [--primary-metric METRIC] "
                  "[--reuse-best-params] "
                  "[--imbalance-strategy class_weight|targeted|random_over|"
                  "borderline_smote|smoteenn] "
                  "[--target-class CLASS] [--target-ratio RATIO] "
                  "[--target-max-fpr RATE] "
                  "[--threshold-validation-size FRACTION] "
                  "[--accelerator cpu|gpu] [--gpu-devices DEVICES] "
                  "[--rf-class-weight none|balanced|balanced_subsample] "
                  "[--preset 8gb|16gb|32gb|full] "
                  "[--models rf,xgb,lgbm,cat,nn,lr,stack|all] "
                  "[--skip-cv] [--skip-label-shuffle] "
                  "[--run-name NAME] [--split-manifest PATH]")
            sys.exit(0)
        else:
            raise SystemExit(f"unknown argument: {tok}")
    return args


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)

    cfg = dict(CONFIG)
    # Preset first (sets RAM-tier defaults), then smoke (overrides for
    # verification regardless of preset), then explicit --run-name.
    if args["preset"]:
        preset = RAM_PRESETS[args["preset"]]
        cfg.update(preset)
        LOG.info("Applied RAM preset %r: %s", args["preset"], preset)
    if args["smoke"]:
        cfg["subsample_n"]    = cfg["smoke_subsample_n"]
        cfg["hp_search"]      = False
        cfg["cv_splits"]      = 3
        cfg["run_name"]       = "smoke"
        LOG.info("Smoke mode: subsample_n=%d, hp_search=off, cv_splits=3",
                 cfg["subsample_n"])
    if args["models"]:
        cfg["models"] = args["models"]
    if args["run_name"]:
        cfg["run_name"] = args["run_name"]
    if args["split_manifest"]:
        cfg["split_manifest"] = args["split_manifest"]
    if args["skip_hp"]:
        cfg["hp_search"] = False
    cfg["cv_check"] = not args["skip_cv"]
    if args["skip_label_shuffle"]:
        cfg["label_shuffle_check"] = False
    if args["primary_metric"]:
        cfg["primary_metric"] = args["primary_metric"]
    if args["imbalance_strategy"]:
        cfg["imbalance_strategy"] = args["imbalance_strategy"]
    if args["target_class"]:
        cfg["target_class"] = args["target_class"]
    if args["target_ratio"] is not None:
        cfg["target_ratio"] = args["target_ratio"]
    if args["target_max_fpr"] is not None:
        cfg["target_max_fpr"] = args["target_max_fpr"]
    if args["threshold_validation_size"] is not None:
        cfg["threshold_validation_size"] = args["threshold_validation_size"]
    if args.get("accelerator") is not None:
        cfg["accelerator"] = args["accelerator"]
    if args.get("gpu_devices") is not None:
        cfg["gpu_devices"] = args["gpu_devices"]
    if args["rf_class_weight"] is not None or "--rf-class-weight" in argv:
        cfg["rf_class_weight"] = args["rf_class_weight"]

    set_seeds(cfg["random_state"])

    if cfg["accelerator"] == "gpu":
        from src.training.gpu import run_gpu_acceptance

        gpu_check = run_gpu_acceptance(str(cfg["gpu_devices"]))
        failures = {
            name: item["detail"]
            for name, item in gpu_check["checks"].items()
            if not item["passed"]
        }
        if failures:
            raise SystemExit(f"GPU acceptance failed: {failures}")
        cpu_only = set(CONFIG["models"]) - GPU_CAPABLE_MODELS
        selected_cpu_models = sorted(set(cfg["models"]) & cpu_only)
        if selected_cpu_models:
            LOG.info(
                "GPU mode accelerates %s only; these selected models stay on "
                "CPU and keep reusing their CPU artifacts: %s",
                ", ".join(sorted(GPU_CAPABLE_MODELS)), selected_cpu_models,
            )
        # Parallel CV folds each hold their own copy of the training matrix on
        # the device. On a single consumer GPU that is the fastest way to hit
        # an out-of-memory abort, and this project must not silently fall back
        # to CPU, so the search is serialised instead.
        if cfg["hp_search_jobs"] > 1:
            LOG.info(
                "Serialising HP search (hp_search_jobs %d -> 1): concurrent "
                "fits would each need their own copy of the data in VRAM",
                cfg["hp_search_jobs"],
            )
            cfg["hp_search_jobs"] = 1

    outdir = result_run_dir(str(cfg["run_name"]), results_root=Path(cfg["results_root"]))
    outdir.mkdir(parents=True, exist_ok=True)
    LOG.info("Artefacts -> %s", outdir)

    # --- Stage 1: load + clean (cached) ---------------------------------
    t0 = time.time()

    # The manifest is read before the corpus because a temporal manifest names
    # the dataset it applies to, which selects the cache file to load.
    split_manifest: dict[str, Any] | None = None
    manifest_is_temporal = False
    if cfg.get("split_manifest"):
        manifest_path = Path(cfg["split_manifest"])
        peeked = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_is_temporal = is_temporal_manifest(peeked)
        if not manifest_is_temporal:
            raise SystemExit(
                f"{manifest_path} is not a CICIDS2017 temporal manifest "
                f"(version={peeked.get('version')!r}). The source-held-out "
                "protocol was removed with CSE-CIC-IDS2018 support."
            )
        split_manifest = load_temporal_manifest(manifest_path)

    df = load_and_clean_cached(cfg, force=args["refresh_cache"])

    # Drop any class unable to support a train/test split. This happens only
    # for malformed/tiny external corpora; the combined CICIDS cache has at
    # least 11 rows for every retained class.
    counts = df[cfg["label_column"]].value_counts()
    too_small = counts[counts < cfg["min_test_per_class"] + 1].index.tolist()
    if too_small:
        LOG.warning("Dropping classes with < %d total rows post-subsample: %s",
                    cfg["min_test_per_class"] + 1, too_small)
        df = df[~df[cfg["label_column"]].isin(too_small)].reset_index(drop=True)

    cfg["data_fingerprint"] = _dataset_fingerprint(df, cfg)

    LOG.info("Full cleaned label distribution:\n%s",
             df[cfg["label_column"]].value_counts().to_string())

    # --- Stage 2/3: natural holdout first; imbalance handling on TRAIN ---
    if manifest_is_temporal:
        assert split_manifest is not None
        cfg["split_protocol"] = split_manifest["version"]
        # Prove the ordering column really is chronological before trusting it
        # to define train/test. Cheap: one groupby over metadata columns.
        chronology = validate_capture_chronology(
            df, label_column=cfg["label_column"], order_column=ROW_INDEX_COLUMN,
        )
        LOG.info("Capture chronology validated against the published CIC "
                 "schedule (%d ordered pair(s))", chronology["checked_pairs"])
        train_df, calibration_df, test_df = temporal_source_split(
            df,
            label_column=cfg["label_column"],
            order_column=ROW_INDEX_COLUMN,
            test_size=float(split_manifest["test_size"]),
            min_test_per_class=int(split_manifest["min_test_per_class"]),
        )
        verification = verify_split_against_manifest(
            train_df, test_df, split_manifest, label_column=cfg["label_column"],
        )
        if not verification["valid"]:
            raise SystemExit(
                "Temporal split does not match its manifest:\n  - "
                + "\n  - ".join(verification["mismatches"])
            )
        LOG.info(
            "Using temporal per-source per-class manifest %s (%s); counts "
            "verified against the manifest",
            cfg["split_manifest"], split_manifest["version"],
        )
    else:
        raise SystemExit(
            "--split-manifest is required. The chronological manifest defines "
            "the delivery protocol; there is no second split path. Use "
            "configs/splits/cicids2017_temporal_70_30.json (main.py passes it "
            "by default)."
        )
    LOG.info(
        "Split: train=%d, calibration=%d, test=%d "
        "(imbalance_strategy=%s, target=%s, ratio=%.3f)",
        len(train_df), len(calibration_df), len(test_df), cfg["imbalance_strategy"],
        cfg["target_class"], cfg["target_ratio"],
    )

    # Encode after split so the LabelEncoder sees every present class.
    feature_cols = [
        c for c in df.columns
        if c != cfg["label_column"] and c not in cfg["metadata_columns"]
    ]
    y_train_str = train_df[cfg["label_column"]].to_numpy()
    y_calibration_str = calibration_df[cfg["label_column"]].to_numpy()
    y_test_str  = test_df[cfg["label_column"]].to_numpy()
    label_encoder = LabelEncoder().fit(np.concatenate([
        y_train_str,
        y_calibration_str,
        y_test_str,
    ]))
    class_names = list(label_encoder.classes_)
    if cfg["target_class"] not in class_names:
        raise ValueError(
            f"Configured target class {cfg['target_class']!r} is absent. "
            f"Present classes: {class_names}"
        )
    target_class_index = int(label_encoder.transform([cfg["target_class"]])[0])
    benign_class_index = int(label_encoder.transform(["BENIGN"])[0])
    y_train = label_encoder.transform(y_train_str).astype(np.int64)
    y_calibration = label_encoder.transform(y_calibration_str).astype(np.int64)
    y_test  = label_encoder.transform(y_test_str ).astype(np.int64)
    n_classes = len(class_names)

    # Keep as DataFrame (with float32 cells) so LightGBM's feature_names_in_
    # is consistent between fit and predict (LGBM 4.x warns if predict gets
    # numpy after fit saw names). DataFrame in == DataFrame out everywhere.
    # float32 halves RAM vs float64; CICIDS features have ~7 sig figs of
    # headroom so no precision is lost.
    X_train = train_df[feature_cols].astype(np.float32)
    X_calibration = calibration_df[feature_cols].astype(np.float32)
    X_test  = test_df [feature_cols].astype(np.float32)

    # Per-class N (for honest reporting alongside metrics).
    per_class_n_train = {cls: int((y_train_str == cls).sum()) for cls in class_names}
    per_class_n_calibration = {
        cls: int((y_calibration_str == cls).sum()) for cls in class_names
    }
    per_class_n_test  = {cls: int((y_test_str  == cls).sum()) for cls in class_names}
    LOG.info("Per-class n_train: %s", per_class_n_train)
    LOG.info("Per-class n_calibration: %s", per_class_n_calibration)
    LOG.info("Per-class n_test:  %s", per_class_n_test)

    groups_train: np.ndarray | None = None
    if manifest_is_temporal:
        # GroupKFold by source_file is wrong here: every CICIDS2017 attack
        # class lives in exactly one capture, so grouping by source would put
        # an entire class inside a single fold and leave the others with none
        # of it. CV feeds only HP search and the stability check -- never the
        # headline number, which comes from the locked chronological test set.
        min_train_class = min(per_class_n_train.values())
        eff_splits = max(2, min(cfg["cv_splits"], min_train_class))
        if eff_splits != cfg["cv_splits"]:
            LOG.warning("Clipping CV n_splits %d -> %d (smallest class has %d train rows)",
                        cfg["cv_splits"], eff_splits, min_train_class)
        cv = StratifiedKFold(n_splits=eff_splits, shuffle=True,
                             random_state=cfg["random_state"])
        LOG.info("Using %d-fold StratifiedKFold (source grouping is degenerate "
                 "for a single-capture-per-class corpus)", eff_splits)
    elif cfg.get("split_manifest"):
        groups_train = train_df["source_file"].to_numpy()
        n_groups = len(np.unique(groups_train))
        eff_splits = max(2, min(cfg["cv_splits"], n_groups))
        cv = GroupKFold(n_splits=eff_splits)
        LOG.info("Using %d-fold GroupKFold by source_file (%d train captures)",
                 eff_splits, n_groups)
    else:
        min_train_class = min(per_class_n_train.values())
        eff_splits = max(2, min(cfg["cv_splits"], min_train_class))
        if eff_splits != cfg["cv_splits"]:
            LOG.warning("Clipping CV n_splits %d -> %d (smallest class has %d train rows)",
                        cfg["cv_splits"], eff_splits, min_train_class)
        cv = StratifiedKFold(n_splits=eff_splits, shuffle=True,
                             random_state=cfg["random_state"])

    # Majority-class baseline (same for all models -- depends only on data).
    # The baseline must describe the untouched test distribution, not the
    # targeted/weighted training prior.  Otherwise targeted sampling can make
    # the benchmark appear to have a near-zero majority baseline.
    test_majority_class = int(np.bincount(y_test).argmax())
    majority_baseline = float(np.mean(y_test == test_majority_class))

    def new_pipeline(model_name: str) -> Pipeline | ImbPipeline:
        return build_pipeline(
            model_name,
            n_classes,
            cfg["random_state"],
            rf_class_weight=cfg["rf_class_weight"],
            imbalance_strategy=cfg["imbalance_strategy"],
            target_class_index=target_class_index,
            target_ratio=cfg["target_ratio"],
            accelerator=cfg["accelerator"],
            gpu_devices=cfg["gpu_devices"],
        )

    # --- Stage 4: per-model train + evaluate (skip-if-exists) -----------
    corpus_label = DATASET_ID
    results: list[EvalResult] = []
    for model_name in cfg["models"]:
        LOG.info("=" * 64)
        LOG.info("Model: %s", model_name)
        model_path = outdir / f"{model_name}.joblib"
        per_model_metrics = outdir / f"{model_name}_metrics.json"

        saved: dict[str, Any] | None = None
        if model_path.exists() and per_model_metrics.exists():
            saved = json.loads(per_model_metrics.read_text(encoding="utf-8"))
        can_reuse = (
            saved is not None
            and not args["force"]
            and _imbalance_config_matches(saved, cfg, model_name)
        )
        if can_reuse:
            LOG.info("Found matching existing %s; skipping fit (use --force to retrain)",
                     model_path.name)
            pipeline = joblib.load(model_path)
            if args["refresh_plots"]:
                plot_X_test = X_test
                fitted_features = getattr(pipeline, "feature_names_in_", None)
                if fitted_features is not None:
                    plot_X_test = X_test.loc[:, list(fitted_features)]
                acc, bacc, f1m, f1w, per_class_df, cm = evaluate(
                    pipeline, plot_X_test, y_test, class_names,
                )
                refreshed_target = target_metrics_from_confusion(
                    cm,
                    target_class_index=target_class_index,
                    benign_class_index=benign_class_index,
                )
                refreshed_extended = extended_metrics_from_confusion(
                    cm, class_names,
                    benign_class_index=benign_class_index,
                    reportable_min_test=cfg["reportable_min_test"],
                )
                per_class_df.to_csv(
                    outdir / f"{model_name}_per_class.csv", index=True
                )
                plot_confusion_matrix(
                    cm, class_names,
                    outdir / f"{model_name}_confusion_matrix.png",
                    title=f"{model_name} -- {corpus_label} test set",
                )
                saved.update({
                    "accuracy": acc,
                    "balanced_accuracy": bacc,
                    "f1_macro": f1m,
                    "f1_weighted": f1w,
                    "majority_baseline_acc": majority_baseline,
                    "near_perfect_flag": acc >= cfg["near_perfect_threshold"],
                    **refreshed_target,
                    **refreshed_extended,
                    **_imbalance_metadata(cfg, model_name),
                })
                LOG.info("Refreshed %s metrics + confusion matrix", model_name)
            assert saved is not None
            # Normalize legacy NaN values to JSON ``null`` even on a resumed run.
            per_model_metrics.write_text(
                json_dumps_strict(saved, indent=2), encoding="utf-8"
            )
            results.append(_eval_result_from_saved(saved, model_name))
            continue
        if saved is not None and not args["force"]:
            LOG.info(
                "Existing %s uses different imbalance settings; retraining",
                model_path.name,
            )

        # ----- Checkpoint / HP search / final fit --------------------
        checkpoint_path = outdir / "checkpoints" / f"{model_name}.json"
        run_signature = _checkpoint_signature(cfg, model_name)
        checkpoint = load_checkpoint(checkpoint_path)
        resume_ready = (
            not args["force"]
            and model_path.exists()
            and checkpoint_matches(
                checkpoint, model_name=model_name, run_signature=run_signature,
            )
            and checkpoint.get("phase") == "model_ready"
        )
        threshold_calibration: ThresholdCalibration | None = None
        # None on a resumed run: the fit happened in an earlier process, so
        # reporting a fit time here would be a fabrication.
        fit_seconds: float | None = None
        peak_rss_before = peak_rss_mb()
        if resume_ready:
            pipeline = joblib.load(model_path)
            best_params = dict(checkpoint.get("best_params", {}))
            LOG.info("Resuming %s from model-ready checkpoint", model_name)
        else:
            best_params: dict[str, Any] = (
                dict(saved.get("best_params", {}))
                if args["reuse_best_params"] and saved is not None
                else {}
            )
            if args["reuse_best_params"] and not best_params:
                raise ValueError(
                    f"No saved best_params are available for {model_name}"
                )
            pipeline = new_pipeline(model_name)
            if best_params:
                pipeline.set_params(**best_params)
                LOG.info("Reusing saved best params: %s", best_params)

            if cfg["hp_search"] and hp_grids(model_name):
                X_hp, y_hp = _hp_subset(
                    X_train, y_train,
                    target_n=cfg["hp_search_subsample"],
                    random_state=cfg["random_state"],
                )
                LOG.info("HP search on %d-row train subset "
                         "(n_iter=%d, cv=%d, scoring=%s)",
                         len(y_hp), cfg["hp_search_n_iter"],
                         eff_splits, cfg["primary_metric"])
                min_hp_class = min(int((y_hp == c).sum()) for c in range(n_classes))
                hp_splits = max(2, min(eff_splits, min_hp_class))
                search = RandomizedSearchCV(
                    pipeline,
                    param_distributions=hp_grids(model_name),
                    n_iter=cfg["hp_search_n_iter"],
                    scoring=search_scorer(cfg["primary_metric"], target_class_index),
                    cv=StratifiedKFold(n_splits=hp_splits, shuffle=True,
                                       random_state=cfg["random_state"]),
                    n_jobs=cfg["hp_search_jobs"], refit=False,
                    random_state=cfg["random_state"], verbose=0,
                    return_train_score=False,
                )
                search.fit(X_hp, y_hp)
                best_params = search.best_params_
                write_checkpoint(checkpoint_path, {
                    "model": model_name, "run_signature": run_signature,
                    "phase": "hp_complete", "best_params": best_params,
                })
                LOG.info("Best params on HP subset: %s", best_params)
                pipeline = new_pipeline(model_name)
                pipeline.set_params(**best_params)

            write_checkpoint(checkpoint_path, {
                "model": model_name, "run_signature": run_signature,
                "phase": "fitting", "best_params": best_params,
            })
            t_fit = time.perf_counter()
            pipeline.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - t_fit
            LOG.info("%s final fit on full train (%d rows): %.1fs",
                     model_name, len(y_train), fit_seconds)

            if len(y_calibration):
                t_calibration = time.time()
                calibration_probabilities = np.asarray(
                    pipeline.predict_proba(X_calibration)
                )[:, target_class_index]
                threshold_calibration = calibrate_target_threshold(
                    y_calibration, calibration_probabilities,
                    target_class_index=target_class_index,
                    max_false_positive_rate=cfg["target_max_fpr"],
                )
                pipeline.set_params(target_threshold=threshold_calibration.threshold)
                LOG.info("%s threshold calibration: threshold=%.6f, recall=%.4f, FPR=%.4f (%.1fs)",
                         model_name, threshold_calibration.threshold,
                         threshold_calibration.recall,
                         threshold_calibration.false_positive_rate,
                         time.time() - t_calibration)
            else:
                LOG.info("%s: no calibration partition; retaining native argmax decision policy", model_name)

            # A crash during CV/plots can now resume from this fitted artifact.
            atomic_joblib_dump(pipeline, model_path)
            write_checkpoint(checkpoint_path, {
                "model": model_name, "run_signature": run_signature,
                "phase": "model_ready", "best_params": best_params,
            })

        # ----- Evaluate ----------------------------------------------
        acc, bacc, f1m, f1w, per_class_df, cm = evaluate(
            pipeline, X_test, y_test, class_names,
        )
        target_metrics = target_metrics_from_confusion(
            cm,
            target_class_index=target_class_index,
            benign_class_index=benign_class_index,
        )
        extended = extended_metrics_from_confusion(
            cm, class_names,
            benign_class_index=benign_class_index,
            reportable_min_test=cfg["reportable_min_test"],
        )
        LOG.info("%s test: acc=%.4f, bal_acc=%.4f, f1_macro=%.4f, "
                 "f1_weighted=%.4f", model_name, acc, bacc, f1m, f1w)
        LOG.info(
            "%s detection: macro P=%.4f R=%.4f | binary FPR=%.5f (%d benign "
            "alerts), FNR=%.5f (%d attacks missed) | f1_macro_reportable=%.4f",
            model_name,
            extended["precision_macro"], extended["recall_macro"],
            extended["binary_fpr"], extended["binary_false_positives"],
            extended["binary_fnr"], extended["binary_false_negatives"],
            extended["f1_macro_reportable"],
        )
        LOG.info(
            "%s target test: recall=%.4f, F2=%.4f, FPR=%.4f, "
            "FN=%d (to BENIGN=%d), FP=%d",
            model_name,
            target_metrics["target_recall"],
            target_metrics["target_f2"],
            target_metrics["target_fpr"],
            target_metrics["target_false_negatives"],
            target_metrics["target_to_benign_fn"],
            target_metrics["target_false_positives"],
        )

        # ----- CV trust check (TRAIN only) ---------------------------
        cv_scores: list[float] = []
        cv_mean: float | None = None
        cv_std: float | None = None
        if cfg["cv_check"]:
            cv_scores, cv_mean, cv_std = cross_validate_clean(
                new_pipeline(model_name),
                X_train, y_train, cv, groups=groups_train,
            )
            LOG.info("%s %d-fold CV f1_macro: %.4f +/- %.4f",
                     model_name, eff_splits, cv_mean, cv_std)
        else:
            LOG.info("%s CV trust check skipped", model_name)

        # ----- Label-shuffle sanity ----------------------------------
        shuf_acc: float | None = None
        shuf_f1m: float | None = None
        if cfg["label_shuffle_check"]:
            shuf_acc, shuf_f1m = label_shuffle_sanity(
                new_pipeline(model_name),
                X_train, y_train, cfg["random_state"],
                min_test_per_class=cfg["min_test_per_class"],
            )
            LOG.info(
                "%s label-shuffle: acc=%.4f (majority=%.4f), "
                "f1_macro=%.4f (chance=%.4f)",
                model_name, shuf_acc, majority_baseline,
                shuf_f1m, 1.0 / n_classes,
            )

        # ----- Persist ------------------------------------------------
        atomic_joblib_dump(pipeline, model_path)

        # ----- Dimension 4: computational efficiency ------------------
        # Measured after the artifact is on disk so model_size_mb is the real
        # deployed size, and after CV/shuffle so those refits cannot inflate
        # the latency sample through cache pressure.
        efficiency: dict[str, Any] = {
            "fit_seconds": fit_seconds,
            "model_size_mb": model_path.stat().st_size / (1024.0 ** 2),
            **measure_inference_cost(
                pipeline, X_test,
                batch_size=cfg["latency_batch_size"],
                repeats=cfg["latency_repeats"],
                random_state=cfg["random_state"],
            ),
        }
        peak_rss_after = peak_rss_mb()
        if peak_rss_before is not None and peak_rss_after is not None:
            efficiency["process_rss_mb"] = peak_rss_after
            efficiency["fit_rss_delta_mb"] = peak_rss_after - peak_rss_before
        LOG.info(
            "%s efficiency: fit=%s, p50=%.2fms, p95=%.2fms, %.0f flows/s, "
            "%.1f MB on disk",
            model_name,
            f"{fit_seconds:.1f}s" if fit_seconds is not None else "resumed",
            efficiency.get("predict_latency_p50_ms", float("nan")),
            efficiency.get("predict_latency_p95_ms", float("nan")),
            efficiency.get("throughput_flows_per_sec", float("nan")),
            efficiency["model_size_mb"],
        )

        per_class_df.to_csv(outdir / f"{model_name}_per_class.csv", index=True)
        plot_confusion_matrix(
            cm, class_names,
            outdir / f"{model_name}_confusion_matrix.png",
            title=f"{model_name} -- {corpus_label} test set",
        )

        r = EvalResult(
            model=model_name,
            accuracy=acc, balanced_accuracy=bacc,
            f1_macro=f1m, f1_weighted=f1w,
            per_class=per_class_df, confusion=cm,
            cv_scores=cv_scores, cv_mean=cv_mean, cv_std=cv_std,
            shuffle_accuracy=shuf_acc, shuffle_f1_macro=shuf_f1m,
            majority_baseline=majority_baseline,
            best_params=best_params,
            target_threshold=(
                threshold_calibration.threshold
                if threshold_calibration is not None else None
            ),
            target_precision=float(target_metrics["target_precision"]),
            target_recall=float(target_metrics["target_recall"]),
            target_f1=float(target_metrics["target_f1"]),
            target_f2=float(target_metrics["target_f2"]),
            target_fpr=float(target_metrics["target_fpr"]),
            target_false_positives=int(
                target_metrics["target_false_positives"]
            ),
            target_false_negatives=int(
                target_metrics["target_false_negatives"]
            ),
            target_to_benign_fn=int(target_metrics["target_to_benign_fn"]),
            extended=extended,
            efficiency=efficiency,
            hp_space_size=hp_search_space_size(model_name),
            hp_tuned=bool(cfg["hp_search"] and hp_grids(model_name)),
            calibration_recall=(
                threshold_calibration.recall
                if threshold_calibration is not None else None
            ),
            calibration_fpr=(
                threshold_calibration.false_positive_rate
                if threshold_calibration is not None else None
            ),
        )
        results.append(r)

        # Per-model metrics JSON (used by skip-if-exists logic on rerun).
        per_model_metrics.write_text(
            json_dumps_strict(_eval_result_to_dict(r, cfg), indent=2),
            encoding="utf-8",
        )
        write_checkpoint(checkpoint_path, {
            "model": model_name,
            "run_signature": run_signature,
            "phase": "complete",
            "best_params": best_params,
        })

    requested_models = set(cfg["models"])
    for model_name in CONFIG["models"]:
        if model_name in requested_models:
            continue
        per_model_metrics = outdir / f"{model_name}_metrics.json"
        if per_model_metrics.exists():
            saved = json.loads(per_model_metrics.read_text(encoding="utf-8"))
            if _imbalance_config_matches(saved, cfg, model_name):
                results.append(_eval_result_from_saved(saved, model_name))

    # --- Stage 5: shared artefacts + aggregate report -------------------
    joblib.dump(label_encoder, outdir / "label_encoder.joblib")
    (outdir / "feature_columns.json").write_text(
        json.dumps(feature_cols, indent=2), encoding="utf-8")

    metrics_payload = {
        "run_name":         cfg["run_name"],
        "random_state":     cfg["random_state"],
        **_imbalance_metadata(cfg),
        "n_train":          int(len(y_train)),
        "n_calibration":    int(len(y_calibration)),
        "n_test":           int(len(y_test)),
        "n_features":       int(X_train.shape[1]),
        "n_classes":        n_classes,
        "class_names":      class_names,
        "per_class_n_train": per_class_n_train,
        "per_class_n_calibration": per_class_n_calibration,
        "per_class_n_test":  per_class_n_test,
        "majority_baseline_acc": majority_baseline,
        "duration_seconds": round(time.time() - t0, 2),
        "split_manifest": str(cfg["split_manifest"]) if cfg.get("split_manifest") else None,
        "split_protocol": cfg["split_protocol"],
        "hardware": hardware_profile(cfg),
        "hp_search_space_size": HP_SEARCH_SPACE_SIZE,
        "hp_untuned_models": HP_UNTUNED_MODELS,
        "models": [_eval_result_to_dict(r, cfg) for r in results],
    }
    (outdir / "metrics.json").write_text(
        json_dumps_strict(metrics_payload, indent=2), encoding="utf-8")

    report_path = write_report(
        results, outdir, cfg, n_classes, class_names,
        per_class_n_test, per_class_n_train,
    )
    LOG.info("Wrote report -> %s", report_path)
    bundle_files = [p for p in outdir.iterdir() if p.is_file() and p.name != "bundle_manifest.json"]
    bundle_manifest = build_bundle_manifest(
        outdir,
        bundle_files,
        run_id=cfg["run_name"],
        metadata={
            "split_protocol": metrics_payload["split_protocol"],
            "class_names": class_names,
        },
    )
    write_bundle_manifest(outdir / "bundle_manifest.json", bundle_manifest)
    LOG.info("Wrote integrity manifest -> %s", outdir / "bundle_manifest.json")
    LOG.info("Done in %.1fs. All artefacts under %s",
             time.time() - t0, outdir)
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def effective_accelerator(cfg: dict[str, Any],
                          model_name: str | None = None) -> str:
    """The accelerator that actually applies to one model.

    ``--accelerator gpu`` is a run-level request, but only
    :data:`GPU_CAPABLE_MODELS` act on it. Recording the *effective* value per
    model keeps artifact identity honest: a RandomForest trained during a GPU
    run is the same artifact as one trained during a CPU run, and must stay
    reusable across both.

    Called with ``model_name=None`` it returns the run-level request, which is
    what the aggregate metrics header should record.
    """
    requested = cfg.get("accelerator", "cpu")
    if model_name is None or model_name in GPU_CAPABLE_MODELS:
        return requested
    return "cpu"


def _imbalance_metadata(cfg: dict[str, Any],
                        model_name: str | None = None) -> dict[str, Any]:
    return {
        "training_protocol_version": TRAINING_PROTOCOL_VERSION,
        "imbalance_protocol_version": IMBALANCE_PROTOCOL_VERSION,
        "data_fingerprint": cfg.get("data_fingerprint"),
        # The split identity has to be part of artifact identity: the dataset
        # fingerprint is computed on the pre-split frame, so without these two
        # a run under a different split protocol would silently reuse models
        # trained on a completely different train/test partition.
        "split_protocol": cfg.get("split_protocol"),
        "dataset_id": DATASET_ID,
        "random_state": cfg["random_state"],
        "primary_metric": cfg["primary_metric"],
        "rf_class_weight": cfg["rf_class_weight"],
        "hp_search": cfg["hp_search"],
        "hp_search_n_iter": cfg["hp_search_n_iter"],
        "hp_search_subsample": cfg["hp_search_subsample"],
        "hp_grid_fingerprint": (
            hp_grid_fingerprint(model_name) if model_name else None
        ),
        "imbalance_strategy": cfg["imbalance_strategy"],
        "target_class": cfg["target_class"],
        "target_ratio": cfg["target_ratio"],
        "target_max_fpr": cfg["target_max_fpr"],
        "threshold_validation_size": cfg["threshold_validation_size"],
        # What this model actually trained on, not what the run asked for.
        "accelerator": effective_accelerator(cfg, model_name),
        "requested_accelerator": cfg["accelerator"],
        "gpu_devices": cfg["gpu_devices"],
    }


def _checkpoint_signature(cfg: dict[str, Any], model_name: str) -> str:
    """Bind a checkpoint to one model, dataset fingerprint, and train policy.

    Only fields that determine the artifact take part. ``requested_accelerator``
    is provenance -- a RandomForest is identical whether or not the run asked
    for a GPU -- and ``gpu_devices`` is irrelevant to a model that never
    touches the GPU. Including either would break mid-run resume for CPU-only
    models the moment the flag changed.
    """
    metadata = _imbalance_metadata(cfg, model_name)
    metadata.pop("requested_accelerator", None)
    if metadata.get("accelerator") != "gpu":
        metadata.pop("gpu_devices", None)
    payload = {
        "model": model_name,
        "split_manifest": str(cfg.get("split_manifest") or ""),
        **metadata,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _imbalance_config_matches(saved: dict[str, Any], cfg: dict[str, Any],
                              model_name: str | None = None) -> bool:
    """Whether a saved artifact was produced by an equivalent configuration.

    ``model_name`` lets the accelerator be compared per model. Without it, a
    CPU->GPU switch would discard every finished CPU-only model even though
    GPU cannot change their output. Artifacts written before this distinction
    existed carry no ``requested_accelerator`` key and compare on their plain
    ``accelerator`` value, which is exactly the effective one.
    """
    if saved.get("training_protocol_version") != TRAINING_PROTOCOL_VERSION:
        return False
    wanted_accelerator = effective_accelerator(cfg, model_name)
    if saved.get("accelerator", "cpu") != wanted_accelerator:
        return False
    # GPU device selection only matters to a model that actually used the GPU.
    if (
        wanted_accelerator == "gpu"
        and saved.get("gpu_devices", "0") != cfg["gpu_devices"]
    ):
        return False
    # An edited search space means the saved model was tuned against a
    # different set of candidates. Absent on artifacts written before this key
    # existed, which are left alone rather than force-retrained wholesale.
    saved_grid = saved.get("hp_grid_fingerprint")
    if (
        saved_grid is not None
        and model_name is not None
        and saved.get("hp_search", False)
        and saved_grid != hp_grid_fingerprint(model_name)
    ):
        return False
    if saved.get("imbalance_protocol_version") != IMBALANCE_PROTOCOL_VERSION:
        return False
    saved_strategy = saved.get("imbalance_strategy", "class_weight")
    if saved_strategy != cfg["imbalance_strategy"]:
        return False
    return (
        saved.get("data_fingerprint") == cfg.get("data_fingerprint")
        and saved.get("split_protocol") == cfg.get("split_protocol")
        and saved.get("random_state") == cfg["random_state"]
        and saved.get("primary_metric") == cfg["primary_metric"]
        and saved.get("rf_class_weight") == cfg["rf_class_weight"]
        and saved.get("hp_search") == cfg["hp_search"]
        and saved.get("hp_search_n_iter") == cfg["hp_search_n_iter"]
        and saved.get("hp_search_subsample") == cfg["hp_search_subsample"]
        and saved.get("target_class") == cfg["target_class"]
        and float(saved.get("target_ratio", -1.0)) == float(cfg["target_ratio"])
        and float(saved.get("target_max_fpr", -1.0))
        == float(cfg["target_max_fpr"])
        and float(saved.get("threshold_validation_size", -1.0))
        == float(cfg["threshold_validation_size"])
    )


def _dataset_fingerprint(df: pd.DataFrame, cfg: dict[str, Any]) -> str:
    """Cheap, deterministic identity for artifact-reuse safety.

    The parquet stat catches cache refreshes without hashing a multi-GB file;
    schema and label counts protect against accidental cache replacement.
    """
    cache_path = resolve_cache_path(cfg)
    cache_stat: dict[str, int | str] = {"path": str(cache_path.resolve())}
    if cache_path.exists():
        stat = cache_path.stat()
        cache_stat.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    label_col = cfg["label_column"]
    payload = {
        "cache": cache_stat,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": [str(column) for column in df.columns],
        "class_counts": {
            str(name): int(count)
            for name, count in df[label_col].value_counts().sort_index().items()
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _hp_subset(X: pd.DataFrame, y: np.ndarray, target_n: int,
               random_state: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Sub-sample the TRAIN set for HP search; keeps all rows of any
    class with <= 200 samples (so HP search still sees minority classes),
    proportional for the rest. X is a DataFrame; we use .iloc for
    positional indexing so feature_names_in_ stays consistent."""
    if target_n >= len(y):
        return X, y
    rng = np.random.default_rng(random_state)
    keep_idx: list[int] = []
    classes, counts = np.unique(y, return_counts=True)
    rare_mask = counts <= 200
    rare_classes = classes[rare_mask]
    common_classes = classes[~rare_mask]
    rare_idx = np.flatnonzero(np.isin(y, rare_classes))
    keep_idx.extend(rare_idx.tolist())
    budget = max(0, target_n - len(rare_idx))
    common_idx = np.flatnonzero(np.isin(y, common_classes))
    if budget > 0 and len(common_idx) > 0:
        for cls in common_classes:
            grp = np.flatnonzero(y == cls)
            quota = max(1, round(budget * len(grp) / len(common_idx)))
            take = min(quota, len(grp))
            keep_idx.extend(rng.choice(grp, size=take, replace=False).tolist())
    keep_idx_arr = np.asarray(keep_idx, dtype=np.int64)
    rng.shuffle(keep_idx_arr)
    return X.iloc[keep_idx_arr].reset_index(drop=True), y[keep_idx_arr]


def atomic_joblib_dump(obj: Any, path: Path) -> None:
    """Write a joblib artifact via a same-directory temp file.

    Windows can be fussy when overwriting large files that were recently
    read by Streamlit or another Python process. Dumping to a fresh path
    first avoids opening the old artifact for writing until the final swap.
    """
    tmp = path.with_name(f"{path.name}.tmp")
    if tmp.exists():
        tmp.unlink()
    joblib.dump(obj, tmp)
    os.replace(tmp, path)


def _eval_result_to_dict(r: EvalResult, cfg: dict[str, Any]) -> dict[str, Any]:
    """Serialise one EvalResult for both the per-model and aggregate JSON.

    Single source of truth for the EvalResult <-> JSON mapping, whose field
    names deliberately differ from the dataclass attribute names for backward
    compatibility (``cv_mean`` -> ``cv_f1_macro_mean`` and friends). Keeping
    one writer means :func:`_eval_result_from_saved` can be its exact inverse.

    Keys are only ever added here, never renamed: ``f1_macro`` and
    ``target_fpr`` drive champion promotion in ``src/artifacts/publish.py``,
    and several readers use ``.get(key, 0.0)``, so a rename would silently
    promote the wrong model rather than fail loudly.
    """
    payload: dict[str, Any] = {
        "model":                  r.model,
        "accuracy":               r.accuracy,
        "balanced_accuracy":      r.balanced_accuracy,
        "f1_macro":               r.f1_macro,
        "f1_weighted":            r.f1_weighted,
        "cv_f1_macro_mean":       r.cv_mean,
        "cv_f1_macro_std":        r.cv_std,
        "cv_f1_macro_scores":     r.cv_scores,
        "label_shuffle_acc":      r.shuffle_accuracy,
        "label_shuffle_f1_macro": r.shuffle_f1_macro,
        "majority_baseline_acc":  r.majority_baseline,
        "best_params":            r.best_params,
        "target_threshold":       r.target_threshold,
        "target_precision":       r.target_precision,
        "target_recall":          r.target_recall,
        "target_f1":              r.target_f1,
        "target_f2":              r.target_f2,
        "target_fpr":             r.target_fpr,
        "target_false_positives": r.target_false_positives,
        "target_false_negatives": r.target_false_negatives,
        "target_to_benign_fn":    r.target_to_benign_fn,
        "calibration_recall":     r.calibration_recall,
        "calibration_fpr":        r.calibration_fpr,
        "near_perfect_flag":      r.accuracy >= cfg["near_perfect_threshold"],
        "hp_search_space_size":   r.hp_space_size,
        "hp_tuned":               r.hp_tuned,
    }
    payload.update({k: v for k, v in r.extended.items() if k in EXTENDED_METRIC_KEYS})
    payload.update({k: v for k, v in r.efficiency.items() if k in EFFICIENCY_KEYS})
    payload.update(_imbalance_metadata(cfg, r.model))
    return payload


def _eval_result_from_saved(saved: dict, model_name: str) -> EvalResult:
    """Reconstruct EvalResult from a previously-saved metrics JSON so the
    report aggregator can include it without re-evaluating.

    Inverse of :func:`_eval_result_to_dict`. ``per_class`` and ``confusion``
    cannot be recovered from the JSON, so a reused model contributes no
    per-class table or confusion matrix -- callers must tolerate that.
    """
    empty_df = pd.DataFrame()
    empty_cm = np.zeros((1, 1), dtype=np.int64)
    return EvalResult(
        model=saved.get("model", model_name),
        accuracy=saved.get("accuracy", 0.0),
        balanced_accuracy=saved.get("balanced_accuracy", 0.0),
        f1_macro=saved.get("f1_macro", 0.0),
        f1_weighted=saved.get("f1_weighted", 0.0),
        per_class=empty_df, confusion=empty_cm,
        cv_scores=saved.get("cv_f1_macro_scores", []),
        cv_mean=saved.get("cv_f1_macro_mean", 0.0),
        cv_std=saved.get("cv_f1_macro_std", 0.0),
        shuffle_accuracy=saved.get("label_shuffle_acc"),
        shuffle_f1_macro=saved.get("label_shuffle_f1_macro"),
        majority_baseline=saved.get("majority_baseline_acc", 0.0),
        best_params=saved.get("best_params", {}),
        target_threshold=saved.get("target_threshold"),
        target_precision=saved.get("target_precision", 0.0),
        target_recall=saved.get("target_recall", 0.0),
        target_f1=saved.get("target_f1", 0.0),
        target_f2=saved.get("target_f2", 0.0),
        target_fpr=saved.get("target_fpr", 0.0),
        target_false_positives=saved.get("target_false_positives", 0),
        target_false_negatives=saved.get("target_false_negatives", 0),
        target_to_benign_fn=saved.get("target_to_benign_fn", 0),
        calibration_recall=saved.get("calibration_recall"),
        calibration_fpr=saved.get("calibration_fpr"),
        extended={k: saved[k] for k in EXTENDED_METRIC_KEYS if k in saved},
        efficiency={k: saved[k] for k in EFFICIENCY_KEYS if k in saved},
        hp_space_size=saved.get("hp_search_space_size", 0),
        hp_tuned=saved.get("hp_tuned", False),
    )


if __name__ == "__main__":
    raise SystemExit(main())
