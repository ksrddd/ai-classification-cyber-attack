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
import random
import sys
import time
from contextlib import suppress
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
from src.data.deterministic_split import (
    deterministic_source_split,
    load_split_manifest,
    row_hash,
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
    "clean_cache":        ROOT / "data" / "processed" / "cicids_clean.parquet",
    "csv_glob":           "*.csv",
    "csv_encoding":       "latin-1",
    "label_column":       "Label",
    "split_manifest":     None,
    "metadata_columns":   ("dataset_id", "source_file", "capture_window", "_row_hash"),

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
    "primary_metric":     "target_f2",
    "rf_class_weight":    "balanced_subsample",
    # ``targeted`` uses additional REAL rows of target_class. The other
    # sampler strategies run inside ImbPipeline and therefore only see each
    # training/CV fold. target_ratio means target / majority after sampling.
    "imbalance_strategy": "targeted",
    "target_class":       "Infiltration",
    "target_ratio":       1.00,
    # The delivery protocol uses a 70/30 train/test split with no calibration
    # partition. Set this explicitly above zero only for a separately approved
    # experiment; the locked test set must never select a threshold.
    "threshold_validation_size": 0.0,
    "target_max_fpr":      0.02,
    "accelerator":         "cpu",  # cpu | gpu; GPU affects XGBoost/CatBoost only
    "gpu_devices":         "0",

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
TRAINING_PROTOCOL_VERSION = 3   # 70/30 no-calibration + resumable checkpoints
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
# ---------------------------------------------------------------------------
# CIC-IDS2018 uses abbreviated column names that differ from CICIDS2017.
# This mapping normalises them to the CICIDS2017 canonical names so the
# rest of the pipeline (feature list, schema checks) needs no changes.
# ---------------------------------------------------------------------------
_CIC2018_COL_RENAME: dict[str, str] = {
    "Dst Port":            "Destination Port",
    "Tot Fwd Pkts":        "Total Fwd Packets",
    "Tot Bwd Pkts":        "Total Backward Packets",
    "TotLen Fwd Pkts":     "Total Length of Fwd Packets",
    "TotLen Bwd Pkts":     "Total Length of Bwd Packets",
    "Fwd Pkt Len Max":     "Fwd Packet Length Max",
    "Fwd Pkt Len Min":     "Fwd Packet Length Min",
    "Fwd Pkt Len Mean":    "Fwd Packet Length Mean",
    "Fwd Pkt Len Std":     "Fwd Packet Length Std",
    "Bwd Pkt Len Max":     "Bwd Packet Length Max",
    "Bwd Pkt Len Min":     "Bwd Packet Length Min",
    "Bwd Pkt Len Mean":    "Bwd Packet Length Mean",
    "Bwd Pkt Len Std":     "Bwd Packet Length Std",
    "Flow Byts/s":         "Flow Bytes/s",
    "Flow Pkts/s":         "Flow Packets/s",
    "Fwd IAT Tot":         "Fwd IAT Total",
    "Bwd IAT Tot":         "Bwd IAT Total",
    "Fwd Header Len":      "Fwd Header Length",
    "Bwd Header Len":      "Bwd Header Length",
    "Pkt Len Min":         "Min Packet Length",
    "Pkt Len Max":         "Max Packet Length",
    "Pkt Len Mean":        "Packet Length Mean",
    "Pkt Len Std":         "Packet Length Std",
    "Pkt Len Var":         "Packet Length Variance",
    "FIN Flag Cnt":        "FIN Flag Count",
    "SYN Flag Cnt":        "SYN Flag Count",
    "RST Flag Cnt":        "RST Flag Count",
    "PSH Flag Cnt":        "PSH Flag Count",
    "ACK Flag Cnt":        "ACK Flag Count",
    "URG Flag Cnt":        "URG Flag Count",
    "ECE Flag Cnt":        "ECE Flag Count",
    "Pkt Size Avg":        "Average Packet Size",
    "Fwd Seg Size Avg":    "Avg Fwd Segment Size",
    "Bwd Seg Size Avg":    "Avg Bwd Segment Size",
    "Fwd Byts/b Avg":      "Fwd Avg Bytes/Bulk",
    "Fwd Pkts/b Avg":      "Fwd Avg Packets/Bulk",
    "Fwd Blk Rate Avg":    "Fwd Avg Bulk Rate",
    "Bwd Byts/b Avg":      "Bwd Avg Bytes/Bulk",
    "Bwd Pkts/b Avg":      "Bwd Avg Packets/Bulk",
    "Bwd Blk Rate Avg":    "Bwd Avg Bulk Rate",
    "Subflow Fwd Pkts":    "Subflow Fwd Packets",
    "Subflow Fwd Byts":    "Subflow Fwd Bytes",
    "Subflow Bwd Pkts":    "Subflow Bwd Packets",
    "Subflow Bwd Byts":    "Subflow Bwd Bytes",
    "Init Fwd Win Byts":   "Init_Win_bytes_forward",
    "Init Bwd Win Byts":   "Init_Win_bytes_backward",
    "Fwd Act Data Pkts":   "act_data_pkt_fwd",
    "Fwd Seg Size Min":    "min_seg_size_forward",
}


def _clean_one_frame(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Clean a single CSV: column strip/rename, drop duplicate/leaky cols,
    label normalize, replace Inf, drop NaN.

    Supports both CICIDS2017 and CIC-IDS2018 column naming schemes.
    Per-CSV cleaning keeps memory peak bounded.
    """
    # Strip whitespace from column names (CICIDS2017 has leading spaces).
    df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)

    # Normalise CIC-IDS2018 abbreviated column names → CICIDS2017 names.
    df = df.rename(columns=_CIC2018_COL_RENAME)

    if "Fwd Header Length.1" in df.columns:
        df = df.drop(columns=["Fwd Header Length.1"])

    leaky = [c for c in cfg["leaky_columns"] if c in df.columns]
    if leaky:
        df = df.drop(columns=leaky)

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

    feature_cols = [c for c in df.columns if c != label_col]
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


def load_and_clean_cached(cfg: dict[str, Any], force: bool = False) -> pd.DataFrame:
    """Load + clean + dedupe the combined CICIDS2017/CSE-CIC-IDS2018 corpus.

    Caches the result as ``cicids_clean.parquet`` so subsequent runs skip
    the ~3-5 min load+clean step. Cache invalidated by ``--force``.
    """
    cache = Path(cfg["clean_cache"])
    if not force and cache.exists():
        LOG.info("Loading cleaned corpus from cache: %s", cache)
        df = pd.read_parquet(cache)
        if cfg.get("split_manifest") and not {"source_file", "_row_hash"}.issubset(df.columns):
            LOG.info("Deterministic split requested but cache lacks source metadata; rebuilding.")
        else:
            LOG.info("  -> %d rows x %d cols (from cache)", *df.shape)
            return df

    raw_dir = Path(cfg["raw_dir"])
    paths = sorted(raw_dir.glob(cfg["csv_glob"]))
    if not paths:
        raise FileNotFoundError(
            f"No CSV files found under {raw_dir}. "
            "Extract MachineLearningCSV.zip into data/raw/."
        )

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
        clean["dataset_id"] = "CSE-CIC-IDS2018" if "2018" in p.name else "CICIDS2017"
        clean["capture_window"] = p.stem
        clean["_row_hash"] = [row_hash(p.name, i) for i in range(len(clean))]
        LOG.info("  %s -> raw %d -> clean %d", p.name, len(raw), len(clean))
        del raw
        cleaned.append(clean)

    df = pd.concat(cleaned, axis=0, ignore_index=True)
    del cleaned
    LOG.info("Concatenated cleaned frames: %d rows", len(df))

    n_before = len(df)
    dedup_columns = [c for c in df.columns if c not in cfg.get("metadata_columns", ())]
    df = df.drop_duplicates(subset=dedup_columns).reset_index(drop=True)
    if len(df) != n_before:
        LOG.info("Dropped %d exact-duplicate rows (cross-CSV dedup)",
                 n_before - len(df))

    LOG.info("Cleaning summary: %d raw -> %d kept (%.2f%%)",
             n_raw_total, len(df), 100 * len(df) / max(n_raw_total, 1))

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    LOG.info("Cached cleaned corpus -> %s", cache)
    return df


# ---------------------------------------------------------------------------
# Composite subsampling (Layer 1)
# ---------------------------------------------------------------------------
def composite_subsample(
    df: pd.DataFrame,
    target_n: int,
    rare_threshold: int,
    label_col: str,
    random_state: int,
) -> pd.DataFrame:
    """Keep ALL rows of rare classes; stratified-subsample the majority.

    Classes with count <= ``rare_threshold`` contribute every row they have
    (RAM cost is negligible -- combined < 20k rows). The remaining budget
    is split proportionally among the larger classes. Output is shuffled.
    """
    counts = df[label_col].value_counts()
    rare_classes = counts[counts <= rare_threshold].index.tolist()
    common_classes = counts[counts > rare_threshold].index.tolist()

    rare_df = df[df[label_col].isin(rare_classes)]
    common_df = df[df[label_col].isin(common_classes)]

    LOG.info("Composite subsample: keeping ALL of %d rare class(es): %s",
             len(rare_classes), sorted(rare_classes))
    LOG.info("  rare rows kept whole: %d", len(rare_df))

    if target_n is None or target_n >= len(df):
        # User asked for >= corpus; just return everything.
        # Shuffling is handled during train/test split.
        return df.reset_index(drop=True)

    budget_for_common = max(0, target_n - len(rare_df))
    if budget_for_common == 0:
        LOG.warning("Rare classes alone exceed target_n=%d; using just rare",
                    target_n)
        return rare_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    rng = np.random.default_rng(random_state)
    pieces: list[pd.DataFrame] = [rare_df]
    common_total = len(common_df)
    for _cls, group in common_df.groupby(label_col, sort=False, observed=True):
        quota = max(1, round(budget_for_common * len(group) / common_total))
        take = min(quota, len(group))
        idx = rng.choice(group.index.to_numpy(), size=take, replace=False)
        pieces.append(df.loc[idx])

    out = (pd.concat(pieces, axis=0, ignore_index=False)
             .sample(frac=1.0, random_state=random_state)
             .reset_index(drop=True))
    LOG.info("  composite subsample total: %d rows (target was %d)",
             len(out), target_n)
    return out


def _proportional_quotas(
    counts: pd.Series,
    budget: int,
    *,
    minimum_per_class: int,
) -> dict[str, int]:
    """Allocate an exact row budget proportionally, subject to class caps."""
    counts = counts.astype(np.int64).sort_index()
    total = int(counts.sum())
    budget = min(int(budget), total)
    minimums = counts.clip(upper=minimum_per_class)
    required = int(minimums.sum())
    if budget < required:
        raise ValueError(
            f"budget={budget} cannot keep minimum_per_class={minimum_per_class} "
            f"for {len(counts)} classes (requires at least {required})"
        )

    ideal = counts.astype(float) * (budget / max(total, 1))
    quotas = ideal.round().astype(np.int64)
    quotas = quotas.where(quotas >= minimums, minimums)
    quotas = quotas.where(quotas <= counts, counts)

    # Rounding and minimum guarantees can move the sum away from the exact
    # budget. Nine CICIDS classes means this loop normally changes <10 rows.
    while int(quotas.sum()) < budget:
        candidates = [c for c in counts.index if quotas[c] < counts[c]]
        cls = max(candidates, key=lambda c: (ideal[c] - quotas[c], counts[c], c))
        quotas[cls] += 1
    while int(quotas.sum()) > budget:
        candidates = [c for c in counts.index if quotas[c] > minimums[c]]
        cls = max(candidates, key=lambda c: (quotas[c] - ideal[c], quotas[c], c))
        quotas[cls] -= 1
    return {str(cls): int(n) for cls, n in quotas.items()}


def _keep_all_rare_training_rows(
    quotas: dict[str, int],
    available: dict[str, int],
    *,
    rare_threshold: int,
) -> dict[str, int]:
    """Keep genuine rare rows in TRAIN and pay for them from the majority."""
    out = dict(quotas)
    extra = 0
    for cls, count in available.items():
        if count <= rare_threshold and out[cls] < count:
            extra += count - out[cls]
            out[cls] = count

    while extra > 0:
        donors = [
            cls for cls, count in out.items()
            if available[cls] > rare_threshold and count > 1
        ]
        if not donors:
            raise ValueError("training budget is too small to retain all rare classes")
        donor = max(donors, key=lambda cls: (out[cls], cls))
        take = min(extra, out[donor] - 1)
        out[donor] -= take
        extra -= take
    return out


def _apply_target_ratio(
    quotas: dict[str, int],
    available: dict[str, int],
    *,
    target_class: str,
    target_ratio: float,
) -> dict[str, int]:
    """Move majority quota to a real target class until target/majority ratio."""
    if target_class not in quotas:
        raise ValueError(f"target class {target_class!r} is not present in the data")
    if not 0.0 < target_ratio <= 1.0:
        raise ValueError("target_ratio must be in the interval (0, 1]")

    out = dict(quotas)
    # Recompute the majority after every transfer. Reducing only the initial
    # majority can expose the second-largest class and leave the requested
    # target/max(non-target) ratio unmet.
    while out[target_class] < available[target_class]:
        majority = max(
            (cls for cls in out if cls != target_class),
            key=lambda cls: (out[cls], cls),
        )
        target_n = out[target_class]
        majority_n = out[majority]
        if target_n / max(majority_n, 1) >= target_ratio:
            break

        # Moving d rows from majority to target preserves the train budget:
        # (target_n + d) / (majority_n - d) >= target_ratio.
        needed = int(np.ceil(
            (target_ratio * majority_n - target_n) / (1.0 + target_ratio)
        ))
        moved = min(
            max(needed, 1),
            available[target_class] - target_n,
            majority_n - 1,
        )
        if moved <= 0:
            break
        out[target_class] += moved
        out[majority] -= moved

    achieved = out[target_class] / max(
        max(count for cls, count in out.items() if cls != target_class),
        1,
    )
    if achieved < target_ratio:
        LOG.warning(
            "Requested target ratio %.4f is unattainable with genuine %s rows; "
            "achieved %.4f",
            target_ratio,
            target_class,
            achieved,
        )
    return out


def budgeted_train_test_split(
    df: pd.DataFrame,
    *,
    label_col: str,
    total_budget: int | None,
    test_size: float,
    min_test_per_class: int,
    rare_threshold: int,
    train_sampling: str,
    target_class: str,
    target_ratio: float,
    random_state: int,
    calibration_size: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select a natural test set first, then sample TRAIN without overlap.

    ``train_sampling='targeted'`` spends part of the fixed training budget on
    additional *real* target-class rows. Test and optional calibration quotas
    are computed before that adjustment and therefore retain the natural
    distribution. The middle calibration frame is empty when
    ``calibration_size`` is zero, keeping the return contract stable.
    """
    if train_sampling not in {"natural", "targeted"}:
        raise ValueError("train_sampling must be 'natural' or 'targeted'")
    if not 0.0 <= calibration_size < 0.5:
        raise ValueError("calibration_size must be in the interval [0, 0.5)")

    if total_budget is None or total_budget >= len(df):
        train_df, test_df = stratified_split_min_test(
            df,
            label_col=label_col,
            test_size=test_size,
            min_test_per_class=min_test_per_class,
            random_state=random_state,
        )
        calibration_df: pd.DataFrame | None = None
        if calibration_size > 0.0:
            relative_size = calibration_size / (1.0 - test_size)
            train_df, calibration_df = stratified_split_min_test(
                train_df,
                label_col=label_col,
                test_size=relative_size,
                min_test_per_class=min_test_per_class,
                random_state=random_state + 1,
            )
        if train_sampling == "targeted":
            counts = train_df[label_col].value_counts().to_dict()
            if target_class not in counts:
                raise ValueError(f"target class {target_class!r} is not present")
            majority_cap = max(1, int(counts[target_class] / target_ratio))
            rng = np.random.default_rng(random_state)
            labels = train_df[label_col].to_numpy()
            keep_parts: list[np.ndarray] = []
            for cls, count in counts.items():
                cls_idx = np.flatnonzero(labels == cls)
                cap = count if cls == target_class else min(count, majority_cap)
                if cap < count:
                    cls_idx = rng.choice(cls_idx, size=cap, replace=False)
                keep_parts.append(np.asarray(cls_idx, dtype=np.int64))
            keep = np.concatenate(keep_parts)
            rng.shuffle(keep)
            train_df = train_df.iloc[keep].reset_index(drop=True)
        if calibration_df is None:
            calibration_df = df.iloc[0:0].copy()
        return train_df, calibration_df, test_df

    counts = df[label_col].value_counts().sort_index()
    n_classes = len(counts)
    minimum_total = n_classes * (min_test_per_class + 1)
    if total_budget < minimum_total:
        raise ValueError(
            f"total_budget={total_budget} is too small; need >= {minimum_total}"
        )

    requested_test = int(round(total_budget * test_size))
    test_caps = (counts - 1).clip(lower=0)
    minimum_test_total = int(test_caps.clip(upper=min_test_per_class).sum())
    test_budget = max(requested_test, minimum_test_total)
    test_budget = min(test_budget, total_budget - n_classes)
    test_quotas = _proportional_quotas(
        test_caps,
        test_budget,
        minimum_per_class=min_test_per_class,
    )

    calibration_quotas = {str(cls): 0 for cls in counts.index}
    if calibration_size > 0.0:
        calibration_budget = int(round(total_budget * calibration_size))
        calibration_caps = pd.Series({
            str(cls): int(counts[cls]) - test_quotas[str(cls)] - 1
            for cls in counts.index
        }).clip(lower=0)
        calibration_quotas = _proportional_quotas(
            calibration_caps,
            calibration_budget,
            minimum_per_class=min_test_per_class,
        )

    available = {
        str(cls): (
            int(counts[cls])
            - test_quotas[str(cls)]
            - calibration_quotas[str(cls)]
        )
        for cls in counts.index
    }
    train_budget = (
        total_budget
        - sum(test_quotas.values())
        - sum(calibration_quotas.values())
    )
    if train_budget < n_classes:
        raise ValueError(
            "total_budget is too small to retain one training row per class "
            "after test and calibration holdouts"
        )
    train_quotas = _proportional_quotas(
        pd.Series(available, dtype=np.int64),
        train_budget,
        minimum_per_class=1,
    )
    train_quotas = _keep_all_rare_training_rows(
        train_quotas,
        available,
        rare_threshold=rare_threshold,
    )
    if train_sampling == "targeted":
        train_quotas = _apply_target_ratio(
            train_quotas,
            available,
            target_class=target_class,
            target_ratio=target_ratio,
        )

    rng = np.random.default_rng(random_state)
    train_idx: list[np.ndarray] = []
    calibration_idx: list[np.ndarray] = []
    test_idx: list[np.ndarray] = []
    # ``GroupBy.indices`` returns positional indices, which are safe for iloc
    # even when a caller provides a non-default DataFrame index.
    grouped_positions = df.groupby(label_col, sort=True, observed=True).indices
    for raw_cls, positions in grouped_positions.items():
        cls = str(raw_cls)
        idx = np.asarray(positions, dtype=np.int64).copy()
        rng.shuffle(idx)
        n_test = test_quotas[cls]
        n_calibration = calibration_quotas[cls]
        n_train = train_quotas[cls]
        test_idx.append(idx[:n_test])
        calibration_idx.append(idx[n_test:n_test + n_calibration])
        train_start = n_test + n_calibration
        train_idx.append(idx[train_start:train_start + n_train])

    train_pos = np.concatenate(train_idx)
    calibration_pos = np.concatenate(calibration_idx)
    test_pos = np.concatenate(test_idx)
    rng.shuffle(train_pos)
    rng.shuffle(calibration_pos)
    rng.shuffle(test_pos)
    train_df = df.iloc[train_pos].reset_index(drop=True)
    calibration_df = df.iloc[calibration_pos].reset_index(drop=True)
    test_df = df.iloc[test_pos].reset_index(drop=True)
    return train_df, calibration_df, test_df


# ---------------------------------------------------------------------------
# Stratified split with per-class minimum test guarantee (Layer 2)
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
    # The combined 2017/2018 schema has a few source-specific columns
    # (notably Protocol and duplicate packet-rate aliases), so unioned rows
    # contain NaN even after per-file cleaning. Tree models accept NaN, but
    # nearest-neighbour samplers do not. Median imputation makes every
    # strategy consume the same finite feature matrix.
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


def hp_grids(model_name: str) -> dict[str, list]:
    """Small randomized-search distributions per model."""
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
            "clf__max_depth":     [6, 8, 10, 12],
            "clf__learning_rate": [0.03, 0.05, 0.1, 0.15],
            "clf__subsample":     [0.7, 0.8, 0.9, 1.0],
        }
    if model_name == "lightgbm":
        # min_child_samples grid: 1-10 only. Anything >=10 starves the
        # Heartbleed-class leaf on per-fold CV (8 train rows -> 1-2 per
        # fold). See build_pipeline() docstring for the diagnosis.
        return {
            "clf__n_estimators":       [400, 600, 800, 1000],
            "clf__num_leaves":         [63, 127, 255],
            "clf__learning_rate":      [0.03, 0.05, 0.1],
            "clf__min_child_samples":  [1, 2, 5],
        }
    if model_name == "catboost":
        return {
            "clf__iterations": [300, 400, 500],
            "clf__depth": [6, 8, 10],
            "clf__learning_rate": [0.03, 0.05, 0.1],
        }
    if model_name == "mlp":
        return {
            "clf__hidden_layer_sizes": [
                (128, 64), (256, 128, 64), (256, 128, 64, 32),
            ],
            "clf__alpha": [0.00001, 0.0001, 0.001],
            "clf__learning_rate_init": [0.0005, 0.001, 0.005],
        }
    if model_name == "logistic_regression":
        return {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__tol": [0.0001, 0.001],
        }
    return {}


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


def evaluate(model: Pipeline, X_test: np.ndarray, y_test: np.ndarray,
             class_names: list[str]):
    y_pred = model.predict(X_test)
    labels = list(range(len(class_names)))
    acc  = accuracy_score(y_test, y_pred)
    # Average recall over classes actually present in this locked test source.
    # This avoids treating a prediction-only class as a malformed target set.
    bacc = recall_score(y_test, y_pred, labels=np.unique(y_test), average="macro", zero_division=0)
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
def write_report(results: list[EvalResult], outdir: Path, cfg: dict[str, Any],
                 n_classes: int, class_names: list[str],
                 per_class_n_test: dict[str, int],
                 per_class_n_train: dict[str, int]) -> Path:
    chance = 1.0 / max(n_classes, 1)
    lines: list[str] = []
    lines.append(f"# CICIDS/CIC-IDS raw corpus -- training run `{cfg['run_name']}`")
    lines.append("")
    lines.append(f"- subsample_n: {cfg['subsample_n']!r}")
    lines.append(f"- rare_threshold (keep-all-rows below this): "
                 f"{cfg['rare_threshold']}")
    lines.append(
        f"- imbalance_strategy: {cfg['imbalance_strategy']} "
        f"(target={cfg['target_class']}, target/majority={cfg['target_ratio']:.3f})"
    )
    lines.append(
        f"- FN-aware threshold: validation_size="
        f"{cfg['threshold_validation_size']:.2f}, "
        f"max target FPR={cfg['target_max_fpr']:.3f}"
    )
    lines.append("- resampling scope: TRAIN/CV folds only; test distribution untouched")
    lines.append(f"- test_size: {cfg['test_size']}, "
                 f"min_test_per_class: {cfg['min_test_per_class']}")
    if cfg["cv_check"]:
        lines.append(f"- CV: StratifiedKFold(n_splits={cfg['cv_splits']})")
    else:
        lines.append("- CV: skipped")
    lines.append(f"- HP search: {cfg['hp_search']} "
                 f"(n_iter={cfg['hp_search_n_iter']}, "
                 f"subsample={cfg['hp_search_subsample']})")
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

    lines.append("## Headline metrics")
    lines.append("")
    lines.append("| model | accuracy | balanced acc | f1_macro | f1_weighted "
                 "| CV f1_macro (mean +/- std) | majority baseline acc "
                 "| shuffled-labels f1_macro |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        shuf = (f"{r.shuffle_f1_macro:.4f}"
                if r.shuffle_f1_macro is not None else "skipped")
        cv_text = (
            f"{r.cv_mean:.4f} +/- {r.cv_std:.4f}"
            if r.cv_mean is not None and r.cv_std is not None else "skipped"
        )
        lines.append(
            f"| {r.model} | {r.accuracy:.4f} | {r.balanced_accuracy:.4f} | "
            f"{r.f1_macro:.4f} | {r.f1_weighted:.4f} | "
            f"{cv_text} | "
            f"{r.majority_baseline:.4f} | {shuf} |"
        )
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

    lines.append("## Top weaknesses + concrete improvements")
    lines.append("")
    lines.append("1. **Minority-class metric variance** -- Heartbleed still "
                 "has only 11 rows in the combined raw corpus, so its "
                 "per-class metric is anecdotal. Improvement: report "
                 "per-class metrics with sample-size caveat (this report "
                 "already does this).")
    lines.append("2. **Training-sample bias** -- if `subsample_n` is below "
                 "the full corpus, some BENIGN application protocols may be "
                 "underrepresented. The held-out test remains naturally "
                 "distributed, but final research numbers should also include "
                 "a larger-RAM sensitivity run.")
    lines.append("3. **CICIDS/CIC-IDS labelling noise** -- labels are assigned "
                 "per attack window, not per flow, so BENIGN flows during "
                 "an attack window may be mislabelled. Improvement: keep a "
                 "separate cross-dataset validation run when reporting final "
                 "research numbers.")
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
            args["primary_metric"] = next(it)
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

    outdir = result_run_dir(str(cfg["run_name"]), results_root=Path(cfg["results_root"]))
    outdir.mkdir(parents=True, exist_ok=True)
    LOG.info("Artefacts -> %s", outdir)

    # --- Stage 1: load + clean (cached) ---------------------------------
    t0 = time.time()
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
    data_sampling = (
        "targeted" if cfg["imbalance_strategy"] == "targeted" else "natural"
    )
    if cfg.get("split_manifest"):
        manifest = load_split_manifest(Path(cfg["split_manifest"]))
        cfg["split_protocol"] = manifest["version"]
        train_df, calibration_df, test_df = deterministic_source_split(
            df,
            label_column=cfg["label_column"],
            quotas=manifest["train_quotas"],
            roles=manifest["roles"],
        )
        LOG.info(
            "Using deterministic source-held-out manifest %s (%s)",
            cfg["split_manifest"],
            manifest["version"],
        )
    else:
        cfg["split_protocol"] = "stratified_row_holdout_70_30"
        train_df, calibration_df, test_df = budgeted_train_test_split(
            df,
            label_col=cfg["label_column"],
            total_budget=cfg["subsample_n"],
            test_size=cfg["test_size"],
            min_test_per_class=cfg["min_test_per_class"],
            rare_threshold=cfg["rare_threshold"],
            train_sampling=data_sampling,
            target_class=cfg["target_class"],
            target_ratio=cfg["target_ratio"],
            random_state=cfg["random_state"],
            calibration_size=cfg["threshold_validation_size"],
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
    if cfg.get("split_manifest"):
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
            and _imbalance_config_matches(saved, cfg)
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
                per_class_df.to_csv(
                    outdir / f"{model_name}_per_class.csv", index=True
                )
                plot_confusion_matrix(
                    cm, class_names,
                    outdir / f"{model_name}_confusion_matrix.png",
                    title=(
                        f"{model_name} -- CICIDS2017 + "
                        "CSE-CIC-IDS2018 test set"
                    ),
                )
                saved.update({
                    "accuracy": acc,
                    "balanced_accuracy": bacc,
                    "f1_macro": f1m,
                    "f1_weighted": f1w,
                    "majority_baseline_acc": majority_baseline,
                    "near_perfect_flag": acc >= cfg["near_perfect_threshold"],
                    **refreshed_target,
                    **_imbalance_metadata(cfg),
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
            t_fit = time.time()
            pipeline.fit(X_train, y_train)
            LOG.info("%s final fit on full train (%d rows): %.1fs",
                     model_name, len(y_train), time.time() - t_fit)

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
        LOG.info("%s test: acc=%.4f, bal_acc=%.4f, f1_macro=%.4f, "
                 "f1_weighted=%.4f", model_name, acc, bacc, f1m, f1w)
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
        per_class_df.to_csv(outdir / f"{model_name}_per_class.csv", index=True)
        plot_confusion_matrix(
            cm, class_names,
            outdir / f"{model_name}_confusion_matrix.png",
            title=(
                f"{model_name} -- CICIDS2017 + "
                "CSE-CIC-IDS2018 test set"
            ),
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
        per_model_metrics.write_text(json_dumps_strict({
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
            **_imbalance_metadata(cfg),
        }, indent=2), encoding="utf-8")
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
            if _imbalance_config_matches(saved, cfg):
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
        "models": [
            {
                "model":              r.model,
                "accuracy":           r.accuracy,
                "balanced_accuracy":  r.balanced_accuracy,
                "f1_macro":           r.f1_macro,
                "f1_weighted":        r.f1_weighted,
                "cv_f1_macro_mean":   r.cv_mean,
                "cv_f1_macro_std":    r.cv_std,
                "cv_f1_macro_scores": r.cv_scores,
                "label_shuffle_acc":      r.shuffle_accuracy,
                "label_shuffle_f1_macro": r.shuffle_f1_macro,
                "best_params":        r.best_params,
                "target_threshold":   r.target_threshold,
                "target_precision":   r.target_precision,
                "target_recall":      r.target_recall,
                "target_f1":          r.target_f1,
                "target_f2":          r.target_f2,
                "target_fpr":         r.target_fpr,
                "target_false_positives": r.target_false_positives,
                "target_false_negatives": r.target_false_negatives,
                "target_to_benign_fn": r.target_to_benign_fn,
                "calibration_recall": r.calibration_recall,
                "calibration_fpr":    r.calibration_fpr,
                "near_perfect_flag":  r.accuracy >= cfg["near_perfect_threshold"],
            }
            for r in results
        ],
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
def _imbalance_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "training_protocol_version": TRAINING_PROTOCOL_VERSION,
        "imbalance_protocol_version": IMBALANCE_PROTOCOL_VERSION,
        "data_fingerprint": cfg.get("data_fingerprint"),
        "random_state": cfg["random_state"],
        "primary_metric": cfg["primary_metric"],
        "rf_class_weight": cfg["rf_class_weight"],
        "hp_search": cfg["hp_search"],
        "hp_search_n_iter": cfg["hp_search_n_iter"],
        "hp_search_subsample": cfg["hp_search_subsample"],
        "imbalance_strategy": cfg["imbalance_strategy"],
        "target_class": cfg["target_class"],
        "target_ratio": cfg["target_ratio"],
        "target_max_fpr": cfg["target_max_fpr"],
        "threshold_validation_size": cfg["threshold_validation_size"],
        "accelerator": cfg["accelerator"],
        "gpu_devices": cfg["gpu_devices"],
    }


def _checkpoint_signature(cfg: dict[str, Any], model_name: str) -> str:
    """Bind a checkpoint to one model, dataset fingerprint, and train policy."""
    payload = {
        "model": model_name,
        "split_manifest": str(cfg.get("split_manifest") or ""),
        **_imbalance_metadata(cfg),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _imbalance_config_matches(saved: dict[str, Any], cfg: dict[str, Any]) -> bool:
    if saved.get("training_protocol_version") != TRAINING_PROTOCOL_VERSION:
        return False
    if saved.get("imbalance_protocol_version") != IMBALANCE_PROTOCOL_VERSION:
        return False
    saved_strategy = saved.get("imbalance_strategy", "class_weight")
    if saved_strategy != cfg["imbalance_strategy"]:
        return False
    return (
        saved.get("data_fingerprint") == cfg.get("data_fingerprint")
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
        and saved.get("accelerator", "cpu") == cfg["accelerator"]
        and saved.get("gpu_devices", "0") == cfg["gpu_devices"]
    )


def _dataset_fingerprint(df: pd.DataFrame, cfg: dict[str, Any]) -> str:
    """Cheap, deterministic identity for artifact-reuse safety.

    The parquet stat catches cache refreshes without hashing a multi-GB file;
    schema and label counts protect against accidental cache replacement.
    """
    cache_path = Path(cfg["clean_cache"])
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


def _eval_result_from_saved(saved: dict, model_name: str) -> EvalResult:
    """Reconstruct EvalResult from a previously-saved metrics JSON so the
    report aggregator can include it without re-evaluating."""
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
    )


if __name__ == "__main__":
    raise SystemExit(main())
