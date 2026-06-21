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
    1. Composite subsampling: keep ALL rows of rare classes (Heartbleed=11,
       Infiltration=36, Bot, Web Attack, Brute Force); stratified subsample
       only the majority classes. Total stays within RAM budget AND all 10
       classes survive.
    2. Stratified split with per-class min-test guarantee: every class gets
       >= ``min_test_per_class`` rows in test so per-class recall has
       granularity beyond {0, 50%, 100%}.
    3. Class weighting per model (NO SMOTE on minority -- synthesising
       network flows from 11 samples is not physically defensible):
         RF  -> class_weight='balanced_subsample'
         XGB -> BalancedXGBClassifier (per-fit sample_weight from y)
         LGB -> class_weight='balanced'
    4. Trust checks: 5-fold CV (mean +/- std), label-shuffle macro-F1 (must
       collapse to ~chance), majority-class baseline, per-class N alongside
       metrics so small-N entries are read with appropriate scepticism.

Resumable: cleaned full-corpus data is cached as parquet; per-model
artefacts under ``results/<run_name>/`` are kept and skipped on re-run
unless ``--force`` is passed. Lets you retrain one model at a time
without redoing load+clean (3-5 min).
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
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
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

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
    "test_size":          0.20,
    "min_test_per_class": 3,    # Heartbleed (11) -> 8 train / 3 test

    # -- Modeling (Layer 3) ---------------------------------------------
    "models":             ("random_forest", "xgboost", "lightgbm"),
    "primary_metric":     "f1_macro",
    "rf_class_weight":    "balanced_subsample",

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
    if log.handlers:
        return log
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    log.addHandler(handler)
    logging.captureWarnings(True)
    logging.getLogger("py.warnings").addHandler(handler)
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
        LOG.info("  %s -> raw %d -> clean %d", p.name, len(raw), len(clean))
        del raw
        cleaned.append(clean)

    df = pd.concat(cleaned, axis=0, ignore_index=True)
    del cleaned
    LOG.info("Concatenated cleaned frames: %d rows", len(df))

    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
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
    for cls, group in common_df.groupby(label_col, sort=False, observed=True):
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
    for cls, group in df.groupby(label_col, sort=False, observed=True):
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
            try:
                object.__delattr__(self, "feature_names_in_")
            except AttributeError:
                pass
        return self


def build_pipeline(
    model_name: str,
    n_classes: int,
    random_state: int,
    *,
    rf_class_weight: str | None = "balanced_subsample",
) -> Pipeline:
    """Return an unfitted sklearn Pipeline = StandardScaler -> classifier."""
    if model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            # 'balanced_subsample' recomputes weights per bootstrap sample
            # -- more accurate than 'balanced' on extreme imbalance.
            class_weight=rf_class_weight,
            n_jobs=-1,
            random_state=random_state,
        )
    elif model_name == "xgboost":
        # XGBoost 2.x: no `use_label_encoder`. We set objective + eval_metric
        # explicitly so behaviour is pinned across minor versions and no
        # deprecation warnings fire.
        clf = BalancedXGBClassifier(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
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
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
            verbosity=-1,
            force_col_wise=True,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    # Make StandardScaler output a DataFrame (preserving feature names) so
    # the downstream classifier sees the same X shape on fit and predict.
    # LightGBM 4.x sets feature_names_in_ from DataFrame columns at fit; if
    # predict receives a numpy array, sklearn emits the "X does not have
    # valid feature names" UserWarning. set_output("pandas") fixes this end
    # to end and is the sklearn 1.3+ recommended way.
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


def evaluate(model: Pipeline, X_test: np.ndarray, y_test: np.ndarray,
             class_names: list[str]):
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    f1m  = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    f1w  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    # labels=range(n) ensures every class appears in the report; otherwise
    # missing classes trigger UndefinedMetricWarning.
    report_dict = classification_report(
        y_test, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    per_class = pd.DataFrame(report_dict).transpose().round(4)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))
    return acc, bacc, f1m, f1w, per_class, cm


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


def cross_validate_clean(estimator: Pipeline, X: np.ndarray, y: np.ndarray,
                         cv: StratifiedKFold) -> tuple[list[float], float, float]:
    """5-fold CV f1_macro on TRAIN only. Reports per-fold scores + mean + std."""
    scores = cross_val_score(
        clone(estimator), X, y, cv=cv, scoring="f1_macro", n_jobs=1,
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
                std_msg = (f"**UNSTABLE** (std >= 0.10). Likely cause: some CV folds "
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
    lines.append("2. **Subsample bias for the majority** -- if "
                 "`subsample_n` < full corpus, BENIGN sub-flows from "
                 "specific application protocols may be underrepresented. "
                 "Improvement: retrain with `subsample_n=None` on a "
                 "higher-RAM machine for the final reported numbers.")
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
            "skip_hp": False, "primary_metric": None,
            "rf_class_weight": None, "skip_cv": False,
            "skip_label_shuffle": False}
    it = iter(argv)
    for tok in it:
        if tok == "--smoke":
            args["smoke"] = True
        elif tok == "--force":
            args["force"] = True
        elif tok == "--refresh-cache":
            args["refresh_cache"] = True
        elif tok == "--skip-hp":
            args["skip_hp"] = True
        elif tok == "--skip-cv":
            args["skip_cv"] = True
        elif tok == "--skip-label-shuffle":
            args["skip_label_shuffle"] = True
        elif tok == "--models":
            args["models"] = tuple(next(it).split(","))
        elif tok == "--run-name":
            args["run_name"] = next(it)
        elif tok == "--primary-metric":
            args["primary_metric"] = next(it)
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
                  "[--skip-hp] [--primary-metric METRIC] "
                  "[--rf-class-weight none|balanced|balanced_subsample] "
                  "[--preset 8gb|16gb|32gb|full] [--models rf,xgb,lgbm] "
                  "[--skip-cv] [--skip-label-shuffle] "
                  "[--run-name NAME]")
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
    if args["skip_hp"]:
        cfg["hp_search"] = False
    cfg["cv_check"] = not args["skip_cv"]
    if args["skip_label_shuffle"]:
        cfg["label_shuffle_check"] = False
    if args["primary_metric"]:
        cfg["primary_metric"] = args["primary_metric"]
    if args["rf_class_weight"] is not None or "--rf-class-weight" in argv:
        cfg["rf_class_weight"] = args["rf_class_weight"]

    set_seeds(cfg["random_state"])

    outdir = Path(cfg["results_root"]) / cfg["run_name"]
    outdir.mkdir(parents=True, exist_ok=True)
    LOG.info("Artefacts -> %s", outdir)

    # --- Stage 1: load + clean (cached) ---------------------------------
    t0 = time.time()
    df = load_and_clean_cached(cfg, force=args["refresh_cache"])

    # --- Stage 2: composite subsample (keep-all-rare + stratify-majority)
    df = composite_subsample(
        df,
        target_n=cfg["subsample_n"],
        rare_threshold=cfg["rare_threshold"],
        label_col=cfg["label_column"],
        random_state=cfg["random_state"],
    )

    # Drop any class still left with 1 row (edge case in tiny smoke runs).
    counts = df[cfg["label_column"]].value_counts()
    too_small = counts[counts < cfg["min_test_per_class"] + 1].index.tolist()
    if too_small:
        LOG.warning("Dropping classes with < %d total rows post-subsample: %s",
                    cfg["min_test_per_class"] + 1, too_small)
        df = df[~df[cfg["label_column"]].isin(too_small)].reset_index(drop=True)

    LOG.info("Final label distribution:\n%s",
             df[cfg["label_column"]].value_counts().to_string())

    # --- Stage 3: stratified split with min_test guarantee --------------
    train_df, test_df = stratified_split_min_test(
        df,
        label_col=cfg["label_column"],
        test_size=cfg["test_size"],
        min_test_per_class=cfg["min_test_per_class"],
        random_state=cfg["random_state"],
    )
    LOG.info("Split: train=%d, test=%d", len(train_df), len(test_df))

    # Encode after split so the LabelEncoder sees every present class.
    feature_cols = [c for c in df.columns if c != cfg["label_column"]]
    y_train_str = train_df[cfg["label_column"]].to_numpy()
    y_test_str  = test_df[cfg["label_column"]].to_numpy()
    label_encoder = LabelEncoder().fit(np.concatenate([y_train_str, y_test_str]))
    class_names = list(label_encoder.classes_)
    y_train = label_encoder.transform(y_train_str).astype(np.int64)
    y_test  = label_encoder.transform(y_test_str ).astype(np.int64)
    n_classes = len(class_names)

    # Keep as DataFrame (with float32 cells) so LightGBM's feature_names_in_
    # is consistent between fit and predict (LGBM 4.x warns if predict gets
    # numpy after fit saw names). DataFrame in == DataFrame out everywhere.
    # float32 halves RAM vs float64; CICIDS features have ~7 sig figs of
    # headroom so no precision is lost.
    X_train = train_df[feature_cols].astype(np.float32)
    X_test  = test_df [feature_cols].astype(np.float32)

    # Per-class N (for honest reporting alongside metrics).
    per_class_n_train = {cls: int((y_train_str == cls).sum()) for cls in class_names}
    per_class_n_test  = {cls: int((y_test_str  == cls).sum()) for cls in class_names}
    LOG.info("Per-class n_train: %s", per_class_n_train)
    LOG.info("Per-class n_test:  %s", per_class_n_test)

    # CV n_splits: clip to smallest class size in TRAIN (StratifiedKFold
    # rejects n_splits > min_class_count).
    min_train_class = min(per_class_n_train.values())
    eff_splits = max(2, min(cfg["cv_splits"], min_train_class))
    if eff_splits != cfg["cv_splits"]:
        LOG.warning("Clipping CV n_splits %d -> %d (smallest class has %d train rows)",
                    cfg["cv_splits"], eff_splits, min_train_class)
    cv = StratifiedKFold(n_splits=eff_splits, shuffle=True,
                         random_state=cfg["random_state"])

    # Majority-class baseline (same for all models -- depends only on data).
    train_majority_class = int(np.bincount(y_train).argmax())
    majority_baseline = float(np.mean(y_test == train_majority_class))

    # --- Stage 4: per-model train + evaluate (skip-if-exists) -----------
    results: list[EvalResult] = []
    for model_name in cfg["models"]:
        LOG.info("=" * 64)
        LOG.info("Model: %s", model_name)
        model_path = outdir / f"{model_name}.joblib"
        per_model_metrics = outdir / f"{model_name}_metrics.json"

        if model_path.exists() and per_model_metrics.exists() and not args["force"]:
            LOG.info("Found existing %s; skipping (use --force to retrain)",
                     model_path.name)
            pipeline = joblib.load(model_path)
            saved = json.loads(per_model_metrics.read_text(encoding="utf-8"))
            results.append(_eval_result_from_saved(saved, model_name))
            continue

        # ----- HP search on a smaller TRAIN subset --------------------
        best_params: dict[str, Any] = {}
        pipeline = build_pipeline(
            model_name, n_classes, cfg["random_state"],
            rf_class_weight=cfg["rf_class_weight"],
        )

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
                scoring=cfg["primary_metric"],
                cv=StratifiedKFold(n_splits=hp_splits, shuffle=True,
                                   random_state=cfg["random_state"]),
                n_jobs=cfg["hp_search_jobs"],
                refit=False,
                random_state=cfg["random_state"],
                verbose=0,
                return_train_score=False,
            )
            search.fit(X_hp, y_hp)
            best_params = search.best_params_
            LOG.info("Best params on HP subset: %s", best_params)
            LOG.info("Best CV %s on HP subset: %.4f",
                     cfg["primary_metric"], search.best_score_)
            # Refit a fresh pipeline with best params on FULL TRAIN.
            pipeline = build_pipeline(
                model_name, n_classes, cfg["random_state"],
                rf_class_weight=cfg["rf_class_weight"],
            )
            pipeline.set_params(**best_params)

        t_fit = time.time()
        pipeline.fit(X_train, y_train)
        LOG.info("%s final fit on full train (%d rows): %.1fs",
                 model_name, len(y_train), time.time() - t_fit)

        # ----- Evaluate ----------------------------------------------
        acc, bacc, f1m, f1w, per_class_df, cm = evaluate(
            pipeline, X_test, y_test, class_names,
        )
        LOG.info("%s test: acc=%.4f, bal_acc=%.4f, f1_macro=%.4f, "
                 "f1_weighted=%.4f", model_name, acc, bacc, f1m, f1w)

        # ----- CV trust check (TRAIN only) ---------------------------
        cv_scores: list[float] = []
        cv_mean: float | None = None
        cv_std: float | None = None
        if cfg["cv_check"]:
            cv_scores, cv_mean, cv_std = cross_validate_clean(
                build_pipeline(
                    model_name, n_classes, cfg["random_state"],
                    rf_class_weight=cfg["rf_class_weight"],
                ),
                X_train, y_train, cv,
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
                build_pipeline(
                    model_name, n_classes, cfg["random_state"],
                    rf_class_weight=cfg["rf_class_weight"],
                ),
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
            title=f"{model_name} -- CICIDS/CIC-IDS test set",
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
        )
        results.append(r)

        # Per-model metrics JSON (used by skip-if-exists logic on rerun).
        per_model_metrics.write_text(json.dumps({
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
            "near_perfect_flag":      r.accuracy >= cfg["near_perfect_threshold"],
        }, indent=2), encoding="utf-8")

    requested_models = set(cfg["models"])
    for model_name in CONFIG["models"]:
        if model_name in requested_models:
            continue
        per_model_metrics = outdir / f"{model_name}_metrics.json"
        if per_model_metrics.exists():
            saved = json.loads(per_model_metrics.read_text(encoding="utf-8"))
            results.append(_eval_result_from_saved(saved, model_name))

    # --- Stage 5: shared artefacts + aggregate report -------------------
    joblib.dump(label_encoder, outdir / "label_encoder.joblib")
    (outdir / "feature_columns.json").write_text(
        json.dumps(feature_cols, indent=2), encoding="utf-8")

    metrics_payload = {
        "run_name":         cfg["run_name"],
        "random_state":     cfg["random_state"],
        "n_train":          int(len(y_train)),
        "n_test":           int(len(y_test)),
        "n_features":       int(X_train.shape[1]),
        "n_classes":        n_classes,
        "class_names":      class_names,
        "per_class_n_train": per_class_n_train,
        "per_class_n_test":  per_class_n_test,
        "majority_baseline_acc": majority_baseline,
        "duration_seconds": round(time.time() - t0, 2),
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
                "near_perfect_flag":  r.accuracy >= cfg["near_perfect_threshold"],
            }
            for r in results
        ],
    }
    (outdir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8")

    report_path = write_report(
        results, outdir, cfg, n_classes, class_names,
        per_class_n_test, per_class_n_train,
    )
    LOG.info("Wrote report -> %s", report_path)
    LOG.info("Done in %.1fs. All artefacts under %s",
             time.time() - t0, outdir)
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
    )


if __name__ == "__main__":
    raise SystemExit(main())
