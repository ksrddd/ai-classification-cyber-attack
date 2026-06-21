"""fast_preprocess_from_cache.py

อ่านข้อมูลจาก data/processed/cicids_clean.parquet ที่ train.py สร้างไว้แล้ว
แทนที่จะโหลด raw CSV ใหม่ทั้งหมด (ประหยัดเวลา 5-10 นาที)

สร้างไฟล์เหล่านี้สำหรับ Dashboard:
  data/processed/train.parquet
  data/processed/val.parquet
  data/processed/test.parquet
  data/processed/label_classes.json
  data/processed/feature_names.json
  data/processed/label_encoder.joblib  (copy จาก results/latest/)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fast_preprocess")

CLEAN_CACHE   = ROOT / "data" / "processed" / "cicids_clean.parquet"
PROCESSED_DIR = ROOT / "data" / "processed"
LATEST_DIR    = ROOT / "results" / "latest"
MODELS_DIR    = ROOT / "models"

# config
SUBSAMPLE_N  = 300_000   # ปรับได้ถ้า RAM เยอะ
TEST_SIZE    = 0.20
VAL_SIZE     = 0.20      # ส่วนที่เหลือจาก train
RANDOM_STATE = 42
MIN_PER_CLASS = 10
LABEL_COL    = "Label"
ENCODED_COL  = "label_encoded"


def load_clean_parquet() -> pd.DataFrame:
    log.info("Loading cicids_clean.parquet  (%s) ...", CLEAN_CACHE)
    df = pd.read_parquet(CLEAN_CACHE)
    log.info("  -> %d rows x %d cols", *df.shape)
    return df


def subsample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Stratified subsample keeping rare classes intact."""
    if n >= len(df):
        return df.copy()
    log.info("Stratified subsample -> %d rows ...", n)
    rng = np.random.default_rng(RANDOM_STATE)
    pieces = []
    for lbl, grp in df.groupby(LABEL_COL, sort=False):
        quota = max(MIN_PER_CLASS, round(n * len(grp) / len(df)))
        take  = min(quota, len(grp))
        idx   = rng.choice(grp.index.to_numpy(), size=take, replace=False)
        pieces.append(df.loc[idx])
    out = pd.concat(pieces).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    log.info("  -> %d rows after subsample", len(out))
    return out


def drop_rare(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[LABEL_COL].value_counts()
    rare   = counts[counts < MIN_PER_CLASS].index.tolist()
    if rare:
        log.warning("Dropping %d rare class(es) (<%d rows): %s", len(rare), MIN_PER_CLASS, rare)
        df = df[~df[LABEL_COL].isin(rare)].reset_index(drop=True)
    return df


def split_and_encode(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c != LABEL_COL]
    # drop non-numeric
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < len(feature_cols):
        dropped = set(feature_cols) - set(numeric_cols)
        log.warning("Dropping non-numeric columns: %s", dropped)
    X = df[numeric_cols]
    y_raw = df[LABEL_COL]

    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_raw), name=ENCODED_COL, index=df.index)

    log.info("Label classes: %s", list(le.classes_))
    log.info("Label distribution:\n%s", y_raw.value_counts().to_string())

    # -- test split first
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    # -- then val from train
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=VAL_SIZE, stratify=y_tv, random_state=RANDOM_STATE,
    )
    log.info(
        "Split: train=%d | val=%d | test=%d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, le, numeric_cols


def save_artefacts(X_train, X_val, X_test, y_train, y_val, y_test, le, feature_cols):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    pd.concat([X_train, y_train], axis=1).to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    pd.concat([X_val,   y_val],   axis=1).to_parquet(PROCESSED_DIR / "val.parquet",   index=False)
    pd.concat([X_test,  y_test],  axis=1).to_parquet(PROCESSED_DIR / "test.parquet",  index=False)
    log.info("Saved train/val/test parquet -> %s", PROCESSED_DIR)

    (PROCESSED_DIR / "feature_names.json").write_text(
        json.dumps(list(feature_cols), indent=2), encoding="utf-8"
    )
    (PROCESSED_DIR / "label_classes.json").write_text(
        json.dumps(list(le.classes_), indent=2), encoding="utf-8"
    )
    log.info("Saved feature_names.json + label_classes.json")

    # Save label encoder
    import joblib
    le_path = PROCESSED_DIR / "label_encoder.joblib"
    joblib.dump(le, le_path)
    log.info("Saved label_encoder.joblib -> %s", le_path)

    # Also copy to models/ so explain.py / predictor.py finds it
    models_le = MODELS_DIR / "label_encoder.joblib"
    joblib.dump(le, models_le)
    log.info("Copied label_encoder.joblib -> %s", models_le)


def main():
    if not CLEAN_CACHE.exists():
        log.error(
            "cicids_clean.parquet not found at %s\n"
            "Run: python main.py --stage train  (or python train.py) first.",
            CLEAN_CACHE,
        )
        sys.exit(1)

    df = load_clean_parquet()
    df = subsample(df, SUBSAMPLE_N)
    df = drop_rare(df)

    X_train, X_val, X_test, y_train, y_val, y_test, le, feature_cols = split_and_encode(df)
    save_artefacts(X_train, X_val, X_test, y_train, y_val, y_test, le, feature_cols)

    log.info("Done! Dashboard pages should now load correctly.")
    log.info("Next: python main.py --stage eda   (optional, for EDA plots)")
    log.info("Next: python main.py --stage explain  (for SHAP page)")


if __name__ == "__main__":
    main()
