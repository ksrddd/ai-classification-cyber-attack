"""Preprocessing pipeline stage -- load + clean + label-map + split + persist.

Outputs (under ``data/processed/``):
- ``train.parquet``, ``val.parquet``, ``test.parquet`` (features + encoded label)
- ``feature_names.json`` (column order for inference-time validation)
- ``label_encoder.joblib`` (saved under ``models/``)
- ``label_classes.json`` (human-readable class list)

Note: feature scaling is intentionally NOT done here -- it lives inside
the sklearn Pipeline so it re-fits per CV fold (ADR-006, leakage proof).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.config.constants import (
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
)
from src.config.loader import get_classification_mode, load_config
from src.data.label_mapping import add_mapped_column, label_distribution
from src.data.loader import load_raw
from src.data.splitter import train_val_test_split
from src.features.cleaning import (
    clean,
    drop_other_class,
    split_features_and_label,
)
from src.features.encoder import fit_label_encoder, save_label_encoder
from src.utils.io import ensure_dir

logger = logging.getLogger(__name__)

LABEL_ENCODED_COLUMN = "label_encoded"


def run(config_path: Path, raw_dir_override: Path | None = None) -> dict:
    """Run the full preprocessing stage end-to-end."""
    cfg = load_config(config_path)
    mode = get_classification_mode(cfg)
    mapped_col = cfg["data"]["mapped_label_column"]
    test_size = cfg["preprocessing"]["test_size"]
    val_size = cfg["preprocessing"]["val_size"]
    stratify = cfg["preprocessing"]["stratify"]
    drop_other = cfg["data"].get("drop_other_class", False)

    # ----- load + clean ---------------------------------------------------
    raw_dir = raw_dir_override if raw_dir_override is not None else Path(cfg["data"]["raw_dir"])
    df = load_raw(
        files=cfg["data"].get("required_files"),
        raw_dir=raw_dir,
        subsample_n=cfg["data"].get("subsample_n"),
    )
    df = clean(
        df,
        drop_dup=cfg["preprocessing"].get("drop_duplicates", True),
        drop_inf=cfg["preprocessing"].get("drop_inf", True),
        drop_na=cfg["preprocessing"].get("drop_na", True),
    )

    # ----- label mapping --------------------------------------------------
    df = add_mapped_column(df, mode=mode)
    if drop_other and mode == "multiclass":
        df = drop_other_class(df, label_col=mapped_col)

    distribution = label_distribution(df, mapped_col).to_dict()
    logger.info("Post-mapping label distribution: %s", distribution)

    # ----- features / label -----------------------------------------------
    X, y_raw = split_features_and_label(df, label_col=mapped_col)
    # Drop any remaining non-numeric columns (defensive -- CICIDS shouldn't
    # have any after schema validation, but Streamlit-uploaded CSVs might).
    non_numeric = X.select_dtypes(exclude="number").columns
    if len(non_numeric):
        logger.warning("Dropping non-numeric feature columns: %s", list(non_numeric))
        X = X.drop(columns=non_numeric)

    le = fit_label_encoder(y_raw, mode=mode, strict=False)
    y = pd.Series(le.transform(y_raw), name=LABEL_ENCODED_COLUMN)

    # ----- three-way split ------------------------------------------------
    split = train_val_test_split(
        X, y,
        test_size=test_size,
        val_size=val_size,
        stratify=stratify,
        random_state=RANDOM_STATE,
    )

    # ----- persist --------------------------------------------------------
    ensure_dir(PROCESSED_DIR)
    ensure_dir(MODELS_DIR)

    train_path = PROCESSED_DIR / "train.parquet"
    val_path   = PROCESSED_DIR / "val.parquet"
    test_path  = PROCESSED_DIR / "test.parquet"
    pd.concat([split.X_train, split.y_train], axis=1).to_parquet(train_path, index=False)
    pd.concat([split.X_val,   split.y_val],   axis=1).to_parquet(val_path,   index=False)
    pd.concat([split.X_test,  split.y_test],  axis=1).to_parquet(test_path,  index=False)

    save_label_encoder(le)

    feature_names_path = PROCESSED_DIR / "feature_names.json"
    with feature_names_path.open("w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, indent=2)

    label_classes_path = PROCESSED_DIR / "label_classes.json"
    with label_classes_path.open("w", encoding="utf-8") as f:
        json.dump(list(le.classes_), f, indent=2)

    summary = {
        "mode": mode,
        "train_path": str(train_path),
        "val_path": str(val_path),
        "test_path": str(test_path),
        "feature_names_path": str(feature_names_path),
        "label_classes_path": str(label_classes_path),
        "n_features": X.shape[1],
        "label_classes": list(le.classes_),
        "label_distribution": {str(k): int(v) for k, v in distribution.items()},
        "train_size": int(len(split.X_train)),
        "val_size":   int(len(split.X_val)),
        "test_size":  int(len(split.X_test)),
    }
    logger.info(
        "Wrote %s rows train / %s rows val / %s rows test",
        summary["train_size"], summary["val_size"], summary["test_size"],
    )
    return summary
