"""Preprocessing pipeline stage — clean + filter + split + persist.

Outputs (under ``data/processed/``):
- ``train.parquet``, ``test.parquet`` (features + encoded label)
- ``feature_names.json`` (column order for inference-time validation)

Note: feature scaling is intentionally NOT done here — it lives inside
the sklearn Pipeline so it re-fits per CV fold (ADR-006, leakage proof).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.constants import (
    LABEL_COLUMN,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
)
from src.config.loader import load_config
from src.data.loader import load_raw
from src.features.cleaning import clean, filter_target_classes
from src.features.encoder import fit_label_encoder, save_label_encoder
from src.utils.io import ensure_dir

logger = logging.getLogger(__name__)


def run(config_path: Path, raw_dir_override: Path | None = None) -> dict:
    """Run the full preprocessing stage end-to-end.

    See ``src.pipelines.eda.run`` for the meaning of ``raw_dir_override``.
    """
    cfg = load_config(config_path)
    target_labels = cfg["data"]["target_labels"]
    test_size = cfg["preprocessing"]["test_size"]

    if raw_dir_override is not None:
        logger.info("Using raw_dir override: %s", raw_dir_override)
        df = load_raw(
            raw_dir=raw_dir_override,
            subsample_n=cfg["data"].get("subsample_n"),
        )
    else:
        df = load_raw(
            files=cfg["data"].get("required_files"),
            subsample_n=cfg["data"].get("subsample_n"),
        )
    df = clean(df)
    df = filter_target_classes(df, target_labels)

    le = fit_label_encoder(df[LABEL_COLUMN])
    y = pd.Series(le.transform(df[LABEL_COLUMN]),
                  index=df.index, name="label_encoded")
    X = df.drop(columns=[LABEL_COLUMN])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y if cfg["preprocessing"]["stratify"] else None,
        random_state=RANDOM_STATE,
    )

    ensure_dir(PROCESSED_DIR)
    ensure_dir(MODELS_DIR)
    train_path = PROCESSED_DIR / "train.parquet"
    test_path = PROCESSED_DIR / "test.parquet"
    pd.concat([X_train, y_train], axis=1).to_parquet(train_path, index=False)
    pd.concat([X_test, y_test], axis=1).to_parquet(test_path, index=False)
    save_label_encoder(le)

    feature_names_path = PROCESSED_DIR / "feature_names.json"
    with feature_names_path.open("w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, indent=2)

    logger.info("Wrote %s (%s rows) and %s (%s rows)",
                train_path, len(X_train), test_path, len(X_test))
    return {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "feature_names_path": str(feature_names_path),
        "n_features": X.shape[1],
        "label_classes": list(le.classes_),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }
