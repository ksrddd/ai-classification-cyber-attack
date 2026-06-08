"""Batch inference on user-uploaded CSVs.

Reads a CSV that follows the CICIDS2017 flow-feature schema, validates
it against the saved ``feature_names.json``, runs the saved Pipeline,
and returns a DataFrame with predicted class labels and per-class
probabilities.

The model + label encoder are loaded lazily and cached in-process so
the Streamlit dashboard doesn't re-load them on every interaction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config.constants import LABEL_COLUMN, PROCESSED_DIR
from src.data.schema import clean_column_names
from src.features.encoder import encoder_classes, load_label_encoder
from src.features.validator import (
    ValidationReport,
    load_expected_features,
    validate_inference_csv,
)
from src.models.base import default_model_path
from src.utils.io import load_joblib

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Output of a single inference run."""

    model_name: str
    predictions: pd.DataFrame   # one row per input, with ``predicted_label`` + per-class probs
    validation: ValidationReport
    n_classes: int

    def write_csv(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.predictions.to_csv(path, index=False)
        logger.info("Wrote predictions -> %s", path)
        return path


def predict_csv(
    input_csv: Path,
    *,
    model_name: str = "random_forest",
    output_csv: Path | None = None,
    include_probabilities: bool = True,
) -> PredictionResult:
    """Run inference on ``input_csv`` and (optionally) write the result."""
    df = pd.read_csv(input_csv, low_memory=False, encoding="latin-1")
    result = predict_dataframe(
        df, model_name=model_name,
        include_probabilities=include_probabilities,
    )
    if output_csv is not None:
        result.write_csv(output_csv)
    return result


def predict_dataframe(
    df: pd.DataFrame,
    *,
    model_name: str = "random_forest",
    include_probabilities: bool = True,
) -> PredictionResult:
    """Run inference on an already-loaded DataFrame."""
    df = clean_column_names(df)
    expected = load_expected_features()
    report = validate_inference_csv(df, expected_features=expected)
    if not report.ok:
        raise ValueError(report.message)

    pipeline = _load_pipeline_cached(model_name)
    le = _load_encoder_cached()
    class_names = encoder_classes(le)

    X = df[expected].copy()
    # Re-apply the same NaN/Inf handling we use at training time.
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        logger.info("Filling %d NaN cells with 0 for inference.", int(X.isna().sum().sum()))
        X = X.fillna(0)

    y_pred = pipeline.predict(X)
    pred_labels = le.inverse_transform(y_pred)

    out = pd.DataFrame({"predicted_label": pred_labels})
    if LABEL_COLUMN in df.columns:
        out["true_label_raw"] = df[LABEL_COLUMN].values

    if include_probabilities and hasattr(pipeline.named_steps["clf"], "predict_proba"):
        try:
            proba = pipeline.predict_proba(X)
            for i, cls in enumerate(class_names):
                out[f"proba_{cls}"] = proba[:, i]
            out["max_proba"] = proba.max(axis=1)
        except Exception as exc:  # noqa: BLE001
            logger.warning("predict_proba failed: %s", exc)

    return PredictionResult(
        model_name=model_name,
        predictions=out,
        validation=report,
        n_classes=len(class_names),
    )


def clear_cache() -> None:
    """Drop the in-process model + encoder cache (Streamlit hot-reload)."""
    _load_pipeline_cached.cache_clear()
    _load_encoder_cached.cache_clear()


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _load_pipeline_cached(model_name: str) -> Pipeline:
    path = default_model_path(model_name)
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model for {model_name!r} at {path}. "
            f"Run `python main.py --stage train --model {model_name}` first."
        )
    pipe = load_joblib(path)
    if not isinstance(pipe, Pipeline):
        raise TypeError(f"Expected Pipeline at {path}, got {type(pipe)}")
    return pipe


@lru_cache(maxsize=1)
def _load_encoder_cached():
    return load_label_encoder()


def list_saved_models() -> list[str]:
    """Return names of every model with a saved ``.joblib`` artefact."""
    from src.models.registry import MODEL_CLASSES
    found: list[str] = []
    for name in MODEL_CLASSES:
        if default_model_path(name).exists():
            found.append(name)
    return found


def expected_schema_path() -> Path:
    return PROCESSED_DIR / "feature_names.json"
