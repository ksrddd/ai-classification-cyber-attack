"""Batch inference on user-uploaded CSVs.

Reads a CSV that follows the combined CICIDS2017/CSE-CIC-IDS2018 flow-feature schema, validates
it against the saved ``feature_names.json``, runs the saved Pipeline,
and returns a DataFrame with predicted class labels and per-class
probabilities.

The model + label encoder are loaded lazily and cached in-process so
the Streamlit dashboard doesn't re-load them on every interaction.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.artifacts.bundle import ArtifactIntegrityError, verify_bundle_manifest
from src.artifacts.paths import result_run_dir
from src.config.constants import LABEL_COLUMN, PROJECT_ROOT
from src.data.schema import clean_column_names
from src.features.encoder import encoder_classes, load_label_encoder
from src.features.validator import (
    FEATURE_NAMES_PATH,
    ValidationReport,
    load_expected_features,
    validate_inference_csv,
)
from src.models.base import default_model_path
from src.utils.io import load_joblib

logger = logging.getLogger(__name__)

LATEST_DIR = PROJECT_ROOT / 'results' / 'latest'
CHAMPION_PATH = PROJECT_ROOT / "results" / "champion.json"


@dataclass
class PredictionResult:
    """Output of a single inference run."""

    model_name: str
    predictions: pd.DataFrame   # one row per input, with ``predicted_label`` + per-class probs
    validation: ValidationReport
    n_classes: int
    run_id: str = "latest"

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
    expected = load_expected_features(_feature_names_path())
    report = validate_inference_csv(df, expected_features=expected)
    if not report.ok:
        raise ValueError(report.message)

    pipeline = _load_pipeline_cached(model_name)
    le = _load_encoder_cached()
    class_names = encoder_classes(le)

    X = prepare_inference_features(df, expected)

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
        run_id=_published_run_id(),
    )


def prepare_inference_features(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    """Align features and preserve missingness for the fitted pipeline imputer."""
    X = df[expected].copy()
    return X.replace([np.inf, -np.inf], np.nan)


def clear_cache() -> None:
    """Drop the in-process model + encoder cache (Streamlit hot-reload)."""
    _load_pipeline_cached.cache_clear()
    _load_encoder_cached.cache_clear()


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _load_pipeline_cached(model_name: str) -> Pipeline:
    model_name = _canonical_model_name(model_name)
    path = _model_path(model_name)
    if not path.exists():
        raise FileNotFoundError(
            f'No saved model for {model_name!r} at {path}. '
            f'Run python main.py --stage train --model {model_name} first.'
        )
    bundle_root = path.parent
    manifest_path = bundle_root / "bundle_manifest.json"
    if manifest_path.exists():
        try:
            verify_bundle_manifest(bundle_root, json.loads(manifest_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ArtifactIntegrityError) as exc:
            raise RuntimeError(f"Refusing unverified model bundle: {bundle_root}") from exc
    pipe = load_joblib(path)
    if not isinstance(pipe, Pipeline):
        raise TypeError(f"Expected Pipeline at {path}, got {type(pipe)}")
    return pipe


@lru_cache(maxsize=1)
def _load_encoder_cached():
    latest = _published_dir() / 'label_encoder.joblib'
    if latest.exists():
        return load_label_encoder(latest)
    if CHAMPION_PATH.exists():
        raise FileNotFoundError(f"Published bundle is missing {latest}")
    return load_label_encoder()


def list_saved_models() -> list[str]:
    """Return names of every model with a saved ``.joblib`` artefact."""
    from src.models.registry import MODEL_CLASSES
    found: list[str] = []
    for name in MODEL_CLASSES:
        if _model_path(name).exists():
            found.append(name)
    return found




def _model_path(model_name: str) -> Path:
    model_name = _canonical_model_name(model_name)
    published = _published_dir()
    # Keep every model/encoder/schema lookup within the same published run.
    # The latest run is preferred; the legacy models directory is retained as
    # a compatibility fallback for older local demos.
    latest = (published / f"{model_name}.joblib").resolve()
    if latest.exists():
        return latest
    if CHAMPION_PATH.exists():
        return latest
    primary = default_model_path(model_name).resolve()
    if primary.exists() and primary.parent == default_model_path(model_name).parent.resolve():
        return primary
    return latest


def _feature_names_path() -> Path:
    latest = _published_dir() / "feature_columns.json"
    if latest.exists():
        return latest
    if CHAMPION_PATH.exists():
        return latest
    if FEATURE_NAMES_PATH.exists():
        return FEATURE_NAMES_PATH
    return FEATURE_NAMES_PATH


def _canonical_model_name(model_name: str) -> str:
    from src.models.registry import MODEL_CLASSES, resolve_name

    canonical = resolve_name(model_name)
    if canonical not in MODEL_CLASSES:
        raise ValueError(f"Unknown model name: {model_name!r}")
    return canonical


def _published_dir() -> Path:
    """Resolve one published run; never combine artifacts from different runs."""
    if CHAMPION_PATH.exists():
        try:
            payload = json.loads(CHAMPION_PATH.read_text(encoding="utf-8"))
            run_id = str(payload["run_id"])
            root = result_run_dir(run_id, results_root=PROJECT_ROOT / "results")
            if root.is_dir():
                return root
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            logger.warning("Ignoring invalid champion pointer: %s", CHAMPION_PATH)
    return LATEST_DIR


def _published_run_id() -> str:
    if CHAMPION_PATH.exists():
        try:
            return str(json.loads(CHAMPION_PATH.read_text(encoding="utf-8"))["run_id"])
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            logger.warning("Invalid champion run ID; using latest")
    return "latest"


def expected_schema_path() -> Path:
    return _feature_names_path()


