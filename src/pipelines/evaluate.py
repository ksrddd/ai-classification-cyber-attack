"""Evaluation pipeline stage.

Loads every saved model from ``models/`` (one ``.joblib`` per algorithm),
runs inference on the held-out test split, computes metrics + confusion
matrix + classification report, then writes the cross-model comparison
artefacts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.constants import (
    FIGURES_DIR,
    METRICS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
)
from src.config.loader import get_classification_mode, load_config
from src.evaluation.comparison import best_model, write_comparison_report
from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.metrics import (
    classification_report_df,
    compute_metrics,
)
from src.features.encoder import load_label_encoder
from src.models.base import default_model_path
from src.models.registry import MODEL_CLASSES, resolve_name
from src.pipelines.preprocess import LABEL_ENCODED_COLUMN
from src.utils.io import ensure_dir, load_joblib

logger = logging.getLogger(__name__)


def run(config_path: Path, model: str = "all") -> dict:
    """Evaluate trained model(s) on the held-out test split."""
    cfg = load_config(config_path)
    mode = get_classification_mode(cfg)

    test_df = _load_test()
    X_test, y_test = _split_xy(test_df)
    le = load_label_encoder()
    labels_encoded = list(range(len(le.classes_)))
    class_names = [str(c) for c in le.classes_]

    candidates = _select_model_paths(model)
    per_model: dict[str, dict] = {}
    ensure_dir(METRICS_DIR)
    ensure_dir(FIGURES_DIR)
    ensure_dir(REPORTS_DIR)

    for name, path in candidates.items():
        if not path.exists():
            logger.warning("No saved model for %s at %s; skipping.", name, path)
            continue
        logger.info("Evaluating %s from %s", name, path)
        pipe = load_joblib(path)
        y_pred = pipe.predict(X_test)
        y_proba = None
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            try:
                y_proba = pipe.predict_proba(X_test)
            except Exception as exc:  # noqa: BLE001
                logger.warning("predict_proba failed for %s: %s", name, exc)

        m = compute_metrics(
            y_test, y_pred,
            y_proba=y_proba,
            labels=labels_encoded,
            class_names=class_names,
        )
        per_model[name] = m

        # Per-model artefacts.
        cm_path = FIGURES_DIR / f"confusion_matrix_{name}.png"
        plot_confusion_matrix(
            y_test, y_pred,
            labels=labels_encoded,
            class_names=class_names,
            normalize="true",
            title=f"Confusion matrix -- {name}",
            save_to=cm_path,
        )

        report_df = classification_report_df(
            y_test, y_pred,
            labels=labels_encoded,
            class_names=class_names,
        )
        report_df.to_csv(METRICS_DIR / f"classification_report_{name}.csv")

        metrics_path = METRICS_DIR / f"{name}_test.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(_jsonable(m), f, indent=2, default=str)

    if not per_model:
        raise RuntimeError(
            "No trained models found. Run `python main.py --stage train` first."
        )

    artefacts = write_comparison_report(per_model)
    leader = best_model(per_model, metric="f1_weighted")
    logger.info("Best model on test F1-weighted: %s", leader)

    return {
        "mode": mode,
        "metrics_per_model": {k: _strip_dict(v) for k, v in per_model.items()},
        "best_model": leader,
        "comparison_csv": str(artefacts["csv"]),
        "comparison_md":  str(artefacts["md"]),
        "comparison_png": str(artefacts["png"]),
        "n_test": int(len(X_test)),
    }


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _load_test() -> pd.DataFrame:
    p = PROCESSED_DIR / "test.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run preprocessing first.")
    return pd.read_parquet(p)


def _split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df[LABEL_ENCODED_COLUMN].astype(np.int64)
    X = df.drop(columns=[LABEL_ENCODED_COLUMN])
    return X, y


def _select_model_paths(model_arg: str) -> dict[str, Path]:
    if model_arg in ("all", None):
        return {name: default_model_path(name) for name in MODEL_CLASSES}
    canonical = resolve_name(model_arg)
    if canonical not in MODEL_CLASSES:
        raise KeyError(f"Unknown model {model_arg!r}.")
    return {canonical: default_model_path(canonical)}


def _strip_dict(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if not isinstance(v, dict)}


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj
