"""Training pipeline stage.

Workflow
--------
1. Load preprocessed parquet artifacts (``train.parquet``, ``val.parquet``).
2. Build the requested model(s) via ``src.models.registry``.
3. For each model:
   a. Fit a baseline (no hyperparameter tuning) on train.
   b. Optionally run ``tune_model`` for CV-based hyperparameter search.
   c. Compute validation metrics on the held-out val split.
   d. Persist the fitted pipeline + cv_results.
4. Write per-model metric JSON to ``results/metrics/<model>.json``.

The held-out test split is left untouched -- it's the evaluation
pipeline's job to compute final test-set metrics.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.constants import (
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
)
from src.config.loader import get_classification_mode, load_config
from src.evaluation.metrics import classification_report_df, compute_metrics
from src.features.encoder import load_label_encoder
from src.models.base import default_model_path
from src.models.registry import (
    MODEL_CLASSES,
    available_models,
    resolve_name,
)
from src.models.tuner import cv_results_summary, tune_model
from src.pipelines.preprocess import LABEL_ENCODED_COLUMN
from src.utils.io import ensure_dir, save_joblib

logger = logging.getLogger(__name__)


def run(
    config_path: Path,
    model: str = "all",
    *,
    skip_tuning: bool = False,
) -> dict:
    """Train the requested model(s) end-to-end. Returns a summary dict."""
    cfg = load_config(config_path)
    mode = get_classification_mode(cfg)

    train_df, val_df = _load_splits()
    X_train, y_train = _split_xy(train_df)
    X_val,   y_val   = _split_xy(val_df)

    le = load_label_encoder()
    labels_encoded = list(range(len(le.classes_)))
    class_names = [str(c) for c in le.classes_]

    models = _select_models(cfg, model)
    if not models:
        raise ValueError(f"No models matched {model!r}.")

    results_summary: dict[str, dict] = {}
    ensure_dir(METRICS_DIR)
    ensure_dir(MODELS_DIR)

    for name, wrapper in models.items():
        logger.info("=" * 70)
        logger.info("Training model: %s (classification_mode=%s)", name, mode)
        t0 = time.time()

        baseline = wrapper.build()
        baseline.fit(X_train, y_train)
        baseline_metrics = _eval_on_val(
            baseline, X_val, y_val,
            labels=labels_encoded, class_names=class_names,
        )
        logger.info(
            "%s baseline val: acc=%.4f, f1_weighted=%.4f, f1_macro=%.4f",
            name, baseline_metrics["accuracy"],
            baseline_metrics["f1_weighted"], baseline_metrics["f1_macro"],
        )

        tuned_pipeline = baseline
        tuned_metrics = baseline_metrics
        tune_dict: dict | None = None
        if not skip_tuning and cfg["tuning"].get("enabled", True):
            tune_dict = _tune_one(name, wrapper, cfg, X_train, y_train)
            if tune_dict is not None and tune_dict["best_estimator"] is not None:
                tuned_pipeline = tune_dict["best_estimator"]
                tuned_metrics = _eval_on_val(
                    tuned_pipeline, X_val, y_val,
                    labels=labels_encoded, class_names=class_names,
                )
                logger.info(
                    "%s tuned val: acc=%.4f, f1_weighted=%.4f, f1_macro=%.4f",
                    name, tuned_metrics["accuracy"],
                    tuned_metrics["f1_weighted"], tuned_metrics["f1_macro"],
                )

        model_path = save_joblib(tuned_pipeline, default_model_path(name))
        elapsed = time.time() - t0

        metrics_path = METRICS_DIR / f"{name}_val.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": name,
                    "classification_mode": mode,
                    "duration_seconds": elapsed,
                    "baseline_val_metrics": _strip_dict_metrics(baseline_metrics),
                    "tuned_val_metrics":    _strip_dict_metrics(tuned_metrics),
                    "best_params": (tune_dict or {}).get("best_params", {}),
                },
                f, indent=2, default=str,
            )

        results_summary[name] = {
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "duration_seconds": elapsed,
            "baseline_val_metrics": _strip_dict_metrics(baseline_metrics),
            "tuned_val_metrics":    _strip_dict_metrics(tuned_metrics),
            "best_params": (tune_dict or {}).get("best_params", {}),
        }

    # Cross-model val comparison artefact -- mirrors what the evaluate
    # pipeline emits for the test set; lets the dashboard show val too.
    _write_val_summary(results_summary)

    return {
        "mode": mode,
        "models": results_summary,
        "n_features": X_train.shape[1],
        "n_train": int(len(X_train)),
        "n_val":   int(len(X_val)),
    }


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = PROCESSED_DIR / "train.parquet"
    val_path   = PROCESSED_DIR / "val.parquet"
    for p in (train_path, val_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run preprocessing first: "
                "`python main.py --stage preprocess`."
            )
    return pd.read_parquet(train_path), pd.read_parquet(val_path)


def _split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if LABEL_ENCODED_COLUMN not in df.columns:
        raise ValueError(
            f"DataFrame missing {LABEL_ENCODED_COLUMN!r}. Re-run preprocessing."
        )
    y = df[LABEL_ENCODED_COLUMN].astype(np.int64)
    X = df.drop(columns=[LABEL_ENCODED_COLUMN])
    return X, y


def _select_models(cfg: dict, model_arg: str):
    if model_arg in ("all", None):
        return available_models(cfg)
    canonical = resolve_name(model_arg)
    if canonical not in MODEL_CLASSES:
        raise KeyError(f"Unknown model {model_arg!r}.")
    return {canonical: MODEL_CLASSES[canonical](
        config=cfg.get("models", {}).get(canonical, {}),
        scaler_kind=cfg["preprocessing"].get("scaler", "standard"),
    )}


def _eval_on_val(pipeline, X_val, y_val, labels, class_names) -> dict:
    y_pred = pipeline.predict(X_val)
    y_proba = None
    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X_val)
        except Exception as exc:  # noqa: BLE001
            logger.warning("predict_proba failed: %s", exc)
    return compute_metrics(
        y_val, y_pred,
        y_proba=y_proba,
        labels=labels,
        class_names=class_names,
    )


def _tune_one(name, wrapper, cfg, X, y) -> dict | None:
    grid = wrapper.grid
    if not grid:
        logger.info("No tuning grid for %s; skipping.", name)
        return None
    result = tune_model(
        model_name=name,
        pipeline=wrapper.build(),
        param_grid=grid,
        X=X, y=y,
        strategy=cfg["tuning"].get("strategy", "random"),
        n_iter=cfg["tuning"].get("random_n_iter", 20),
        cv_splits=cfg["cv"]["n_splits"],
        scoring=cfg["cv"]["scoring_primary"],
        n_jobs=-1,
        verbose=cfg["tuning"].get("verbose", 1),
    )
    cv_path = METRICS_DIR / f"{name}_cv_results.csv"
    ensure_dir(cv_path.parent)
    cv_results_summary(result.cv_results, top_n=10).to_csv(cv_path, index=False)
    return {
        "best_estimator": result.best_estimator,
        "best_params": result.best_params,
        "best_score": result.best_score,
        "scoring": result.scoring,
        "strategy": result.strategy,
    }


def _strip_dict_metrics(metrics: dict) -> dict:
    """Strip the nested ``per_class`` dict for the flat summary JSON."""
    return {k: v for k, v in metrics.items() if not isinstance(v, dict)}


def _write_val_summary(results: dict[str, dict]) -> Path:
    """Write a CSV row per model with tuned-val metrics."""
    rows = []
    for name, r in results.items():
        row = {"model": name}
        row.update(r.get("tuned_val_metrics", {}))
        rows.append(row)
    df = pd.DataFrame(rows).set_index("model")
    path = METRICS_DIR / "val_summary.csv"
    ensure_dir(path.parent)
    df.to_csv(path)
    logger.info("Wrote validation summary -> %s", path)
    return path
