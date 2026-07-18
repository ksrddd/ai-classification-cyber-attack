"""Explainability pipeline stage.

For every saved model under ``models/``:
1. Load the Pipeline.
2. Run SHAP on a stratified sample of the test split.
3. Save per-class summary plots + a top-features JSON.

Tree-based models use ``TreeExplainer``; MLP falls back to
``KernelExplainer`` with the configured background/analysis sizes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config.constants import (
    PROCESSED_DIR,
    RESULTS_DIR,
    SHAP_DIR,
)
from src.config.loader import load_config
from src.explainability.shap_analyzer import (
    ShapResult,
    analyze_model,
    write_shap_report,
)
from src.features.encoder import load_label_encoder
from src.models.base import default_model_path
from src.models.registry import MODEL_CLASSES, resolve_name
from src.pipelines.preprocess import LABEL_ENCODED_COLUMN
from src.utils.io import ensure_dir, load_joblib

LATEST_DIR = RESULTS_DIR / "latest"

logger = logging.getLogger(__name__)


def run(config_path: Path, model: str = "all") -> dict:
    """Run SHAP explainability on the requested model(s)."""
    cfg = load_config(config_path)
    shap_cfg = cfg["shap"]

    test_parquet = PROCESSED_DIR / "test.parquet"
    if not test_parquet.exists():
        raise FileNotFoundError(
            f"Preprocessed test data not found at {test_parquet}. "
            "Please run the preprocess stage first: "
            "python main.py --stage preprocess"
        )
    test_df = pd.read_parquet(test_parquet)
    X_test = test_df.drop(columns=[LABEL_ENCODED_COLUMN])
    le = load_label_encoder()
    class_names = [str(c) for c in le.classes_]

    candidates = _select(model)
    results: dict[str, ShapResult] = {}
    ensure_dir(SHAP_DIR)

    for name, path in candidates.items():
        if not path.exists():
            logger.warning("No saved model for %s at %s; skipping.", name, path)
            continue
        pipe = load_joblib(path)
        results[name] = analyze_model(
            pipe,
            X_test,
            class_names=class_names,
            model_name=name,
            background_samples=int(shap_cfg.get("background_samples", 200)),
            analysis_samples=int(shap_cfg.get("analysis_samples", 1000)),
            top_k=int(shap_cfg.get("top_k_features", 10)),
            save_dir=SHAP_DIR / name,
        )

    if not results:
        raise RuntimeError("No trained models found for SHAP analysis.")

    report_path = write_shap_report(results)
    return {
        "report_path": str(report_path),
        "models": {
            name: {
                "explainer": r.explainer_kind,
                "top_overall": r.top_features_overall[:5],
                "artefacts": {k: str(v) for k, v in r.artefacts.items()},
            }
            for name, r in results.items()
        },
    }


def _select(model_arg: str) -> dict[str, Path]:
    if model_arg in ("all", None):
        return {name: _resolve_model_path(name) for name in MODEL_CLASSES}
    canonical = resolve_name(model_arg)
    if canonical not in MODEL_CLASSES:
        raise KeyError(f"Unknown model {model_arg!r}.")
    return {canonical: _resolve_model_path(canonical)}


def _resolve_model_path(name: str) -> Path:
    """Return the path to a saved model, checking models/ then results/latest/."""
    primary = default_model_path(name)  # models/<name>.joblib
    if primary.exists():
        return primary
    fallback = LATEST_DIR / f"{name}.joblib"  # results/latest/<name>.joblib
    return fallback
