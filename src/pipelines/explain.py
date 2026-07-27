"""SHAP explainability stage.

Runs against a completed training run bundle: every artifact it needs comes
from ``results/<run-name>/`` plus the cleaned corpus, so the explanations are
guaranteed to describe the models that were actually evaluated.

The test rows are rebuilt with the same chronological split the run used, from
the same manifest, rather than read from a separately-materialised parquet.
That removes a whole class of silent mismatch: the explained rows cannot drift
out of sync with the evaluated ones.

Tree-based models use ``TreeExplainer``; MLP falls back to ``KernelExplainer``.

Invoked via ``python main.py --stage explain --run-name <name>``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.artifacts.paths import result_run_dir
from src.config.constants import (
    CLEAN_CACHE_PATH,
    RESULTS_DIR,
    SPLIT_MANIFEST_PATH,
)
from src.data.temporal_split import load_temporal_manifest, temporal_source_split
from src.explainability.shap_analyzer import (
    ShapResult,
    analyze_model,
    write_shap_report,
)
from src.models.registry import MODEL_CLASSES, resolve_name
from src.utils.io import ensure_dir

logger = logging.getLogger(__name__)

ROW_INDEX_COLUMN = "_row_index"


def run(
    run_name: str = "latest",
    model: str = "all",
    *,
    background_samples: int = 200,
    analysis_samples: int = 1000,
    top_k: int = 10,
    split_manifest: Path | None = None,
    cache_path: Path | None = None,
) -> dict:
    """Run SHAP on the requested model(s) of a training run.

    Artifacts are written under ``results/<run-name>/shap/`` so they travel
    with the bundle they describe instead of into a shared global directory.
    """
    run_dir = result_run_dir(run_name, results_root=RESULTS_DIR)
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"No training run at {run_dir}. Train first: "
            f"python main.py --stage train --run-name {run_name}"
        )

    X_test, class_names = _load_test_split(
        run_dir,
        split_manifest=split_manifest or SPLIT_MANIFEST_PATH,
        cache_path=cache_path or CLEAN_CACHE_PATH,
    )

    shap_dir = run_dir / "shap"
    ensure_dir(shap_dir)

    results: dict[str, ShapResult] = {}
    for name in _select(model):
        model_path = run_dir / f"{name}.joblib"
        if not model_path.exists():
            logger.warning("No saved model for %s in %s; skipping.", name, run_dir)
            continue
        logger.info("SHAP: %s", name)
        pipeline = joblib.load(model_path)
        results[name] = analyze_model(
            pipeline,
            X_test,
            class_names=class_names,
            model_name=name,
            background_samples=background_samples,
            analysis_samples=analysis_samples,
            top_k=top_k,
            save_dir=shap_dir / name,
        )

    if not results:
        raise RuntimeError(f"No trained models found in {run_dir} for SHAP analysis.")

    report_path = write_shap_report(results, save_to=shap_dir / "shap_report.md")
    return {
        "run_dir": str(run_dir),
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


def _load_test_split(
    run_dir: Path,
    *,
    split_manifest: Path,
    cache_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Rebuild the run's locked test partition and its class names."""
    feature_file = run_dir / "feature_columns.json"
    encoder_file = run_dir / "label_encoder.joblib"
    for required in (feature_file, encoder_file):
        if not required.exists():
            raise FileNotFoundError(
                f"{required} is missing; the run bundle is incomplete."
            )
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cleaned corpus not found at {cache_path}. Build it first: "
            "python main.py --stage preprocess --refresh-cache"
        )

    feature_cols = json.loads(feature_file.read_text(encoding="utf-8"))
    class_names = [str(c) for c in joblib.load(encoder_file).classes_]

    manifest = load_temporal_manifest(split_manifest)
    # Only the split keys and the run's own features -- loading all 83 columns
    # of the 2.5M-row corpus to keep 30% of it costs several GB for nothing.
    split_keys = ["Label", "source_file", ROW_INDEX_COLUMN]
    frame = pd.read_parquet(
        cache_path, columns=split_keys + [c for c in feature_cols if c not in split_keys]
    )
    _, _, test_df = temporal_source_split(
        frame,
        order_column=ROW_INDEX_COLUMN,
        test_size=float(manifest["test_size"]),
        min_test_per_class=int(manifest["min_test_per_class"]),
    )
    logger.info("Rebuilt locked test partition: %d rows", len(test_df))

    missing = [c for c in feature_cols if c not in test_df.columns]
    if missing:
        raise ValueError(
            f"Cached corpus is missing {len(missing)} feature column(s) the run "
            f"was trained on, e.g. {missing[:5]}"
        )
    return test_df[feature_cols].astype(np.float32), class_names


def _select(model_arg: str) -> list[str]:
    if model_arg in ("all", None):
        return list(MODEL_CLASSES)
    canonical = resolve_name(model_arg)
    if canonical not in MODEL_CLASSES:
        raise KeyError(f"Unknown model {model_arg!r}.")
    return [canonical]
