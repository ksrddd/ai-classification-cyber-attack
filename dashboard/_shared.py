"""Shared helpers for the Streamlit pages.

Streamlit re-runs page scripts from scratch on every interaction, so any
expensive load (config, parquet, models) is wrapped in ``@st.cache_*``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.constants import (  # noqa: E402
    FIGURES_DIR,
    METRICS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    SHAP_DIR,
)
from src.config.loader import (  # noqa: E402
    get_active_target_labels,
    get_classification_mode,
    load_config,
)
from src.inference.predictor import list_saved_models  # noqa: E402

LATEST_DIR = PROJECT_ROOT / "results" / "latest"


@st.cache_resource
def cfg():
    return load_config()


# ttl=60 so a new train run is picked up within 1 minute without restart
@st.cache_data(ttl=60, show_spinner=False)
def cached_list_models() -> list[str]:
    models = list_saved_models()
    if models:
        return models
    if not LATEST_DIR.exists():
        return []
    from src.models.registry import MODEL_CLASSES
    return [name for name in MODEL_CLASSES if (LATEST_DIR / f"{name}.joblib").exists()]


@st.cache_data(ttl=30, show_spinner=False)
def load_parquet_row_counts() -> dict[str, int]:
    """Row counts via parquet metadata — ~1 ms, no column data loaded."""
    import pyarrow.parquet as pq
    result: dict[str, int] = {}
    for key, fname in [("train", "train.parquet"), ("test", "test.parquet")]:
        p = PROCESSED_DIR / fname
        if p.exists():
            result[key] = pq.read_metadata(str(p)).num_rows
    return result


@st.cache_data(ttl=30, show_spinner=False)
def load_train_parquet() -> pd.DataFrame | None:
    p = PROCESSED_DIR / "train.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data(ttl=30, show_spinner=False)
def load_test_parquet() -> pd.DataFrame | None:
    p = PROCESSED_DIR / "test.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data(ttl=30, show_spinner=False)
def load_label_distribution() -> pd.Series | None:
    """Read only label_encoded column + decode via label_classes.json."""
    p = PROCESSED_DIR / "train.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p, columns=["label_encoded"])
        counts = df["label_encoded"].value_counts().sort_index()
        classes_path = PROCESSED_DIR / "label_classes.json"
        if classes_path.exists():
            classes: list[str] = json.loads(classes_path.read_text(encoding="utf-8"))
            counts = counts.rename(index={i: name for i, name in enumerate(classes)})
        return counts
    except Exception:
        df = pd.read_parquet(p)
        label_col = next(
            (c for c in df.columns if "label" in c.lower() and "encoded" not in c.lower()),
            df.columns[-1],
        )
        return df[label_col].value_counts().sort_index()


@st.cache_data(ttl=30, show_spinner=False)
def load_eda_summary() -> dict | None:
    p = METRICS_DIR / "eda_summary.json"
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=30, show_spinner=False)
def load_model_metrics(model_name: str) -> dict | None:
    p = METRICS_DIR / f"{model_name}_test.json"
    if not p.exists():
        p = METRICS_DIR / f"{model_name}_val.json"
        if not p.exists():
            p = LATEST_DIR / f"{model_name}_metrics.json"
            if not p.exists():
                return None
    with p.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    if p.parent == LATEST_DIR:
        _enrich_latest_metrics(model_name, metrics)
    return metrics


@st.cache_data(ttl=30, show_spinner=False)
def load_comparison_csv() -> pd.DataFrame | None:
    p = METRICS_DIR / "model_comparison.csv"
    if p.exists():
        return pd.read_csv(p, index_col=0)
    latest_metrics = LATEST_DIR / "metrics.json"
    if not latest_metrics.exists():
        return None
    payload = json.loads(latest_metrics.read_text(encoding="utf-8"))
    rows = []
    for item in payload.get("models", []):
        row = dict(item)
        row.pop("cv_f1_macro_scores", None)
        row.pop("best_params", None)
        row["model"] = item.get("model")
        rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows).set_index("model")


@st.cache_data(ttl=30, show_spinner=False)
def load_shap_top_features(model_name: str) -> dict | None:
    """Cached SHAP top-features JSON — avoids re-read on every class selectbox change."""
    p = SHAP_DIR / model_name / "top_features.json"
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def figures_dir() -> Path:
    return FIGURES_DIR


def shap_dir() -> Path:
    return SHAP_DIR


def metrics_dir() -> Path:
    return METRICS_DIR


def latest_dir() -> Path:
    return LATEST_DIR


def confusion_matrix_path(model_name: str) -> Path | None:
    candidates = [
        FIGURES_DIR / f"confusion_matrix_{model_name}.png",
        LATEST_DIR / f"{model_name}_confusion_matrix.png",
    ]
    return next((p for p in candidates if p.exists()), None)


def classification_report_path(model_name: str) -> Path | None:
    candidates = [
        METRICS_DIR / f"classification_report_{model_name}.csv",
        LATEST_DIR / f"{model_name}_per_class.csv",
    ]
    return next((p for p in candidates if p.exists()), None)


def _enrich_latest_metrics(model_name: str, metrics: dict) -> None:
    report = LATEST_DIR / f"{model_name}_per_class.csv"
    if not report.exists():
        return
    df = pd.read_csv(report, index_col=0)
    if "per_class" not in metrics:
        per_class = {}
        for idx, row in df.iterrows():
            if str(idx) in {"accuracy", "macro avg", "weighted avg"}:
                continue
            per_class[str(idx)] = {
                "precision": float(row.get("precision", 0.0)),
                "recall": float(row.get("recall", 0.0)),
                "f1": float(row.get("f1-score", 0.0)),
            }
        metrics["per_class"] = per_class
    if "weighted avg" in df.index:
        w = df.loc["weighted avg"]
        metrics.setdefault("precision_weighted", float(w.get("precision", 0.0)))
        metrics.setdefault("recall_weighted", float(w.get("recall", 0.0)))
    if "macro avg" in df.index:
        m = df.loc["macro avg"]
        metrics.setdefault("precision_macro", float(m.get("precision", 0.0)))


def active_labels() -> list[str]:
    return list(get_active_target_labels(cfg()))


def active_mode() -> str:
    return get_classification_mode(cfg())


def warn_no_data() -> None:
    st.warning(
        "No preprocessed data found. "
        "Run `python main.py --stage preprocess` first.",
        icon="⚠️",
    )


def warn_no_models() -> None:
    st.warning(
        "No trained models found. "
        "Run `python main.py --stage train` (and optionally `--stage evaluate`) first.",
        icon="⚠️",
    )
