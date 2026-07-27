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
    NON_SCALAR_METRIC_KEYS,
    PROJECT_ROOT,
)
from src.config.loader import (  # noqa: E402
    get_active_target_labels,
    get_classification_mode,
    load_config,
)
from src.inference.predictor import (  # noqa: E402
    list_saved_models,
    published_run_dir,
)


def latest_dir() -> Path:
    """Directory of the run currently published by results/champion.json.

    Resolved on every call, not bound at import: Streamlit keeps modules loaded
    across reruns, so a module-level constant would keep serving the run that
    was promoted when the server started. The API resolves per request for the
    same reason.
    """
    return published_run_dir()


@st.cache_resource
def cfg():
    return load_config()


# ttl=60 so a new train run is picked up within 1 minute without restart
@st.cache_data(ttl=60, show_spinner=False)
def cached_list_models() -> list[str]:
    models = list_saved_models()
    if models:
        return models
    if not latest_dir().exists():
        return []
    from src.models.registry import MODEL_CLASSES
    return [name for name in MODEL_CLASSES if (latest_dir() / f"{name}.joblib").exists()]


@st.cache_data(ttl=30, show_spinner=False)
def load_parquet_row_counts() -> dict[str, int]:
    """Train/test row counts, taken from the published run's own metrics."""
    metrics = latest_dir() / "metrics.json"
    if not metrics.exists():
        return {}
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    return {
        key: int(payload[key])
        for key in ("n_train", "n_test")
        if payload.get(key) is not None
    }


@st.cache_data(ttl=30, show_spinner=False)
def load_label_distribution() -> pd.Series | None:
    """Class distribution of the cleaned corpus (train + locked test)."""
    metrics = latest_dir() / "metrics.json"
    if not metrics.exists():
        return None
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    train = payload.get("per_class_n_train") or {}
    test = payload.get("per_class_n_test") or {}
    if not train and not test:
        return None
    names = sorted(set(train) | set(test))
    return pd.Series(
        {name: int(train.get(name, 0)) + int(test.get(name, 0)) for name in names}
    ).sort_values(ascending=False)


@st.cache_data(ttl=30, show_spinner=False)
def load_eda_summary() -> dict | None:
    p = METRICS_DIR / "eda_summary.json"
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=30, show_spinner=False)
def load_model_metrics(model_name: str) -> dict | None:
    p = latest_dir() / f"{model_name}_metrics.json"
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=30, show_spinner=False)
def load_comparison_csv() -> pd.DataFrame | None:
    latest_metrics = latest_dir() / "metrics.json"
    if not latest_metrics.exists():
        return None
    payload = json.loads(latest_metrics.read_text(encoding="utf-8"))
    rows = []
    for item in payload.get("models", []):
        row = dict(item)
        # Drop every dict/list-valued key before flattening into a DataFrame:
        # the comparison table formats each cell as a number, so a nested
        # per-class mapping would land in a cell and break the formatter.
        for key in NON_SCALAR_METRIC_KEYS:
            row.pop(key, None)
        row["model"] = item.get("model")
        rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows).set_index("model")


@st.cache_data(ttl=30, show_spinner=False)
def load_shap_top_features(model_name: str) -> dict | None:
    """Cached SHAP top-features JSON — avoids re-read on every class selectbox change."""
    p = shap_dir() / model_name / "top_features.json"
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def figures_dir() -> Path:
    return FIGURES_DIR


def shap_dir() -> Path:
    """SHAP output directory of the published run."""
    return latest_dir() / "shap"


def metrics_dir() -> Path:
    return METRICS_DIR


def confusion_matrix_path(model_name: str) -> Path | None:
    p = latest_dir() / f"{model_name}_confusion_matrix.png"
    return p if p.exists() else None


def classification_report_path(model_name: str) -> Path | None:
    p = latest_dir() / f"{model_name}_per_class.csv"
    return p if p.exists() else None


def active_labels() -> list[str]:
    latest_metrics = latest_dir() / "metrics.json"
    if latest_metrics.exists():
        try:
            payload = json.loads(latest_metrics.read_text(encoding="utf-8"))
            trained_classes = payload.get("class_names")
            if trained_classes:
                return [str(name) for name in trained_classes]
        except (OSError, json.JSONDecodeError, TypeError):
            pass
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
