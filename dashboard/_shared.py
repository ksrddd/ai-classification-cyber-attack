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
    SHAP_DIR,
)
from src.config.loader import (  # noqa: E402
    get_active_target_labels,
    get_classification_mode,
    load_config,
)
from src.inference.predictor import list_saved_models  # noqa: E402


@st.cache_resource
def cfg():
    return load_config()


# ttl=60 so a new train run is picked up within 1 minute without restart
@st.cache_data(ttl=60, show_spinner=False)
def cached_list_models() -> list[str]:
    return list_saved_models()


@st.cache_data(show_spinner=False)
def load_parquet_row_counts() -> dict[str, int]:
    """Row counts via parquet metadata — ~1 ms, no column data loaded."""
    import pyarrow.parquet as pq
    result: dict[str, int] = {}
    for key, fname in [("train", "train.parquet"), ("test", "test.parquet")]:
        p = PROCESSED_DIR / fname
        if p.exists():
            result[key] = pq.read_metadata(str(p)).num_rows
    return result


@st.cache_data(show_spinner=False)
def load_train_parquet() -> pd.DataFrame | None:
    p = PROCESSED_DIR / "train.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False)
def load_test_parquet() -> pd.DataFrame | None:
    p = PROCESSED_DIR / "test.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
def load_eda_summary() -> dict | None:
    p = METRICS_DIR / "eda_summary.json"
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_model_metrics(model_name: str) -> dict | None:
    p = METRICS_DIR / f"{model_name}_test.json"
    if not p.exists():
        p = METRICS_DIR / f"{model_name}_val.json"
        if not p.exists():
            return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_comparison_csv() -> pd.DataFrame | None:
    p = METRICS_DIR / "model_comparison.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=0)


@st.cache_data(show_spinner=False)
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
