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


@st.cache_resource
def cfg():
    return load_config()


@st.cache_data
def load_train_parquet() -> pd.DataFrame | None:
    p = PROCESSED_DIR / "train.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data
def load_test_parquet() -> pd.DataFrame | None:
    p = PROCESSED_DIR / "test.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data
def load_eda_summary() -> dict | None:
    p = METRICS_DIR / "eda_summary.json"
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_model_metrics(model_name: str) -> dict | None:
    p = METRICS_DIR / f"{model_name}_test.json"
    if not p.exists():
        # Fallback to val metrics if test hasn't run.
        p = METRICS_DIR / f"{model_name}_val.json"
        if not p.exists():
            return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_comparison_csv() -> pd.DataFrame | None:
    p = METRICS_DIR / "model_comparison.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=0)


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
        "Run `python main.py --stage preprocess` first."
    )


def warn_no_models() -> None:
    st.warning(
        "No trained models found. "
        "Run `python main.py --stage train` (and optionally `--stage evaluate`) first."
    )
