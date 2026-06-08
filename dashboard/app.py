"""Streamlit dashboard -- entry point.

Run with::

    streamlit run dashboard/app.py

Page routing uses Streamlit's native multi-page mechanism: every file
under ``dashboard/pages/`` becomes a page in the sidebar automatically.

This entry file shows the landing/welcome screen and configures global
state (page config, cached resource loaders).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure ``src`` is importable when Streamlit launches this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import (  # noqa: E402
    get_active_target_labels,
    get_classification_mode,
    load_config,
)
from src.inference.predictor import list_saved_models  # noqa: E402

st.set_page_config(
    page_title="AI-Based Cyber Attack Classification",
    page_icon=":shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_config():
    return load_config()


st.title("AI-Based Cyber Attack Classification")
st.caption("CICIDS2017 -- Random Forest, XGBoost, LightGBM, CatBoost, MLP")

cfg = get_config()
mode = get_classification_mode(cfg)
labels = get_active_target_labels(cfg)
saved = list_saved_models()

col1, col2, col3 = st.columns(3)
col1.metric("Classification mode", mode)
col2.metric("Target classes", len(labels))
col3.metric("Trained models", len(saved) if saved else 0)

st.markdown(
    """
    Use the sidebar to navigate:

    1. **Dataset Overview** -- CICIDS2017 summary and class distribution
    2. **EDA** -- feature distributions, correlations, missing-value audit
    3. **Model Performance** -- per-model metrics, confusion matrix
    4. **Model Comparison** -- RF vs XGBoost vs LightGBM vs CatBoost vs MLP
    5. **SHAP Explainability** -- what features drove each prediction
    6. **Predict New CSV** -- upload your own traffic CSV, get classifications
    """
)

with st.expander("Active configuration"):
    st.json({
        "classification_mode": mode,
        "labels": list(labels),
        "trained_models": saved,
        "raw_dir": cfg["data"]["raw_dir"],
        "subsample_n": cfg["data"].get("subsample_n"),
    })

with st.sidebar:
    st.header("About")
    st.caption("Senior project -- KMITL Faculty of Information Technology")
    st.caption("Advisor: Asst. Prof. Dr. Prapan Pavarangkoon")
