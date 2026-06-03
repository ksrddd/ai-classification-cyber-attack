"""Streamlit dashboard — entry point.

Run with::

    streamlit run dashboard/app.py

Page routing uses Streamlit's native multi-page mechanism: every file
under ``dashboard/pages/`` becomes a page in the sidebar automatically.

This entry file shows the landing/welcome screen and configures global
state (page config, cached resource loaders).

Phase 10 implementation — this file ships a skeleton so the structure is
visible from day one and the build doesn't break when you ``streamlit run``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure ``src`` is importable when Streamlit launches this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="AI-Based Cyber Attack Classification",
    page_icon=":shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AI-Based Cyber Attack Classification")
st.markdown(
    """
    Senior Project — KMITL Faculty of Information Technology.

    Use the sidebar to navigate:

    1. **Dataset Overview** — CICIDS2017 summary and class distribution
    2. **EDA** — feature distributions, correlations, missing-value audit
    3. **Model Performance** — per-model metrics, confusion matrix
    4. **Model Comparison** — Logistic Regression vs Random Forest vs MLP
    5. **SHAP Explainability** — what features drove each prediction
    6. **Predict New CSV** — upload your own traffic CSV, get classifications

    *Status: scaffolded in Phase 3 — pages are wired up but show placeholders
    until Phases 4-10 land.*
    """
)

with st.sidebar:
    st.header("About")
    st.caption("Built by Sirachet & Sukhum")
    st.caption("Advisor: Asst. Prof. Dr. Prapan Pavarangkoon")
