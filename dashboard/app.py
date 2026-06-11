"""Streamlit dashboard -- entry point.

Run with::

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard._shared import (  # noqa: E402
    active_labels,
    active_mode,
    cached_list_models,
    cfg,
)
from dashboard._style import (  # noqa: E402
    ACCENT,
    apply_style,
    footer,
    hero,
    pill,
    sidebar_header,
)

st.set_page_config(
    page_title="CyberML — Cyber Attack Classification",
    page_icon=":shield:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": "AI-Based Cyber Attack Classification · KMITL IT · Senior Project 2569",
    },
)

apply_style()

cfg_data   = cfg()
mode       = active_mode()
labels     = active_labels()
saved      = cached_list_models()

# --- hero ---------------------------------------------------------------
hero(
    "AI-Based Cyber Attack Classification",
    "CICIDS2017 · Random Forest · XGBoost · LightGBM · CatBoost · MLP",
)

# --- KPI strip ----------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Classification mode", mode.upper())
k2.metric("Target classes",      len(labels))
k3.metric("Models trained",      f"{len(saved)} / 5")
k4.metric("Dataset",             "CICIDS2017")

# --- pipeline status pills ----------------------------------------------
st.markdown("&nbsp;")
status_html = " &nbsp; ".join([
    pill("CICIDS2017 ready", "success"),
    pill(f"{len(saved)} models trained" if saved else "no models yet",
         "success" if saved else "warn"),
    pill("Streamlit dashboard", "accent"),
    pill("SHAP explainability", "accent"),
])
st.markdown(status_html, unsafe_allow_html=True)

# --- nav cards ----------------------------------------------------------
st.markdown("&nbsp;")
PAGES = [
    ("📊 Dataset Overview",    "CICIDS2017 summary, label distribution, post-mapping audit."),
    ("🔬 EDA",                 "Class balance, missing-value audit, correlation heatmap."),
    ("🎯 Model Performance",   "Per-model metrics, confusion matrix, classification report."),
    ("🏆 Model Comparison",    "Cross-model leaderboard and grouped bar chart."),
    ("💡 SHAP Explainability", "Top features per class, beeswarm summary, per-model drill-down."),
    ("🔮 Predict New CSV",     "Upload a CICIDS-format CSV, get predictions with probabilities."),
]
cols = st.columns(3)
for i, (name, desc) in enumerate(PAGES):
    with cols[i % 3]:
        st.markdown(
            f'<div class="navcard"><h4>{name}</h4><p>{desc}</p></div>',
            unsafe_allow_html=True,
        )

# --- config expander ----------------------------------------------------
with st.expander("Active configuration", expanded=False):
    st.json({
        "classification_mode": mode,
        "labels":              list(labels),
        "trained_models":      saved,
        "raw_dir":             cfg_data["data"]["raw_dir"],
        "subsample_n":         cfg_data["data"].get("subsample_n"),
    })

# --- sidebar ------------------------------------------------------------
with st.sidebar:
    sidebar_header(n_models=len(saved), n_classes=len(labels))
    st.divider()
    st.markdown(
        f'<div style="font-size:.75rem;text-transform:uppercase;'
        f'letter-spacing:.12em;color:#4A5163;font-weight:600;'
        f'margin-bottom:.5rem;">Authors</div>',
        unsafe_allow_html=True,
    )
    st.caption("Sirachet Chotthakunanan  ·  66070191")
    st.caption("Sukhum Rudeemaetakul  ·  66070315")
    st.markdown(
        f'<div style="font-size:.75rem;text-transform:uppercase;'
        f'letter-spacing:.12em;color:#4A5163;font-weight:600;'
        f'margin:.75rem 0 .5rem 0;">Advisor</div>',
        unsafe_allow_html=True,
    )
    st.caption("Asst. Prof. Dr. Prapan Pavarangkoon")

footer("CyberML · CICIDS2017 · KMITL IT · Senior Project 2569")
