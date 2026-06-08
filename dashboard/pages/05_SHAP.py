"""Page 5 -- SHAP Explainability."""

from __future__ import annotations

import json

import streamlit as st

from dashboard._shared import shap_dir, warn_no_models
from src.inference.predictor import list_saved_models

st.title("SHAP Explainability")

saved = list_saved_models()
if not saved:
    warn_no_models()
    st.stop()

model = st.selectbox("Model", saved, index=0)
mdir = shap_dir() / model
if not mdir.exists():
    st.warning(f"No SHAP artefacts for {model}. Run `python main.py --stage explain` first.")
    st.stop()

bar_path = mdir / "summary_bar.png"
if bar_path.exists():
    st.subheader("Overall feature importance (mean |SHAP|)")
    st.image(str(bar_path), use_container_width=True)

top_json = mdir / "top_features.json"
if top_json.exists():
    with top_json.open("r", encoding="utf-8") as f:
        top = json.load(f)
    st.subheader("Top features overall")
    st.table([{"feature": f, "mean_abs_shap": v} for f, v in top.get("overall", [])])

    st.subheader("Top features per class")
    per_class = top.get("per_class", {})
    if per_class:
        cls = st.selectbox("Class", list(per_class.keys()))
        st.table([{"feature": f, "mean_abs_shap": v} for f, v in per_class[cls]])

st.subheader("Per-class summary plots")
summary_pngs = sorted(p for p in mdir.glob("summary_*.png") if p.name != "summary_bar.png")
for png in summary_pngs:
    st.caption(png.stem.replace("summary_", "Class: ").replace("_", " "))
    st.image(str(png), use_container_width=True)
