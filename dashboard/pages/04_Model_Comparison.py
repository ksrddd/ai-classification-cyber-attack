"""Page 4 -- Model Comparison.

Side-by-side ranking + bar chart of every trained model.
"""

from __future__ import annotations

import streamlit as st

from dashboard._shared import (
    load_comparison_csv,
    metrics_dir,
    warn_no_models,
)

st.title("Model Comparison")

df = load_comparison_csv()
if df is None:
    warn_no_models()
    st.info("Run `python main.py --stage evaluate` to compute the comparison.")
    st.stop()

st.subheader("Test-set metrics")
st.dataframe(df.round(4))

st.subheader("Side-by-side bar chart")
png = metrics_dir() / "model_comparison.png"
if png.exists():
    st.image(str(png), use_container_width=True)
else:
    cols = [c for c in ("accuracy", "f1_weighted", "f1_macro", "roc_auc") if c in df.columns]
    if cols:
        st.bar_chart(df[cols])

st.subheader("Narrative")
md = (metrics_dir() / "model_comparison.md")
# Real report path is reports/model_comparison.md, but we wrote it via the
# comparison module which uses REPORTS_DIR. Try both for back-compat.
from src.config.constants import REPORTS_DIR
md_alt = REPORTS_DIR / "model_comparison.md"
if md_alt.exists():
    st.markdown(md_alt.read_text(encoding="utf-8"))
elif md.exists():
    st.markdown(md.read_text(encoding="utf-8"))
