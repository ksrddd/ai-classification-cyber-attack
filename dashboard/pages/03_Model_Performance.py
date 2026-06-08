"""Page 3 -- Model Performance.

Per-model deep dive: metric table, confusion matrix, per-class breakdown.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard._shared import (
    figures_dir,
    load_model_metrics,
    metrics_dir,
    warn_no_models,
)
from src.inference.predictor import list_saved_models

st.title("Model Performance")

saved = list_saved_models()
if not saved:
    warn_no_models()
    st.stop()

model = st.selectbox("Model", saved, index=0)
m = load_model_metrics(model)
if m is None:
    st.warning(f"No metrics file for {model}. Run `--stage evaluate` to populate.")
    st.stop()

cols = st.columns(4)
cols[0].metric("Accuracy",         f"{m.get('accuracy', float('nan')):.4f}")
cols[1].metric("F1 (weighted)",    f"{m.get('f1_weighted', float('nan')):.4f}")
cols[2].metric("F1 (macro)",       f"{m.get('f1_macro', float('nan')):.4f}")
cols[3].metric("ROC-AUC",          f"{m.get('roc_auc', float('nan')):.4f}")

c1, c2 = st.columns(2)
c1.metric("Precision (weighted)", f"{m.get('precision_weighted', float('nan')):.4f}")
c1.metric("Recall (weighted)",    f"{m.get('recall_weighted', float('nan')):.4f}")
c2.metric("Precision (macro)",    f"{m.get('precision_macro', float('nan')):.4f}")
c2.metric("Recall (macro)",       f"{m.get('recall_macro', float('nan')):.4f}")

st.metric("Matthews correlation coef", f"{m.get('matthews_corrcoef', float('nan')):.4f}")

st.subheader("Confusion matrix (normalized by true)")
cm_path = figures_dir() / f"confusion_matrix_{model}.png"
if cm_path.exists():
    st.image(str(cm_path), use_container_width=True)
else:
    st.info("Run `python main.py --stage evaluate` to generate the confusion matrix.")

st.subheader("Per-class metrics")
per_class = m.get("per_class")
if per_class:
    df = pd.DataFrame(per_class).T
    st.dataframe(df.round(4))

st.subheader("Classification report")
report_path = metrics_dir() / f"classification_report_{model}.csv"
if report_path.exists():
    df = pd.read_csv(report_path, index_col=0)
    st.dataframe(df.round(4))
