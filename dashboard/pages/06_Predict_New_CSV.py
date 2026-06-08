"""Page 6 -- Predict New CSV.

Upload a CICIDS-format flow CSV, validate the schema, run inference, see
predicted class + probabilities. Offers a download of the result.
"""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st

from dashboard._shared import warn_no_models
from src.data.schema import clean_column_names
from src.features.validator import load_expected_features, validate_inference_csv
from src.inference.predictor import list_saved_models, predict_dataframe

st.title("Predict New CSV")

saved = list_saved_models()
if not saved:
    warn_no_models()
    st.stop()

model = st.selectbox("Model", saved, index=0)
uploaded = st.file_uploader("Upload a network-flow CSV (CICIDS2017 schema)", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV with the 78 CICIDS flow features (column names will be auto-stripped).")
    st.stop()

try:
    df = pd.read_csv(uploaded, low_memory=False, encoding="latin-1")
except Exception as exc:
    st.error(f"Failed to read CSV: {exc}")
    st.stop()

df = clean_column_names(df)
try:
    expected = load_expected_features()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

report = validate_inference_csv(df, expected_features=expected)
if not report.ok:
    st.error(report.message)
    st.write({"missing": report.missing[:20], "extra": report.extra[:20]})
    st.stop()
st.success(report.message)

with st.spinner(f"Running inference with {model} ..."):
    try:
        result = predict_dataframe(df, model_name=model, include_probabilities=True)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Inference failed: {exc}")
        st.stop()

st.subheader("Predictions")
preds = result.predictions
st.dataframe(preds.head(50))

# Quick summary.
st.subheader("Predicted class distribution")
st.bar_chart(preds["predicted_label"].value_counts())

buf = BytesIO()
preds.to_csv(buf, index=False)
buf.seek(0)
st.download_button(
    "Download predictions CSV",
    data=buf,
    file_name=f"predictions_{model}.csv",
    mime="text/csv",
)
