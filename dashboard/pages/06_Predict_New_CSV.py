"""Page 6 — Predict New CSV.

User uploads a CICIDS-format CSV; we validate the schema, run inference
through the saved Pipeline (scaler + best Random Forest), display the
predictions, and offer a CSV download of the result.
"""

from __future__ import annotations

import streamlit as st

st.title("Predict New CSV")

st.info("Phase 10 implementation pending.")

uploaded = st.file_uploader("Upload a network-flow CSV", type=["csv"])
if uploaded is not None:
    st.warning(
        "Inference pipeline not wired up yet — comes online in Phase 12. "
        "The schema validator runs in Phase 10."
    )

# TODO(phase 10/12):
#   1. validator.validate_inference_csv(df, expected_features)
#   2. pipelines.predict.run(uploaded_path)
#   3. Display table with predicted class + probability per row.
#   4. st.download_button for predictions.csv.
