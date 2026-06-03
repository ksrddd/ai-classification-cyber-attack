"""Page 1 — Dataset Overview.

Shows: source description, row/column counts, target_labels, class balance,
license + download link. Pulls from ``data/sample/`` so it works without
the user having downloaded the full CICIDS CSVs.
"""

from __future__ import annotations

import streamlit as st

st.title("Dataset Overview")

st.info(
    "Phase 10 implementation pending. This page will show: source description, "
    "row/column counts, target labels, class balance, license + download link."
)

# TODO(phase 10): load data/sample/sample.parquet, show df.head(), shape,
# class distribution bar chart, link to UNB CIC CICIDS2017 page.
