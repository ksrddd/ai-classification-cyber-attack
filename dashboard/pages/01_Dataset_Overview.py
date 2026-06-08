"""Page 1 -- Dataset Overview.

Source description, row/column counts, target labels, class balance.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard._shared import (
    active_labels,
    active_mode,
    cfg,
    load_eda_summary,
    load_test_parquet,
    load_train_parquet,
    warn_no_data,
)

st.title("Dataset Overview")

st.markdown(
    """
    **Source:** CICIDS2017 (Canadian Institute for Cybersecurity).
    [Project page](https://www.unb.ca/cic/datasets/ids-2017.html).

    Raw input: 8 CSVs of pre-extracted flow features (~2.8M rows total).
    Labels are normalized into a {mode} classification scheme.
    """.format(mode=active_mode())
)

col1, col2 = st.columns(2)
col1.metric("Active mode", active_mode())
col2.metric("Target classes", len(active_labels()))

train = load_train_parquet()
test = load_test_parquet()

if train is None or test is None:
    warn_no_data()
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Train rows", f"{len(train):,}")
c2.metric("Test rows", f"{len(test):,}")
c3.metric("Features", f"{train.shape[1] - 1:,}")

st.subheader("Label distribution (post-mapping)")
summary = load_eda_summary()
if summary is not None and "label_distribution" in summary:
    dist = pd.Series(summary["label_distribution"]).sort_values(ascending=False)
    st.bar_chart(dist)
    st.dataframe(dist.rename("count").to_frame())
else:
    # Fallback -- compute from the train parquet.
    from src.pipelines.preprocess import LABEL_ENCODED_COLUMN
    counts = train[LABEL_ENCODED_COLUMN].value_counts().sort_index()
    st.bar_chart(counts)
    st.dataframe(counts.rename("count").to_frame())

with st.expander("Configured target labels"):
    st.write(active_labels())

with st.expander("Configuration excerpt"):
    st.json({k: cfg()[k] for k in ("classification", "data", "preprocessing") if k in cfg()})
