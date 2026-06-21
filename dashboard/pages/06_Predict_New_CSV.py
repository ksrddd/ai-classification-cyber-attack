"""Page 6 -- Predict New CSV."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard._shared import cached_list_models, warn_no_models
from dashboard._style import (
    BG_CANVAS,
    apply_style,
    class_badge,
    class_color,
    hero,
    model_label,
    pill,
    plotly_axis,
    plotly_config,
    plotly_layout,
    section,
)
from src.data.schema import clean_column_names
from src.features.validator import load_expected_features, validate_inference_csv
from src.inference.predictor import predict_dataframe

st.set_page_config(page_title="Predict New CSV", page_icon=":crystal_ball:", layout="wide")
apply_style()

hero(
    "Predict New CSV",
    "Upload a combined CICIDS2017/CSE-CIC-IDS2018 flow CSV · schema validation · per-row predictions + probabilities.",
)

saved = cached_list_models()
if not saved:
    warn_no_models()
    st.stop()

# --- controls -----------------------------------------------------------
c1, c2 = st.columns([1, 2])
with c1:
    model = st.selectbox(
        "Model", saved, index=0, format_func=model_label,
        help="Trained Pipeline to use for inference.",
    )
with c2:
    uploaded = st.file_uploader(
        "Upload a network-flow CSV",
        type=["csv"],
        help="Must follow the combined CICIDS2017/CSE-CIC-IDS2018 80-feature schema. "
             "Column names are auto-stripped. Need a sample? "
             "Run `python scripts/generate_sample.py`.",
    )

if uploaded is None:
    st.info(
        "Drop a CSV with the 80 combined CICIDS2017/CSE-CIC-IDS2018 flow features.  \n"
        "Need a test file? Run `python scripts/generate_sample.py` "
        "to generate `data/sample/synthetic_cicids.csv`.",
        icon="ℹ️",
    )
    st.stop()

# --- load + validate ----------------------------------------------------
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

c1, c2, c3 = st.columns(3)
c1.metric("Rows",    f"{report.n_rows:,}")
c2.metric("Columns", f"{report.n_cols:,}")
c3.metric("Model",   model_label(model))

if not report.ok:
    st.error(report.message)
    with st.expander("Missing / extra columns"):
        st.write({"missing": report.missing[:30], "extra": report.extra[:30]})
    st.stop()

st.markdown(pill(report.message, "success"), unsafe_allow_html=True)

# --- inference ----------------------------------------------------------
section("Predictions", f"Running {model_label(model)} on {report.n_rows:,} rows...")
with st.spinner("Predicting..."):
    try:
        result = predict_dataframe(df, model_name=model, include_probabilities=True)
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        st.stop()
preds = result.predictions

# --- summary charts -----------------------------------------------------
class_counts = preds["predicted_label"].value_counts()
c1, c2 = st.columns([3, 2])

with c1:
    section("Predicted class distribution")
    ordered = class_counts.sort_values(ascending=True)
    fig = go.Figure(
        go.Bar(
            x=ordered.values,
            y=ordered.index.astype(str),
            orientation="h",
            marker=dict(
                color=[class_color(str(idx)) for idx in ordered.index],
                line=dict(color="rgba(255,255,255,.08)", width=1),
            ),
            text=[f"{int(v):,}" for v in ordered.values],
            textposition="inside",
            insidetextanchor="end",
            textfont=dict(color=BG_CANVAS, size=11, family="JetBrains Mono, monospace"),
            hovertemplate="<b>%{y}</b><br>%{x:,} rows<extra></extra>",
        )
    )
    fig.update_layout(
        **plotly_layout(
            height=max(260, 30 * len(ordered) + 60),
            showlegend=False,
            margin=dict(l=10, r=20, t=10, b=10),
            uirevision="predict_dist",
        ),
        xaxis=plotly_axis(title="Records"),
        yaxis=plotly_axis(),
    )
    st.plotly_chart(
        fig, use_container_width=True,
        config=plotly_config("predicted_distribution"),
    )

with c2:
    section("Class counts")
    total = int(class_counts.sum())
    rows  = []
    for cls, n in class_counts.items():
        pct = 100 * n / total
        rows.append(
            f"<tr>"
            f"  <td>{class_badge(str(cls))}</td>"
            f"  <td class='num'>{int(n):,}</td>"
            f"  <td class='num dim'>{pct:.1f}%</td>"
            f"</tr>"
        )
    st.markdown(
        '<div class="tbl-wrap"><table class="tbl">'
        "<thead><tr>"
        "<th>Class</th>"
        "<th class='num'>Count</th>"
        "<th class='num'>Share</th>"
        "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table></div>",
        unsafe_allow_html=True,
    )

# --- prediction table ---------------------------------------------------
section("Prediction details", f"First 50 rows — download the full result below.")

head    = preds.head(50).copy()
display = head.copy()
if "max_proba" in display.columns:
    display["max_proba"] = display["max_proba"].map(lambda v: f"{v:.3f}")
for c in (col for col in display.columns if col.startswith("proba_")):
    display[c] = display[c].map(lambda v: f"{v:.3f}")

st.dataframe(display, use_container_width=True, hide_index=True)

# --- download -----------------------------------------------------------
buf = BytesIO()
preds.to_csv(buf, index=False)
buf.seek(0)
st.download_button(
    "⬇️ Download all predictions (CSV)",
    data=buf,
    file_name=f"predictions_{model}.csv",
    mime="text/csv",
)
