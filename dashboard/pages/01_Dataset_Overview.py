"""Page 1 -- Dataset Overview."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard._shared import (
    active_labels,
    active_mode,
    cached_list_models,
    cfg,
    load_eda_summary,
    load_label_distribution,
    load_parquet_row_counts,
    warn_no_data,
)
from dashboard._style import (
    BG_CANVAS,
    apply_style,
    class_badge,
    class_color,
    hero,
    plotly_axis,
    plotly_config,
    plotly_layout,
    section,
)

st.set_page_config(page_title="Dataset Overview", page_icon=":bar_chart:", layout="wide")
apply_style()

hero(
    "Dataset Overview",
    "CICIDS2017 + CSE-CIC-IDS2018 (Canadian Institute for Cybersecurity) · "
    "18 CSVs · 13.9M cleaned flow records · 80 features",
)

labels     = active_labels()
mode       = active_mode()
row_counts = load_parquet_row_counts()   # ~1 ms via parquet metadata

k1, k2, k3, k4 = st.columns(4)
k1.metric("Mode",           mode.upper())
k2.metric("Target classes", len(labels))
k3.metric("Train rows",     f"{row_counts['train']:,}" if "train" in row_counts else "—")
k4.metric("Test rows",      f"{row_counts['test']:,}"  if "test"  in row_counts else "—")

if not row_counts:
    warn_no_data()
    st.stop()

# --- label distribution -------------------------------------------------
section("Label distribution", "Post-mapping class counts (after label normalisation)")

summary = load_eda_summary()
if summary and "label_distribution" in summary:
    import pandas as pd
    dist = pd.Series(summary["label_distribution"]).sort_values(ascending=False)
else:
    # Efficient: read only label column instead of full DataFrame
    dist = load_label_distribution()
    if dist is None:
        warn_no_data()
        st.stop()

ordered = dist.sort_values(ascending=True)
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
        height=380, showlegend=False,
        margin=dict(l=10, r=20, t=10, b=10),
        uirevision="dist",
    ),
    xaxis=plotly_axis(title="Records (log scale)", type="log"),
    yaxis=plotly_axis(),
    bargap=0.28,
)

c1, c2 = st.columns([3, 2])
with c1:
    st.plotly_chart(fig, use_container_width=True, config=plotly_config("class_distribution"))
with c2:
    total = int(dist.sum())
    rows = []
    for cls in dist.sort_values(ascending=False).index:
        cls_s = str(cls)
        count = int(dist[cls])
        pct   = 100 * count / total if total else 0
        rows.append(
            f"<tr>"
            f"  <td>{class_badge(cls_s)}</td>"
            f"  <td class='num'>{count:,}</td>"
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

# --- active label scheme -----------------------------------------------
section("Active label scheme", f"{mode} classification")
chips = " ".join(class_badge(lb) for lb in labels)
st.markdown(
    f'<div style="display:flex;flex-wrap:wrap;gap:8px;">{chips}</div>',
    unsafe_allow_html=True,
)

# --- config / dataset details ------------------------------------------
section("Data pipeline configuration")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Preprocessing**")
    st.json({k: cfg()[k] for k in ("classification", "preprocessing") if k in cfg()})
with c2:
    st.markdown("**Data sources**")
    st.json({k: cfg()["data"][k] for k in (
        "raw_dir", "subsample_n", "drop_other_class", "required_files",
    ) if k in cfg()["data"]})
