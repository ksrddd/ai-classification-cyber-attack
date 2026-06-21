"""Page 3 -- Model Performance."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard._shared import (
    cached_list_models,
    confusion_matrix_path,
    load_model_metrics,
    classification_report_path,
    warn_no_models,
)
from dashboard._style import (
    BG_CANVAS,
    apply_style,
    class_color,
    hero,
    model_color,
    model_label,
    plotly_axis,
    plotly_config,
    plotly_layout,
    section,
)

st.set_page_config(page_title="Model Performance", page_icon=":dart:", layout="wide")
apply_style()

hero(
    "Model Performance",
    "Per-model metrics, confusion matrix, per-class breakdown.",
)

saved = cached_list_models()
if not saved:
    warn_no_models()
    st.stop()

model = st.selectbox(
    "Model",
    saved,
    index=0,
    format_func=model_label,
    help="Pick one of the trained models. Add more by running `python main.py --stage train`.",
)
m = load_model_metrics(model)
if m is None:
    st.warning(f"No metrics file for **{model_label(model)}**. Run `--stage evaluate`.")
    st.stop()

# --- headline KPIs ------------------------------------------------------
nan = float("nan")
mc  = model_color(model)
st.markdown(
    f'<div style="display:flex;gap:10px;align-items:center;margin-bottom:.4rem;">'
    f'  <span class="pill accent">{model_label(model)}</span>'
    f'  <span style="background:{mc};width:10px;height:10px;border-radius:50%;'
    f'  display:inline-block;box-shadow:0 0 8px {mc};"></span>'
    f'</div>',
    unsafe_allow_html=True,
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy",      f"{m.get('accuracy', nan):.4f}")
c2.metric("F1 (weighted)", f"{m.get('f1_weighted', nan):.4f}")
c3.metric("F1 (macro)",    f"{m.get('f1_macro', nan):.4f}")
c4.metric("ROC-AUC",       f"{m.get('roc_auc', nan):.4f}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Precision (W)", f"{m.get('precision_weighted', nan):.4f}")
c2.metric("Recall (W)",    f"{m.get('recall_weighted', nan):.4f}")
c3.metric("Precision (M)", f"{m.get('precision_macro', nan):.4f}")
c4.metric("MCC",           f"{m.get('matthews_corrcoef', nan):.4f}")

# --- radar (metric fingerprint) -----------------------------------------
section("Metric fingerprint", "Radar of weighted + macro scores")

radar_keys = (
    ("accuracy",           "Acc"),
    ("precision_weighted", "Prec (W)"),
    ("recall_weighted",    "Rec (W)"),
    ("f1_weighted",        "F1 (W)"),
    ("f1_macro",           "F1 (M)"),
    ("roc_auc",            "ROC-AUC"),
)
radar_vals = [m.get(k, 0.0) or 0.0 for k, _ in radar_keys]
radar_lbls = [lbl for _, lbl in radar_keys]

radar = go.Figure(
    go.Scatterpolar(
        r=radar_vals + [radar_vals[0]],
        theta=radar_lbls + [radar_lbls[0]],
        fill="toself",
        line=dict(color=mc, width=2),
        fillcolor=f"rgba({int(mc[1:3],16)},{int(mc[3:5],16)},{int(mc[5:7],16)},.12)",
        hovertemplate="<b>%{theta}</b>: %{r:.4f}<extra></extra>",
    )
)
radar.update_layout(
    **plotly_layout(
        height=400, showlegend=False,
        margin=dict(l=60, r=60, t=30, b=30),
        uirevision=model,
    ),
    polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(
            visible=True, range=[0, 1],
            tickfont=dict(color="#6C7488", size=10),
            gridcolor="rgba(255,255,255,.06)",
        ),
        angularaxis=dict(
            tickfont=dict(color="#E6E9F2", size=11),
            gridcolor="rgba(255,255,255,.06)",
        ),
    ),
)

c_left, c_right = st.columns([1, 1])
with c_left:
    st.plotly_chart(
        radar, use_container_width=True,
        config=plotly_config(f"radar_{model}"),
    )
with c_right:
    cm_path = confusion_matrix_path(model)
    st.caption("Confusion matrix (normalised by true class)")
    if cm_path is not None:
        st.image(str(cm_path), use_container_width=True)
    else:
        st.info("Run `--stage evaluate` to generate the confusion matrix.")

# --- per-class breakdown ------------------------------------------------
section("Per-class metrics", "Precision / Recall / F1 per attack family")
per_class = m.get("per_class")
if per_class:
    rows = []
    for cls, vals in per_class.items():
        rows.append({
            "Class":     cls,
            "Precision": vals.get("precision", 0.0),
            "Recall":    vals.get("recall",    0.0),
            "F1":        vals.get("f1",        0.0),
        })
    df = pd.DataFrame(rows)

    fig = go.Figure()
    metric_colors = {"Precision": "#3B82F6", "Recall": "#22D3EE", "F1": mc}
    for metric_name, bar_color in metric_colors.items():
        fig.add_trace(go.Bar(
            y=df["Class"],
            x=df[metric_name],
            orientation="h",
            name=metric_name,
            marker_color=bar_color,
            text=[f"{v:.3f}" for v in df[metric_name]],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=10, color=BG_CANVAS),
            hovertemplate="<b>%{y}</b><br>" + metric_name + ": %{x:.4f}<extra></extra>",
        ))
    fig.update_layout(
        **plotly_layout(
            height=max(280, 36 * len(df) + 80),
            barmode="group",
            legend=dict(orientation="h", y=1.10, x=0),
            margin=dict(l=10, r=20, t=40, b=10),
            uirevision=model,
        ),
        xaxis=plotly_axis(range=[0, 1.0]),
        yaxis=plotly_axis(),
    )
    st.plotly_chart(
        fig, use_container_width=True,
        config=plotly_config(f"per_class_{model}"),
    )
    st.dataframe(df.round(4), use_container_width=True, hide_index=True)

# --- classification report ---------------------------------------------
section("Classification report")
report_path = classification_report_path(model)
if report_path is not None:
    df = pd.read_csv(report_path, index_col=0)
    st.dataframe(df.round(4), use_container_width=True)
else:
    st.info("Run `--stage evaluate` to produce the report.")
