"""Page 5 -- SHAP Explainability."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard._shared import (
    cached_list_models,
    load_shap_top_features,
    shap_dir,
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

st.set_page_config(page_title="SHAP Explainability", page_icon=":bulb:", layout="wide")
apply_style()

hero(
    "SHAP Explainability",
    "TreeExplainer (RF / XGB / LightGBM / CatBoost) · KernelExplainer (MLP / LR).",
)

saved = cached_list_models()
if not saved:
    warn_no_models()
    st.stop()

c1, c2 = st.columns([1, 3])
with c1:
    model = st.selectbox("Model", saved, index=0, format_func=model_label)

mdir = shap_dir() / model
if not mdir.exists():
    st.warning(
        f"No SHAP artefacts for **{model_label(model)}**. "
        "Run `python main.py --stage explain`."
    )
    st.stop()

# Cached JSON read — no re-read on class selectbox interactions
top = load_shap_top_features(model)

# --- KPIs ---------------------------------------------------------------
k1, k2, k3 = st.columns(3)
if top and top.get("overall"):
    k1.metric("Top-1 feature",    top["overall"][0][0])
    k1.caption(f"mean |SHAP| = {top['overall'][0][1]:.4f}")
n_classes = len(top.get("per_class", {})) if top else 0
k2.metric("Classes explained", n_classes)
explainer = "TreeExplainer" if model not in ("mlp", "logistic_regression") else "KernelExplainer"
k3.metric("Explainer", explainer)

# --- overall importance bar --------------------------------------------
section("Overall feature importance", "Mean |SHAP| across all classes and samples")
if top and top.get("overall"):
    feats = [f for f, _ in top["overall"]][::-1]
    vals  = [v for _, v in top["overall"]][::-1]
    mc    = model_color(model)
    fig   = go.Figure(
        go.Bar(
            x=vals,
            y=feats,
            orientation="h",
            marker=dict(
                color=mc,
                line=dict(color="rgba(255,255,255,.08)", width=1),
            ),
            text=[f"{v:.4f}" for v in vals],
            textposition="inside",
            insidetextanchor="end",
            textfont=dict(color=BG_CANVAS, size=10, family="JetBrains Mono, monospace"),
            hovertemplate="<b>%{y}</b><br>mean |SHAP| = %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        **plotly_layout(
            height=max(280, 28 * len(feats) + 60),
            showlegend=False,
            margin=dict(l=10, r=20, t=10, b=10),
            uirevision=model,
        ),
        xaxis=plotly_axis(title="mean |SHAP|"),
        yaxis=plotly_axis(),
    )
    st.plotly_chart(
        fig, use_container_width=True,
        config=plotly_config(f"shap_overall_{model}"),
    )

bar_path = mdir / "summary_bar.png"
if bar_path.exists():
    with st.expander("Static mean |SHAP| bar (matplotlib)"):
        st.image(str(bar_path), use_container_width=True)

# --- per-class drill-down ----------------------------------------------
section("Per-class feature importance", "Pick a class to inspect its top features")
per_class = top.get("per_class", {}) if top else {}
if not per_class:
    st.info("No per-class data found.")
else:
    classes  = list(per_class.keys())
    cc1, _cc2 = st.columns([1, 3])
    with cc1:
        cls = st.selectbox("Class", classes, key="shap_class")

    cc = class_color(cls)
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin:.4rem 0 .8rem 0;">'
        f'<span style="width:12px;height:12px;background:{cc};border-radius:3px;'
        f'flex-shrink:0;box-shadow:0 0 6px {cc};"></span>'
        f'<b style="font-size:1rem;color:#E6E9F2;">{cls}</b>'
        f'</div>',
        unsafe_allow_html=True,
    )

    items = per_class[cls]
    if not items:
        st.caption("No data for this class.")
    else:
        feats = [f for f, _ in items][::-1]
        vals  = [v for _, v in items][::-1]
        fig   = go.Figure(
            go.Bar(
                x=vals,
                y=feats,
                orientation="h",
                marker=dict(
                    color=cc,
                    line=dict(color="rgba(255,255,255,.08)", width=1),
                ),
                text=[f"{v:.4f}" for v in vals],
                textposition="inside",
                insidetextanchor="end",
                textfont=dict(color=BG_CANVAS, size=10, family="JetBrains Mono, monospace"),
                hovertemplate="<b>%{y}</b><br>mean |SHAP| = %{x:.4f}<extra></extra>",
            )
        )
        fig.update_layout(
            **plotly_layout(
                height=max(220, 26 * len(feats) + 40),
                showlegend=False,
                margin=dict(l=10, r=20, t=10, b=10),
                # uirevision per class so zoom resets only when class changes
                uirevision=cls,
            ),
            xaxis=plotly_axis(title=f"mean |SHAP| for {cls}"),
            yaxis=plotly_axis(),
        )
        st.plotly_chart(
            fig, use_container_width=True,
            config=plotly_config(f"shap_{model}_{cls}"),
        )

        slug = "".join(c if c.isalnum() else "_" for c in cls).strip("_") or "class"
        png  = mdir / f"summary_{slug}.png"
        if png.exists():
            with st.expander(f"Beeswarm summary plot — {cls}"):
                st.image(str(png), use_container_width=True)
