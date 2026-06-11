"""Page 4 -- Model Comparison."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard._shared import load_comparison_csv, metrics_dir, warn_no_models
from dashboard._style import (
    BG_CANVAS,
    INK_0,
    apply_style,
    hero,
    model_color,
    model_label,
    plotly_axis,
    plotly_config,
    plotly_layout,
    section,
)
from src.config.constants import REPORTS_DIR

st.set_page_config(page_title="Model Comparison", page_icon=":trophy:", layout="wide")
apply_style()

hero(
    "Model Comparison",
    "Cross-model leaderboard on the held-out test set.",
)

df = load_comparison_csv()
if df is None:
    warn_no_models()
    st.info("Run `python main.py --stage evaluate` to compute the comparison.")
    st.stop()

rank_metric = st.selectbox(
    "Rank by",
    [c for c in ("f1_weighted", "f1_macro", "accuracy", "roc_auc") if c in df.columns],
    index=0,
)
ranked    = df.sort_values(rank_metric, ascending=False)
best_name = ranked.index[0]
medals    = ["1st", "2nd", "3rd"] + [f"{i+1}th" for i in range(3, len(ranked))]

c1, c2, c3 = st.columns(3)
c1.metric("Best model",             model_label(best_name))
c2.metric(f"Best {rank_metric}",    f"{ranked.iloc[0][rank_metric]:.4f}")
c3.metric("Models compared",        len(ranked))

# --- leaderboard --------------------------------------------------------
section("Leaderboard", f"Sorted by **{rank_metric}** (descending)")

metric_cols = [c for c in (
    "accuracy", "f1_weighted", "f1_macro",
    "precision_weighted", "recall_weighted", "roc_auc", "matthews_corrcoef",
) if c in ranked.columns]

header_html = (
    "<tr>"
    "<th style='width:64px;'>Rank</th>"
    "<th>Model</th>"
    + "".join(f"<th class='num'>{c}</th>" for c in metric_cols)
    + "</tr>"
)
body_rows = []
for i, (name, row) in enumerate(ranked.iterrows()):
    color    = model_color(name)
    is_best  = i == 0
    rank_html = (
        f'<span style="background:{color};color:{BG_CANVAS};font-weight:700;'
        f'padding:2px 10px;border-radius:6px;font-size:.75rem;white-space:nowrap;'
        f'font-family:\'JetBrains Mono\',monospace;">'
        f'{medals[i]}</span>'
    )
    name_html = (
        f'<span style="display:inline-flex;align-items:center;gap:8px;">'
        f'<span style="width:8px;height:8px;background:{color};border-radius:50%;'
        f'flex-shrink:0;box-shadow:0 0 6px {color};"></span>'
        f'<b style="color:{INK_0};">{model_label(name)}</b>'
        f'{"<span style=\'font-size:.7rem;color:#F59E0B;margin-left:4px;\'>★</span>" if is_best else ""}'
        f'</span>'
    )
    cells = "".join(
        f"<td class='num' style='color:{'#10B981' if is_best else '#A8AFC0'};'>"
        f"{row[c]:.4f}</td>"
        for c in metric_cols
    )
    body_rows.append(f"<tr><td>{rank_html}</td><td>{name_html}</td>{cells}</tr>")

st.markdown(
    '<div class="tbl-wrap"><table class="tbl">'
    f"<thead>{header_html}</thead>"
    f'<tbody>{"".join(body_rows)}</tbody>'
    "</table></div>",
    unsafe_allow_html=True,
)

# --- side-by-side bar chart --------------------------------------------
section("Side-by-side metrics")
metrics_to_plot = [c for c in (
    "accuracy", "f1_weighted", "f1_macro", "roc_auc",
) if c in ranked.columns]

bar_colors = ["#3B82F6", "#22D3EE", "#10B981", "#F59E0B"]
fig = go.Figure()
for j, metric_name in enumerate(metrics_to_plot):
    fig.add_trace(go.Bar(
        x=[model_label(m) for m in ranked.index],
        y=ranked[metric_name],
        name=metric_name,
        marker_color=bar_colors[j % len(bar_colors)],
        text=[f"{v:.3f}" for v in ranked[metric_name]],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=10, color=BG_CANVAS),
        hovertemplate="<b>%{x}</b><br>" + metric_name + ": %{y:.4f}<extra></extra>",
    ))
fig.update_layout(
    **plotly_layout(
        height=440,
        barmode="group",
        legend=dict(orientation="h", y=1.10, x=0),
        margin=dict(l=10, r=10, t=40, b=10),
        uirevision="comparison",
    ),
    xaxis=plotly_axis(),
    yaxis=plotly_axis(range=[0, 1.0], title="Score"),
)
st.plotly_chart(
    fig, use_container_width=True,
    config=plotly_config("model_comparison"),
)

# --- narrative report ---------------------------------------------------
section("Narrative report")
md_path = REPORTS_DIR / "model_comparison.md"
if md_path.exists():
    with st.expander("Show full report", expanded=False):
        st.markdown(md_path.read_text(encoding="utf-8"))
else:
    st.caption("Report file missing — run `--stage evaluate`.")

# --- download -----------------------------------------------------------
csv_path = metrics_dir() / "model_comparison.csv"
if csv_path.exists():
    with csv_path.open("rb") as f:
        st.download_button(
            "⬇ Download comparison CSV",
            data=f.read(),
            file_name="model_comparison.csv",
            mime="text/csv",
        )
