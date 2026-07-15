"""Shared styling + design tokens for the dashboard.

Design system extracted from CyberML_Console.html.
Single source of truth for colours, CSS, and HTML helpers.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Design tokens  (mirror CyberML_Console.html)
# ---------------------------------------------------------------------------

# Canvas / surface layers
BG_CANVAS   = "#07090E"
BG_PAGE     = "#0B0E15"   # surface default
BG_PANEL    = "#11151F"   # surface-raised
BG_ELEVATED = "#161B27"   # surface-elevated
BG_HOVER    = "#1A2030"

# Borders
BORDER        = "rgba(255,255,255,0.08)"   # line-base
BORDER_SUBTLE = "rgba(255,255,255,0.05)"   # line-subtle

# Typography
INK_0 = "#E6E9F2"   # primary
INK_1 = "#A8AFC0"   # dim
INK_2 = "#6C7488"   # muted
INK_3 = "#4A5163"   # very muted

# Brand
ACCENT      = "#22D3EE"   # brand-cyan
ACCENT_BLUE = "#3B82F6"   # brand-blue
ACCENT_IND  = "#6366F1"   # brand-indigo
ACCENT_DEEP = "#0891B2"   # legacy alias

# Status
COLOR_SUCCESS  = "#10B981"   # ok / emerald
COLOR_INFO     = "#38BDF8"   # sky
COLOR_WARN     = "#F59E0B"   # amber
COLOR_DANGER   = "#EF4444"   # red
COLOR_CRITICAL = "#F43F5E"   # rose

# Legacy alias used in older template references
TEXT_DIM = INK_1

# ---------------------------------------------------------------------------
# Per-class & per-model palettes
# ---------------------------------------------------------------------------

CLASS_COLORS: dict[str, str] = {
    "BENIGN":       "#10B981",
    "DoS":          "#EF4444",
    "DDoS":         "#F43F5E",
    "PortScan":     "#F59E0B",
    "Bot":          "#A855F7",
    "Web Attack":   "#EC4899",
    "Brute Force":  "#F97316",
    "Infiltration": "#6366F1",
    "Heartbleed":   "#F43F5E",
    "Other":        "#6C7488",
    "Normal":       "#10B981",
    "Attack":       "#EF4444",
}

MODEL_COLORS: dict[str, str] = {
    "random_forest":       "#10B981",
    "xgboost":             "#3B82F6",
    "lightgbm":            "#A855F7",
    "catboost":            "#F59E0B",
    "mlp":                 "#EC4899",
    "logistic_regression": "#6C7488",
    "stacking":            "#14B8A6",
}

MODEL_LABELS: dict[str, str] = {
    "random_forest":       "Random Forest",
    "xgboost":             "XGBoost",
    "lightgbm":            "LightGBM",
    "catboost":            "CatBoost",
    "mlp":                 "MLP (sklearn)",
    "logistic_regression": "Logistic Regression",
    "stacking":            "Stacking Ensemble",
}


def class_color(name: str, default: str = INK_2) -> str:
    return CLASS_COLORS.get(name, default)


def model_color(name: str, default: str = ACCENT) -> str:
    return MODEL_COLORS.get(name, default)


def model_label(name: str) -> str:
    return MODEL_LABELS.get(name, name.replace("_", " ").title())


# ---------------------------------------------------------------------------
# CSS  — extracted visual DNA from CyberML_Console.html
# ---------------------------------------------------------------------------

_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap">

<style>
  /* ── Typography ── */
  html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
    letter-spacing: -0.005em;
  }
  code, pre, .font-mono {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
  }

  /* ── Canvas background with radial glow ── */
  html, body {
    background: #07090E !important;
  }
  [data-testid="stAppViewContainer"] {
    background:
      radial-gradient(ellipse 90% 50% at 50% 0%, rgba(59,130,246,.07), transparent 60%),
      radial-gradient(ellipse 55% 35% at 92% 18%, rgba(34,211,238,.05), transparent 60%),
      #07090E !important;
    min-height: 100vh;
  }

  /* ── Subtle grid backdrop ── */
  [data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
      linear-gradient(rgba(255,255,255,0.022) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,0.022) 1px, transparent 1px);
    background-size: 48px 48px;
    mask-image: radial-gradient(ellipse 80% 70% at 50% 25%, #000 20%, transparent 75%);
  }

  /* ── Page layout ── */
  .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1300px;
    position: relative;
    z-index: 1;
  }
  h1, h2, h3, h4 { letter-spacing: -0.02em; color: #E6E9F2; }
  h1 { font-weight: 700; }
  p  { color: #A8AFC0; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar              { width: 6px; height: 6px; }
  ::-webkit-scrollbar-thumb        { background: rgba(255,255,255,.07); border-radius: 6px; }
  ::-webkit-scrollbar-thumb:hover  { background: rgba(255,255,255,.13); }
  ::-webkit-scrollbar-track        { background: transparent; }

  /* ── Selection ── */
  ::selection { background: rgba(56,189,248,.30); color: #fff; }

  /* ── Animations ── */
  @keyframes pulseDot {
    0%,100% { opacity: 1; }
    50%      { opacity: .4; }
  }
  @keyframes scanline {
    from { transform: translateY(-100%); }
    to   { transform: translateY(200%); }
  }

  /* ── Hero strip ── */
  .hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg,
      rgba(59,130,246,.14) 0%,
      rgba(99,102,241,.12) 50%,
      rgba(34,211,238,.10) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.4rem 1.75rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 1px 0 rgba(0,0,0,.25), inset 0 0 0 1px rgba(255,255,255,.06);
  }
  .hero::after {
    content: '';
    position: absolute;
    left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(34,211,238,.35), transparent);
    animation: scanline 7s linear infinite;
    pointer-events: none;
    opacity: .5;
  }
  .hero h1 {
    margin: 0 0 .25rem 0;
    font-size: 1.65rem;
    font-weight: 700;
    background: linear-gradient(90deg, #22D3EE 0%, #A855F7 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
  }
  .hero p {
    color: #6C7488;
    margin: 0;
    font-size: .93rem;
  }

  /* ── Metric cards ── */
  div[data-testid="stMetric"] {
    background: #11151F;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.15rem;
    box-shadow: 0 1px 0 rgba(0,0,0,.20), inset 0 0 0 1px rgba(255,255,255,.05);
    transition: border-color .14s ease, transform .14s ease;
  }
  div[data-testid="stMetric"]:hover {
    border-color: rgba(34,211,238,.40);
    transform: translateY(-1px);
  }
  div[data-testid="stMetricLabel"] p {
    color: #6C7488;
    font-size: .75rem;
    text-transform: uppercase;
    letter-spacing: .10em;
    font-weight: 500;
  }
  div[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 600;
    color: #E6E9F2;
    font-variant-numeric: tabular-nums;
    font-feature-settings: 'tnum' on;
  }

  /* ── Pills / badges ── */
  .pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 2px 10px;
    border-radius: 6px;
    font-size: .75rem;
    font-weight: 600;
    white-space: nowrap;
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,0.08);
    color: #A8AFC0;
  }
  .pill.success {
    background: rgba(16,185,129,.10);
    color: #10B981;
    border-color: rgba(16,185,129,.25);
  }
  .pill.success::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #10B981;
    box-shadow: 0 0 6px #10B981;
    animation: pulseDot 1.6s ease-in-out infinite;
  }
  .pill.warn   { background: rgba(245,158,11,.10); color: #F59E0B; border-color: rgba(245,158,11,.25); }
  .pill.danger { background: rgba(239,68,68,.10);  color: #EF4444; border-color: rgba(239,68,68,.25); }
  .pill.info   { background: rgba(56,189,248,.10); color: #38BDF8; border-color: rgba(56,189,248,.25); }
  .pill.accent { background: rgba(34,211,238,.10); color: #22D3EE; border-color: rgba(34,211,238,.28); }

  /* ── Class badge ── */
  .class-badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 5px;
    font-size: .75rem;
    font-weight: 700;
    color: #07090E;
    white-space: nowrap;
  }

  /* ── Handcrafted tables ── */
  .tbl { width: 100%; border-collapse: collapse; font-size: .86rem; }
  .tbl th {
    text-align: left;
    padding: 8px 14px;
    color: #6C7488;
    font-weight: 500;
    font-size: .72rem;
    text-transform: uppercase;
    letter-spacing: .10em;
    border-bottom: 1px solid rgba(255,255,255,0.08);
  }
  .tbl td {
    padding: 8px 14px;
    color: #A8AFC0;
    border-bottom: 1px solid rgba(255,255,255,.04);
  }
  .tbl tr:last-child td { border-bottom: none; }
  .tbl tr:hover td     { background: rgba(34,211,238,.04); }
  .tbl-wrap {
    background: #11151F;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 1px 0 rgba(0,0,0,.20), inset 0 0 0 1px rgba(255,255,255,.05);
  }
  .num { text-align: right; font-variant-numeric: tabular-nums; font-family: 'JetBrains Mono', monospace; }
  .dim { color: #6C7488; }

  /* ── Nav cards (landing page) ── */
  .navcard {
    background: #11151F;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.15rem;
    height: 100%;
    box-shadow: 0 1px 0 rgba(0,0,0,.20), inset 0 0 0 1px rgba(255,255,255,.05);
    transition: border-color .14s ease, transform .14s ease;
  }
  .navcard:hover { border-color: rgba(34,211,238,.40); transform: translateY(-1px); }
  .navcard h4 { margin: 0 0 .3rem 0; color: #22D3EE; font-size: .95rem; font-weight: 600; }
  .navcard p  { margin: 0; color: #6C7488; font-size: .86rem; line-height: 1.5; }

  /* ── Section header ── */
  .section {
    border-left: 3px solid #22D3EE;
    padding-left: 12px;
    margin: 1.6rem 0 .8rem 0;
  }
  .section h3 { margin: 0; font-size: 1.1rem; font-weight: 600; color: #E6E9F2; }
  .section .sub { color: #6C7488; font-size: .82rem; margin-top: 3px; }

  /* ── Dataframes ── */
  div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08) !important;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #0B0E15 !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
  }
  [data-testid="stSidebarContent"] { background: transparent !important; }

  /* ── Selectbox / inputs ── */
  [data-testid="stSelectbox"] > div > div {
    background: #11151F !important;
    border-color: rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    color: #E6E9F2 !important;
  }
  [data-testid="stSelectbox"] > div > div:focus-within {
    border-color: rgba(34,211,238,.45) !important;
    box-shadow: 0 0 0 1px rgba(34,211,238,.20) !important;
  }

  /* ── Expander ── */
  [data-testid="stExpander"] {
    background: #11151F !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
  }

  /* ── Tabs ── */
  [data-testid="stTabs"] [role="tab"] {
    color: #6C7488;
    font-size: .82rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: .08em;
  }
  [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #22D3EE !important;
  }
  [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: #22D3EE !important;
  }

  /* ── Alerts / info boxes ── */
  [data-testid="stAlert"] {
    background: #11151F !important;
    border-radius: 10px !important;
    border-left-width: 3px !important;
  }

  /* ── Footer ── */
  .footer {
    text-align: center;
    color: #4A5163;
    font-size: .75rem;
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255,255,255,0.05);
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: .04em;
  }
</style>
"""


def apply_style() -> None:
    """Inject the project CSS into the current page. Call once per page."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


def hero(title: str, subtitle: str | None = None) -> None:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="hero"><h1>{title}</h1>{sub}</div>', unsafe_allow_html=True)


def section(title: str, subtitle: str | None = None) -> None:
    sub = f'<div class="sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="section"><h3>{title}</h3>{sub}</div>',
        unsafe_allow_html=True,
    )


def pill(text: str, kind: str = "info") -> str:
    kind = kind if kind in {"success", "warn", "danger", "info", "accent"} else "info"
    return f'<span class="pill {kind}">{text}</span>'


def class_badge(label: str) -> str:
    color = class_color(label)
    return f'<span class="class-badge" style="background:{color}">{label}</span>'


def footer(text: str = "AI-Based Cyber Attack Classification") -> None:
    st.markdown(f'<div class="footer">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly defaults
# ---------------------------------------------------------------------------


def plotly_layout(**overrides) -> dict:
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK_1, family="Inter, system-ui, sans-serif", size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, borderwidth=1, font=dict(color=INK_1)),
        colorway=[ACCENT, "#A855F7", "#10B981", "#F59E0B", "#EC4899", ACCENT_BLUE],
    )
    layout.update(overrides)
    return layout


def plotly_axis(**overrides) -> dict:
    axis = dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,.05)",
        zeroline=False,
        tickfont=dict(color=INK_2, size=11),
        title=dict(font=dict(color=INK_2, size=12)),
        automargin=True,
    )
    axis.update(overrides)
    return axis


def plotly_config(filename: str = "chart") -> dict:
    """Streamlit plotly_chart `config` arg — clean mode bar with PNG export."""
    return {
        "displayModeBar": True,
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "png",
            "filename": filename,
            "scale": 2,
        },
        "modeBarButtonsToRemove": [
            "pan2d", "lasso2d", "select2d", "autoScale2d",
            "zoomIn2d", "zoomOut2d",
        ],
    }


def sidebar_header(n_models: int, n_classes: int) -> None:
    """Branded sidebar header with live-session indicator and project stats."""
    st.markdown(
        f"""
        <div style="margin-bottom:1rem;">
          <div style="display:flex;align-items:center;gap:10px;
                      background:#11151F;border:1px solid rgba(255,255,255,.08);
                      border-radius:10px;padding:10px 12px;margin-bottom:10px;">
            <div style="width:32px;height:32px;border-radius:8px;flex-shrink:0;
                        background:linear-gradient(135deg,#3B82F6,#22D3EE);
                        display:flex;align-items:center;justify-content:center;
                        font-size:14px;font-weight:700;color:#07090E;">C</div>
            <div style="line-height:1.25;">
              <div style="font-size:.88rem;font-weight:600;color:#E6E9F2;">CyberML</div>
              <div style="font-size:.72rem;color:#4A5163;
                          text-transform:uppercase;letter-spacing:.12em;
                          font-family:'JetBrains Mono',monospace;">KMITL · prod</div>
            </div>
          </div>
          <div style="background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.18);
                      border-radius:8px;padding:8px 12px;margin-bottom:10px;">
            <div style="display:flex;align-items:center;gap:8px;">
              <span style="width:7px;height:7px;border-radius:50%;background:#10B981;
                           box-shadow:0 0 6px #10B981;
                           animation:pulseDot 1.6s ease-in-out infinite;"></span>
              <span style="font-size:.72rem;text-transform:uppercase;letter-spacing:.12em;
                           color:#10B981;font-weight:600;">Active session</span>
            </div>
            <div style="font-size:.82rem;color:#A8AFC0;margin-top:6px;">
              {n_models} model{"s" if n_models != 1 else ""} trained
              &nbsp;·&nbsp; {n_classes} classes
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
