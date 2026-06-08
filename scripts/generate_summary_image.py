"""Generate a single PNG summarizing project status.

Output: docs/project_summary.png
Run:    python scripts/generate_summary_image.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------
PROJECT_TITLE = "AI-Based Cyber Attack Classification"
SUBTITLE = "Senior Project - KMITL Faculty of Information Technology"
AUTHORS = "Sirachet Chotthakunanan (66070191) and Sukhum Rudeemaetakul (66070315)"
ADVISOR = "Asst. Prof. Dr. Prapan Pavarangkoon"

PHASES = [
    ("1.  Requirement Analysis",       "done"),
    ("2.  Architecture Design",        "done"),
    ("3.  Project Scaffold",           "done"),
    ("4.  Data Engineering",           "done"),
    ("5.  Feature Selection",          "pending"),
    ("6.  Model Development",          "pending"),
    ("7.  Optimization (GridSearch)",  "pending"),
    ("8.  Evaluation",                 "pending"),
    ("9.  Explainable AI (SHAP)",      "pending"),
    ("10. Streamlit Dashboard",        "pending"),
    ("11. Testing (broader)",          "pending"),
    ("12. MLOps Preparation",          "pending"),
    ("13. Final Documentation",        "pending"),
]

STATS = [
    ("Commits",               "2"),
    ("Tracked files",         "97"),
    ("Tests passing",         "34 / 34"),
    ("Code coverage",         "75%"),
    ("Python version",        "3.13.12"),
    ("Models planned",        "LR + RF + MLP"),
    ("Target attack classes", "4"),
    ("Primary metric",        "F1-weighted"),
]

ARCHITECTURE = [
    ("Raw CSV",          "data/raw/  or  data/sample/"),
    ("Loader",           "src/data/loader.py"),
    ("Cleaning",         "src/features/cleaning.py"),
    ("Filter 4 classes", "src/features/cleaning.filter_target_classes"),
    ("Label Encoder",    "src/features/encoder.py"),
    ("Train/Test Split", "stratified, 80/20, random_state=42"),
    ("Pipeline",         "Scaler + Classifier  (Phase 5+)"),
    ("GridSearchCV",     "5-fold CV  (Phase 7)"),
    ("Trained model",    "models/*.joblib  (Phase 6)"),
    ("SHAP",             "TreeExplainer on RF  (Phase 9)"),
    ("Dashboard",        "streamlit run dashboard/app.py  (Phase 10)"),
]

PROJECT_TREE = [
    "cyber_attack_classification/",
    "  data/{raw,interim,processed,sample}/",
    "  notebooks/   -- 01_EDA.ipynb, 02_Preprocessing.ipynb done",
    "  src/",
    "    config/      -- constants, config.yaml, loader",
    "    data/        -- schema, loader, eda    [Phase 4]",
    "    features/    -- cleaning, encoder      [Phase 4]",
    "    models/      -- base, LR, RF, MLP      [Phase 6]",
    "    evaluation/  -- metrics, CM            [Phase 8]",
    "    explainability/ shap_analyzer          [Phase 9]",
    "    pipelines/   -- eda, preprocess        [Phase 4]",
    "    utils/       -- logging, seeds, io",
    "  dashboard/   -- app.py + 6 pages         [Phase 10]",
    "  tests/       -- pytest, 34 passing",
    "  docs/        -- architecture / pipeline / eval / shap",
    "  scripts/     -- generate_sample.py",
    "  main.py      -- CLI: --stage {eda,preprocess,...}",
]

HOW_TO_RUN = [
    "cd <project>",
    ".\\.venv\\Scripts\\Activate.ps1",
    "pytest                                            -> 34 passed",
    "python scripts/generate_sample.py                 -> data/sample/*.csv",
    "python main.py --stage eda       --raw-dir data\\sample",
    "python main.py --stage preprocess --raw-dir data\\sample",
    "jupyter lab                                        # 01_EDA.ipynb",
    "streamlit run dashboard\\app.py                    # 6-page skeleton",
]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
GREEN = "#2e7d32"
GREEN_BG = "#c8e6c9"
GRAY = "#9e9e9e"
GRAY_BG = "#eceff1"
NAVY = "#1a237e"
ORANGE = "#ef6c00"


def main() -> Path:
    fig = plt.figure(figsize=(18, 13), dpi=140)
    fig.patch.set_facecolor("white")

    gs = fig.add_gridspec(
        nrows=5, ncols=2,
        height_ratios=[0.9, 0.35, 3.2, 3.2, 1.6],
        width_ratios=[1.0, 1.0],
        hspace=0.45, wspace=0.18,
        left=0.03, right=0.97, top=0.97, bottom=0.04,
    )

    _draw_title(fig.add_subplot(gs[0, :]))
    _draw_meta(fig.add_subplot(gs[1, :]))
    _draw_phases(fig.add_subplot(gs[2, 0]))
    _draw_stats(fig.add_subplot(gs[2, 1]))
    _draw_architecture(fig.add_subplot(gs[3, 0]))
    _draw_tree(fig.add_subplot(gs[3, 1]))
    _draw_how_to_run(fig.add_subplot(gs[4, :]))

    out = PROJECT_ROOT / "docs" / "project_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def _ax_off(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _draw_title(ax) -> None:
    _ax_off(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.72, PROJECT_TITLE, ha="center", va="center",
            fontsize=24, fontweight="bold", color=NAVY)
    ax.text(0.5, 0.30, SUBTITLE, ha="center", va="center",
            fontsize=13, color="#37474f")


def _draw_meta(ax) -> None:
    _ax_off(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    line = f"{AUTHORS}   |   Advisor: {ADVISOR}   |   Status: Phase 4 of 13"
    ax.text(0.5, 0.5, line, ha="center", va="center", fontsize=10, color="#455a64")


def _section_header(ax, text):
    ax.text(0.0, 1.01, text, ha="left", va="bottom",
            fontsize=13, fontweight="bold", color=NAVY,
            transform=ax.transAxes)


def _draw_phases(ax) -> None:
    _ax_off(ax)
    _section_header(ax, "Phase Progress  (4 of 13 complete)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(PHASES))

    row_h = 0.78
    for i, (name, status) in enumerate(PHASES):
        y = len(PHASES) - 1 - i
        is_done = (status == "done")
        bg = GREEN_BG if is_done else GRAY_BG
        fg = GREEN if is_done else GRAY
        mark = "[ done ]" if is_done else "[      ]"

        ax.add_patch(FancyBboxPatch(
            (0.005, y + 0.05), 0.99, row_h,
            boxstyle="round,pad=0.005,rounding_size=0.02",
            facecolor=bg, edgecolor=fg, linewidth=0.8,
        ))
        ax.text(0.025, y + 0.05 + row_h / 2, mark,
                ha="left", va="center", fontsize=9.5, color=fg,
                fontweight="bold", family="monospace")
        ax.text(0.13, y + 0.05 + row_h / 2, name,
                ha="left", va="center", fontsize=10.5,
                color="#263238" if is_done else "#546e7a")


def _draw_stats(ax) -> None:
    _ax_off(ax)
    _section_header(ax, "Project Stats")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(STATS))

    row_h = 0.78
    for i, (k, v) in enumerate(STATS):
        y = len(STATS) - 1 - i
        ax.add_patch(FancyBboxPatch(
            (0.005, y + 0.05), 0.99, row_h,
            boxstyle="round,pad=0.005,rounding_size=0.02",
            facecolor="#f5f5f5", edgecolor="#cfd8dc", linewidth=0.6,
        ))
        ax.text(0.025, y + 0.05 + row_h / 2, k,
                ha="left", va="center", fontsize=10.5, color="#37474f")
        ax.text(0.975, y + 0.05 + row_h / 2, v,
                ha="right", va="center", fontsize=10.5, color=NAVY,
                fontweight="bold", family="monospace")


def _draw_architecture(ax) -> None:
    _ax_off(ax)
    _section_header(ax, "Data Flow  (Phase 4 done up to Label Encoder)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(ARCHITECTURE))

    row_h = 0.72
    done_through = 5  # rows 0..5 are implemented in Phase 4
    for i, (step, where) in enumerate(ARCHITECTURE):
        y = len(ARCHITECTURE) - 1 - i
        is_done = i <= done_through
        bg = GREEN_BG if is_done else "#fff3e0"
        fg = GREEN if is_done else ORANGE

        ax.add_patch(FancyBboxPatch(
            (0.005, y + 0.10), 0.99, row_h,
            boxstyle="round,pad=0.005,rounding_size=0.02",
            facecolor=bg, edgecolor=fg, linewidth=0.7,
        ))
        ax.text(0.025, y + 0.10 + row_h / 2, step,
                ha="left", va="center", fontsize=10, color="#263238",
                fontweight="bold")
        ax.text(0.42, y + 0.10 + row_h / 2, where,
                ha="left", va="center", fontsize=9, color="#455a64",
                family="monospace")


def _draw_tree(ax) -> None:
    _ax_off(ax)
    _section_header(ax, "Project Tree")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    text = "\n".join(PROJECT_TREE)
    ax.text(0.0, 0.99, text, ha="left", va="top",
            fontsize=9.2, family="monospace", color="#263238")


def _draw_how_to_run(ax) -> None:
    _ax_off(ax)
    _section_header(ax, "How to Run")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    text = "\n".join(HOW_TO_RUN)
    ax.add_patch(FancyBboxPatch(
        (0.0, 0.02), 1.0, 0.92,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor="#263238", edgecolor="#263238",
    ))
    ax.text(0.012, 0.92, text, ha="left", va="top",
            fontsize=10, family="monospace", color="#eceff1")


if __name__ == "__main__":  # pragma: no cover
    main()
