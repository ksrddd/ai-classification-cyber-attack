"""Cross-model comparison report.

Builds a single Markdown + CSV + PNG comparing every trained model on
the held-out test set. This is the artifact you'll show the panel.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.constants import METRICS_DIR, REPORTS_DIR
from src.utils.io import ensure_dir

logger = logging.getLogger(__name__)

_BAR_METRICS = ("accuracy", "f1_weighted", "f1_macro", "roc_auc")


def build_comparison_table(per_model_metrics: dict[str, dict]) -> pd.DataFrame:
    """Stack per-model metric dicts into a single DataFrame.

    ``per_model_metrics`` shape::

        {
            "random_forest": {"accuracy": 0.99, "f1_weighted": 0.99, ...},
            "xgboost":       {"accuracy": 0.98, ...},
            ...
        }
    """
    rows: list[dict] = []
    for name, m in per_model_metrics.items():
        row = {"model": name}
        for k, v in m.items():
            if isinstance(v, dict):
                continue  # per_class lives in its own table
            row[k] = v
        rows.append(row)
    df = pd.DataFrame(rows).set_index("model")
    return df.sort_values("f1_weighted", ascending=False) if "f1_weighted" in df.columns else df


def plot_comparison_bars(
    df: pd.DataFrame,
    metrics: Iterable[str] = _BAR_METRICS,
    save_to: Path | None = None,
) -> Path:
    """Grouped bar chart of selected metrics across all models."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from src.utils.io import ensure_dir as _ensure_dir
    from src.visualization.plots import save_fig, set_style

    set_style()
    cols = [m for m in metrics if m in df.columns]
    if not cols:
        raise ValueError(f"None of {list(metrics)} are columns of df ({list(df.columns)}).")
    bars = df[cols]

    fig, ax = plt.subplots(figsize=(max(6, len(df) * 1.2), 5))
    bars.plot(kind="bar", ax=ax)
    ax.set_title("Model comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(loc="lower right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)

    save_to = save_to or (METRICS_DIR / "model_comparison.png")
    _ensure_dir(save_to.parent)
    return save_fig(fig, save_to)


def write_comparison_report(
    per_model_metrics: dict[str, dict],
    *,
    csv_path: Path | None = None,
    md_path: Path | None = None,
    png_path: Path | None = None,
) -> dict[str, Path]:
    """Persist comparison CSV + Markdown + bar chart. Returns a path map."""
    csv_path = csv_path or (METRICS_DIR / "model_comparison.csv")
    md_path  = md_path  or (REPORTS_DIR / "model_comparison.md")
    png_path = png_path or (METRICS_DIR / "model_comparison.png")
    ensure_dir(csv_path.parent)
    ensure_dir(md_path.parent)

    df = build_comparison_table(per_model_metrics)
    df.to_csv(csv_path)

    plot_comparison_bars(df, save_to=png_path)

    lines = ["# Model Comparison", ""]
    lines.append(df.round(4).to_markdown())
    lines.append("")
    lines.append("## Per-class breakdown")
    for model, m in per_model_metrics.items():
        per_class = m.get("per_class") if isinstance(m, dict) else None
        if not per_class:
            continue
        lines.append(f"\n### {model}\n")
        pc_df = pd.DataFrame(per_class).T
        lines.append(pc_df.round(4).to_markdown())
    md_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Comparison written: %s, %s, %s", csv_path, md_path, png_path)
    return {"csv": csv_path, "md": md_path, "png": png_path}


def best_model(per_model_metrics: dict[str, dict], metric: str = "f1_weighted") -> str:
    """Return the model name with the highest score on ``metric``."""
    scores = {
        name: m.get(metric, float("-inf"))
        for name, m in per_model_metrics.items()
        if isinstance(m, dict)
    }
    if not scores:
        raise ValueError("No models in metrics dict.")
    return max(scores, key=lambda k: float(scores[k]) if not np.isnan(float(scores[k])) else float("-inf"))
