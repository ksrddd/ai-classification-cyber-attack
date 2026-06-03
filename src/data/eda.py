"""Exploratory Data Analysis.

Every function returns either (a) a serializable summary dict or (b) a
saved PNG path. Output directory is ``results/figures/`` by default so
the dashboard and reports can pull from a known location without
re-running EDA on every page load.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config.constants import FIGURES_DIR, LABEL_COLUMN
from src.utils.io import ensure_dir
from src.visualization.plots import save_fig, set_style

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def describe_dataset(df: pd.DataFrame, label_col: str = LABEL_COLUMN) -> dict:
    """Return a JSON-serializable summary of the DataFrame.

    Captures: shape, dtype counts, label distribution, missing-value
    audit, inf-value audit. Used by both the EDA notebook and the
    dashboard's Overview page.
    """
    numeric = df.select_dtypes(include=np.number)
    inf_count = int(np.isinf(numeric.to_numpy()).sum())

    summary = {
        "shape": tuple(df.shape),
        "n_features": df.shape[1] - (1 if label_col in df.columns else 0),
        "dtype_counts": df.dtypes.astype(str).value_counts().to_dict(),
        "missing_total": int(df.isna().sum().sum()),
        "missing_per_column_top5":
            df.isna().sum().sort_values(ascending=False).head(5).to_dict(),
        "infinite_total": inf_count,
        "infinite_columns":
            numeric.columns[np.isinf(numeric).any(axis=0)].tolist(),
        "label_distribution":
            df[label_col].value_counts().to_dict() if label_col in df.columns else {},
        "duplicate_row_count": int(df.duplicated().sum()),
    }
    logger.info("Dataset summary: %s", summary["shape"])
    return summary


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_class_distribution(
    df: pd.DataFrame,
    label_col: str = LABEL_COLUMN,
    save_to: Path | None = None,
) -> Path:
    """Bar chart of label counts (log-scale to handle imbalance)."""
    set_style()
    counts = df[label_col].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(
        x=counts.index, y=counts.values, ax=ax,
        hue=counts.index, palette="viridis", legend=False,
    )
    ax.set_yscale("log")
    ax.set_title("Class distribution (log scale)")
    ax.set_xlabel("Attack class")
    ax.set_ylabel("Record count (log)")
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    for i, (_, v) in enumerate(counts.items()):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    path = save_to or (FIGURES_DIR / "class_distribution.png")
    return save_fig(fig, path)


def plot_missing_value_audit(
    df: pd.DataFrame,
    top_k: int = 20,
    save_to: Path | None = None,
) -> Path:
    """Bar chart of the top-K columns by missing-value count."""
    set_style()
    miss = df.isna().sum().sort_values(ascending=False).head(top_k)
    if (miss == 0).all():
        logger.info("No missing values detected; skipping missing-value plot.")
        # Still produce a chart so the dashboard has something to show.
    fig, ax = plt.subplots()
    sns.barplot(
        x=miss.values, y=miss.index, ax=ax,
        hue=miss.index, palette="rocket", legend=False,
    )
    ax.set_title(f"Top-{top_k} columns by missing-value count")
    ax.set_xlabel("Missing-value count")
    path = save_to or (FIGURES_DIR / "missing_value_audit.png")
    return save_fig(fig, path)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    top_k: int | None = 25,
    save_to: Path | None = None,
) -> Path:
    """Correlation heatmap of the top-K most-varying numeric features.

    Full 78×78 heatmap is unreadable; we pick the ``top_k`` columns by
    variance so the panel sees the interesting structure.
    """
    set_style()
    numeric = df.select_dtypes(include=np.number)
    if top_k is not None and numeric.shape[1] > top_k:
        top_cols = numeric.var().sort_values(ascending=False).head(top_k).index
        numeric = numeric[top_cols]
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax,
                xticklabels=True, yticklabels=True,
                cbar_kws={"shrink": 0.7})
    ax.set_title(f"Correlation heatmap (top-{top_k or 'all'} by variance)")
    path = save_to or (FIGURES_DIR / "correlation_heatmap.png")
    return save_fig(fig, path)


def plot_feature_distributions(
    df: pd.DataFrame,
    features: list[str] | None = None,
    label_col: str = LABEL_COLUMN,
    save_to: Path | None = None,
) -> Path:
    """KDE/violin grid for selected features split by class.

    Default: pick 6 features with the highest between-class variance
    ratio (a rough discriminative-power proxy).
    """
    set_style()
    if features is None:
        features = _pick_discriminative_features(df, label_col, k=6)
    n = len(features)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5))
    axes = np.array(axes).reshape(-1)
    for ax, feat in zip(axes, features, strict=False):
        sns.violinplot(data=df, x=label_col, y=feat, ax=ax, inner="quartile")
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel("")
        for label in ax.get_xticklabels():
            label.set_rotation(25)
            label.set_ha("right")
    for ax in axes[len(features):]:
        ax.axis("off")
    fig.suptitle("Feature distributions by attack class", fontsize=14)
    path = save_to or (FIGURES_DIR / "feature_distributions.png")
    return save_fig(fig, path)


def _pick_discriminative_features(
    df: pd.DataFrame,
    label_col: str,
    k: int = 6,
) -> list[str]:
    """Pick top-K numeric features by class-mean dispersion."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) == 0:
        return []
    class_means = df.groupby(label_col)[numeric_cols].mean()
    # Coefficient of variation of class means: higher = more discriminative.
    overall_mean = class_means.mean().replace(0, np.nan)
    score = (class_means.std() / overall_mean.abs()).fillna(0)
    return score.sort_values(ascending=False).head(k).index.tolist()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_eda(
    df: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
    label_col: str = LABEL_COLUMN,
) -> dict:
    """Generate the full EDA bundle: summary dict + 4 plots.

    Returns the summary dict, with plot paths added under ``figures``.
    """
    ensure_dir(output_dir)
    summary = describe_dataset(df, label_col=label_col)
    summary["figures"] = {
        "class_distribution":  str(plot_class_distribution(df, label_col, output_dir / "class_distribution.png")),
        "missing_value_audit": str(plot_missing_value_audit(df, save_to=output_dir / "missing_value_audit.png")),
        "correlation_heatmap": str(plot_correlation_heatmap(df, save_to=output_dir / "correlation_heatmap.png")),
        "feature_distributions": str(plot_feature_distributions(df, save_to=output_dir / "feature_distributions.png")),
    }
    return summary
