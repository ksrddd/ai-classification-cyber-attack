"""Shared matplotlib/seaborn helpers.

Project-wide visual conventions live here so every chart in the EDA,
evaluation, and SHAP outputs looks consistent. Use ``set_style`` at the
start of any module that plots; use ``save_fig`` to write PNGs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.io import ensure_dir

logger = logging.getLogger(__name__)

DEFAULT_DPI = 120
DEFAULT_FIGSIZE = (10, 6)


def set_style() -> None:
    """Apply project-wide matplotlib + seaborn defaults. Idempotent."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.figsize": DEFAULT_FIGSIZE,
        "figure.dpi":     DEFAULT_DPI,
        "savefig.dpi":    DEFAULT_DPI,
        "savefig.bbox":   "tight",
        "axes.titleweight": "bold",
    })


def save_fig(fig: plt.Figure, path: Path) -> Path:
    """Save a figure with project defaults then close it."""
    ensure_dir(path.parent)
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved figure -> %s", path)
    return path
