"""Confusion matrix computation + plotting."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from src.config.constants import FIGURES_DIR

logger = logging.getLogger(__name__)


def compute_confusion(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    labels: Iterable[int] | None = None,
    normalize: str | None = None,
) -> np.ndarray:
    """Return the confusion matrix as a numpy array.

    ``normalize`` is forwarded to sklearn: ``None`` | ``"true"`` | ``"pred"`` | ``"all"``.
    """
    lbls = list(labels) if labels is not None else None
    return sk_confusion_matrix(y_true, y_pred, labels=lbls, normalize=normalize)


def confusion_to_df(
    cm: np.ndarray,
    class_names: Iterable[str],
) -> pd.DataFrame:
    """Wrap a confusion matrix array as a labelled DataFrame."""
    names = list(class_names)
    if cm.shape != (len(names), len(names)):
        raise ValueError(
            f"Confusion matrix shape {cm.shape} doesn't match {len(names)} classes."
        )
    return pd.DataFrame(cm, index=names, columns=names)


def plot_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    labels: Iterable[int] | None = None,
    class_names: Iterable[str] | None = None,
    normalize: str | None = "true",
    title: str = "Confusion matrix",
    save_to: Path | None = None,
) -> Path:
    """Save a confusion-matrix heatmap to disk and return the path."""
    import matplotlib.pyplot as plt  # noqa: PLC0415 -- lazy for headless test env
    import seaborn as sns  # noqa: PLC0415

    from src.utils.io import ensure_dir
    from src.visualization.plots import save_fig, set_style

    set_style()

    cm = compute_confusion(y_true, y_pred, labels=labels, normalize=normalize)
    names = list(class_names) if class_names is not None else (
        [str(label) for label in (labels or sorted(np.unique(y_true).tolist()))]
    )
    df = confusion_to_df(cm, names)

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.8), max(5, len(names) * 0.6)))
    fmt = ".2f" if normalize else "d"
    sns.heatmap(
        df, annot=True, fmt=fmt, cmap="Blues", cbar=True, ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")

    save_to = save_to or (FIGURES_DIR / "confusion_matrix.png")
    ensure_dir(save_to.parent)
    return save_fig(fig, save_to)
