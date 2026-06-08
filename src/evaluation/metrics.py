"""Per-model metric computation.

Computes the full metric set:
- accuracy
- precision / recall / F1 (weighted, macro, per-class)
- ROC-AUC (binary or OvR multi-class, falls back to NaN if predict_proba
  is unavailable for the estimator)
- Matthews correlation coefficient

``compute_metrics`` returns a flat dict that becomes a row in the
comparison CSV. ``classification_report_df`` returns the per-class
breakdown as a DataFrame for the dashboard.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_proba: np.ndarray | None = None,
    labels: Iterable[int] | None = None,
    class_names: Iterable[str] | None = None,
) -> dict[str, float | dict[str, float]]:
    """Return a metric dict suitable for both the comparison CSV and JSON.

    Parameters
    ----------
    y_true, y_pred
        Encoded label arrays (integers).
    y_proba
        2-D array of class probabilities (n_samples, n_classes). Optional.
    labels
        Encoded label values to include in per-class metrics. If None,
        sklearn picks them from ``y_true``.
    class_names
        Human-readable names aligned with ``labels`` for per-class output.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: dict[str, float | dict[str, float]] = {
        "accuracy":             float(accuracy_score(y_true, y_pred)),
        "precision_weighted":   float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted":      float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted":          float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro":      float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro":         float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro":             float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "matthews_corrcoef":    float(matthews_corrcoef(y_true, y_pred)),
    }

    metrics["roc_auc"] = _safe_roc_auc(y_true, y_proba)

    metrics["per_class"] = _per_class_metrics(y_true, y_pred, labels, class_names)
    return metrics


def classification_report_df(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    labels: Iterable[int] | None = None,
    class_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Sklearn classification_report rendered as a DataFrame."""
    report = classification_report(
        y_true,
        y_pred,
        labels=list(labels) if labels is not None else None,
        target_names=list(class_names) if class_names is not None else None,
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(report).T
    # Reorder so per-class rows come before macro/weighted summary rows.
    summary = ["accuracy", "macro avg", "weighted avg"]
    body = [r for r in df.index if r not in summary]
    df = df.loc[body + [r for r in summary if r in df.index]]
    return df


def metrics_row(
    metrics: dict[str, float | dict[str, float]],
    *,
    model_name: str,
    variant: str = "tuned",
) -> dict[str, float | str]:
    """Flatten a metrics dict to a single row for the comparison CSV."""
    row: dict[str, float | str] = {"model": model_name, "variant": variant}
    for k, v in metrics.items():
        if isinstance(v, dict):
            continue  # per_class handled separately
        row[k] = v  # type: ignore[assignment]
    return row


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _safe_roc_auc(y_true: np.ndarray, y_proba: np.ndarray | None) -> float:
    """Return ROC-AUC or NaN if it can't be computed (no proba, 1 class, etc.)."""
    if y_proba is None:
        return float("nan")
    try:
        n_classes = y_proba.shape[1] if y_proba.ndim == 2 else 2
        if n_classes == 2:
            # Binary classification -- use the positive class probability.
            pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            return float(roc_auc_score(y_true, pos))
        return float(
            roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="weighted",
            )
        )
    except (ValueError, IndexError) as exc:
        logger.warning("ROC-AUC could not be computed: %s", exc)
        return float("nan")


def _per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Iterable[int] | None,
    class_names: Iterable[str] | None,
) -> dict[str, dict[str, float]]:
    lbls = list(labels) if labels is not None else sorted(np.unique(y_true).tolist())
    names = list(class_names) if class_names is not None else [str(l) for l in lbls]

    p = precision_score(y_true, y_pred, average=None, labels=lbls, zero_division=0)
    r = recall_score(y_true, y_pred, average=None, labels=lbls, zero_division=0)
    f = f1_score(y_true, y_pred, average=None, labels=lbls, zero_division=0)

    out: dict[str, dict[str, float]] = {}
    for name, pi, ri, fi in zip(names, p, r, f, strict=False):
        out[str(name)] = {
            "precision": float(pi),
            "recall":    float(ri),
            "f1":        float(fi),
        }
    return out
