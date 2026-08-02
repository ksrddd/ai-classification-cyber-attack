"""Metrics, confusion matrices and the 7-model comparison table.

Macro and weighted averages are both reported, and the gap between them is
the number to read first on this dataset: weighted average is dominated by
the ~83% Benign class and stays near 1.00 even for a model that misses
every SQL Injection flow, while macro average treats all 15 classes equally
and exposes exactly that failure.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Per-model evaluation
# ----------------------------------------------------------------------
def evaluate_model(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_seconds: float = 0.0,
    predict_seconds: float = 0.0,
) -> dict:
    """Compute the full metric set for one model.

    ``zero_division=0`` keeps the run alive when a model predicts a rare
    class zero times -- precision would otherwise be 0/0.
    """
    # Every score is cast to a builtin float: scikit-learn returns numpy
    # scalars, which json.dumps refuses to serialise.
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "train_seconds": float(train_seconds),
        "predict_seconds": float(predict_seconds),
    }

    logger.info(
        "  %-20s acc=%.4f  f1_macro=%.4f  f1_weighted=%.4f",
        name,
        metrics["accuracy"],
        metrics["f1_macro"],
        metrics["f1_weighted"],
    )
    return metrics


def per_class_report(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> pd.DataFrame:
    """Precision / recall / F1 / support for each of the 15 classes."""
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose().rename_axis("class").reset_index()


def confusion_frame(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> pd.DataFrame:
    """Confusion matrix as a labelled DataFrame (rows = true, cols = predicted)."""
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    return pd.DataFrame(cm, index=class_names, columns=class_names)


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------
def save_model_artifacts(
    name: str,
    metrics: dict,
    cm: pd.DataFrame,
    report: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write per-model metrics, confusion matrix and class report to disk."""
    model_dir = output_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)

    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cm.to_csv(model_dir / "confusion_matrix.csv", encoding="utf-8")
    report.to_csv(model_dir / "per_class_report.csv", index=False, encoding="utf-8")
    _plot_confusion_matrix(cm, name, model_dir / "confusion_matrix.png")


def _plot_confusion_matrix(cm: pd.DataFrame, name: str, path: Path) -> None:
    """Render a row-normalised heatmap.

    Row normalisation (i.e. recall per class) is the only readable option
    here: raw counts make every row except Benign invisible.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless-safe; no display needed
        import matplotlib.pyplot as plt
    except ImportError:  # plotting is a convenience, not a requirement
        logger.warning("matplotlib unavailable -- skipping confusion matrix plot for %s", name)
        return

    row_sums = cm.to_numpy().sum(axis=1, keepdims=True)
    normalised = np.divide(
        cm.to_numpy(), row_sums, out=np.zeros_like(cm.to_numpy(), dtype=float), where=row_sums != 0
    )

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cm.columns)), cm.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cm.index)), cm.index, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name} — row-normalised confusion matrix")

    for i in range(normalised.shape[0]):
        for j in range(normalised.shape[1]):
            value = normalised[i, j]
            if value >= 0.005:
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if value > 0.5 else "black",
                )

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Cross-model comparison
# ----------------------------------------------------------------------
COMPARISON_COLUMNS = [
    "model",
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "train_seconds",
    "predict_seconds",
]


def build_comparison_table(all_metrics: list[dict], output_dir: Path | None = None) -> pd.DataFrame:
    """Assemble the summary table for every model with results on disk.

    Metrics from earlier runs are read back from each model's
    ``metrics.json``, so re-running a single model (``--models stacking``)
    updates that one row instead of replacing the whole table with it.
    Results from the current run always win over the stored copy.

    Sorted by accuracy -- the headline number -- with macro F1 alongside it,
    since on an 83%-Benign dataset the two rank models differently.
    """
    merged: dict[str, dict] = {}

    if output_dir is not None:
        for path in sorted(output_dir.glob("*/metrics.json")):
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("Ignoring unreadable %s", path)
                continue
            if all(column in stored for column in COMPARISON_COLUMNS):
                merged[stored["model"]] = stored

    for metrics in all_metrics:
        merged[metrics["model"]] = metrics

    table = pd.DataFrame(list(merged.values()))[COMPARISON_COLUMNS]
    return table.sort_values("accuracy", ascending=False).reset_index(drop=True)


def save_comparison_table(table: pd.DataFrame, output_dir: Path) -> None:
    """Persist the comparison as CSV plus a Markdown copy for the report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "model_comparison.csv", index=False, encoding="utf-8")

    rounded = table.copy()
    score_cols = [c for c in table.columns if c not in {"model", "train_seconds", "predict_seconds"}]
    rounded[score_cols] = rounded[score_cols].round(4)
    rounded[["train_seconds", "predict_seconds"]] = rounded[
        ["train_seconds", "predict_seconds"]
    ].round(1)

    md = ["# CSE-CIC-IDS2018 — Model Comparison", "", rounded.to_markdown(index=False), ""]
    (output_dir / "model_comparison.md").write_text("\n".join(md), encoding="utf-8")

    logger.info("\n%s", rounded.to_string(index=False))
    logger.info("Comparison table written to %s", output_dir / "model_comparison.csv")
