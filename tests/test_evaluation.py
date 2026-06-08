"""Tests for src.evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import comparison, confusion_matrix, metrics


def test_compute_metrics_returns_expected_keys() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=100)
    y_pred = rng.integers(0, 3, size=100)
    proba = rng.random(size=(100, 3))
    proba = proba / proba.sum(axis=1, keepdims=True)
    m = metrics.compute_metrics(y_true, y_pred, y_proba=proba)
    for key in (
        "accuracy", "precision_weighted", "recall_weighted", "f1_weighted",
        "precision_macro", "recall_macro", "f1_macro",
        "matthews_corrcoef", "roc_auc", "per_class",
    ):
        assert key in m


def test_classification_report_df_has_rows_for_classes_and_summary() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2])
    df = metrics.classification_report_df(y_true, y_pred, class_names=["a", "b", "c"])
    assert "a" in df.index and "b" in df.index and "c" in df.index
    assert "macro avg" in df.index


def test_compute_confusion_matches_shape() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 0])
    cm = confusion_matrix.compute_confusion(y_true, y_pred, labels=[0, 1, 2])
    assert cm.shape == (3, 3)


def test_build_comparison_table_sorts_by_f1_weighted() -> None:
    per_model = {
        "a": {"accuracy": 0.8, "f1_weighted": 0.7},
        "b": {"accuracy": 0.9, "f1_weighted": 0.9},
    }
    df = comparison.build_comparison_table(per_model)
    assert df.index.tolist()[0] == "b"


def test_best_model_picks_top_f1() -> None:
    per_model = {
        "a": {"accuracy": 0.8, "f1_weighted": 0.7},
        "b": {"accuracy": 0.9, "f1_weighted": 0.9},
    }
    assert comparison.best_model(per_model) == "b"
