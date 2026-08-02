"""Tests for the CSE-CIC-IDS2018 model builders.

Focused on the two behaviours that are easy to break silently: the label
remapping the stacking ensemble depends on, and the global class-weighting
switch that keeps the 7-way comparison fair.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.model_selection import cross_val_predict

from src.ids2018.models import LabelSafeXGBClassifier, build_model, set_class_weighting

N_CLASSES = 15
RARE_CLASS = 13


@pytest.fixture
def gapped_data() -> tuple[np.ndarray, np.ndarray]:
    """A 15-class problem where one class has exactly one member.

    This is SQL Injection's situation in the real 300k sample. Any CV split
    leaves that class out of most folds, so a base learner sees labels with
    a hole in them.
    """
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(N_CLASSES), 40)
    y[y == RARE_CLASS] = 0
    y[0] = RARE_CLASS
    rng.shuffle(y)
    X = rng.normal(size=(len(y), 6)) + y[:, None] * 0.4
    return X, y


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _estimator():
    return LabelSafeXGBClassifier(n_estimators=5, max_depth=3, verbosity=0)


def test_fits_on_non_contiguous_labels(gapped_data) -> None:
    """Plain XGBoost raises on gapped labels; the wrapper must not."""
    X, y = gapped_data
    mask = y != RARE_CLASS
    model = _estimator().fit(X[mask], y[mask])
    assert len(model.classes_) == N_CLASSES - 1
    assert RARE_CLASS not in model.classes_


def test_predict_returns_original_label_codes(gapped_data) -> None:
    """XGBoost predicts dense column indices; they must be mapped back."""
    X, y = gapped_data
    model = _estimator().fit(X, y)
    assert set(np.unique(model.predict(X))).issubset(set(np.unique(y)))


def test_proba_columns_align_with_classes(gapped_data) -> None:
    """scikit-learn pads missing classes using classes_, so they must match."""
    X, y = gapped_data
    mask = y != RARE_CLASS
    model = _estimator().fit(X[mask], y[mask])
    assert model.predict_proba(X).shape[1] == len(model.classes_)


def test_cross_val_predict_yields_full_class_space(gapped_data) -> None:
    """The regression test for the crash that killed the stacking ensemble."""
    X, y = gapped_data
    proba = cross_val_predict(_estimator(), X, y, cv=3, method="predict_proba")
    assert proba.shape == (len(y), N_CLASSES)


@pytest.mark.parametrize(
    ("model_name", "attribute", "expected"),
    [
        ("random_forest", "class_weight", "balanced_subsample"),
        ("lightgbm", "class_weight", "balanced"),
        ("logistic_regression", "class_weight", "balanced"),
    ],
)
def test_class_weighting_switch(model_name: str, attribute: str, expected: str) -> None:
    """'none' must leave every model unweighted; 'balanced' must apply it."""
    try:
        set_class_weighting("none")
        assert getattr(build_model(model_name), attribute) is None

        set_class_weighting("balanced")
        assert getattr(build_model(model_name), attribute) == expected
    finally:
        set_class_weighting("none")


def test_lightgbm_regularisation_is_pinned() -> None:
    """reg_lambda=0 makes LightGBM diverge on this dataset -- never ship it.

    With an unregularised leaf the optimal value -g/h is unbounded, and the
    single-row SQL Injection class drives it to overflow: accuracy peaks at
    0.978 by round 10 and collapses to 0.678 by round 100.
    """
    set_class_weighting("none")
    assert build_model("lightgbm").reg_lambda == 1.0
    assert build_model("xgboost").reg_lambda == 1.0
