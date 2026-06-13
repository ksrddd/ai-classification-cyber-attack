"""sklearn Pipeline assembly -- the leak-proof core.

Every model is wrapped as::

    Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    <model>),
    ])

When this Pipeline is handed to ``GridSearchCV(cv=5)``, the scaler refits
on each training fold -- test data never influences scaler statistics.
That's the structural defence against data leakage.
"""

from __future__ import annotations

import logging
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


_SCALERS = {
    "standard": StandardScaler,
    "minmax":   MinMaxScaler,
    "robust":   RobustScaler,
}


def build_scaler(kind: str = "standard") -> BaseEstimator:
    """Return a fresh, unfitted scaler instance.

    ``kind`` must be one of ``standard``, ``minmax``, ``robust``.
    """
    if kind not in _SCALERS:
        raise ValueError(
            f"Unknown scaler kind {kind!r}. Choices: {sorted(_SCALERS)}."
        )
    return _SCALERS[kind]()


def build_model_pipeline(
    estimator: BaseEstimator,
    scaler_kind: str = "standard",
    *,
    extra_steps: list[tuple[str, Any]] | None = None,
) -> Pipeline:
    """Wrap ``estimator`` in a scaler+clf Pipeline.

    ``extra_steps`` is rarely needed but lets callers insert PCA or
    SelectFromModel between scaler and clf if they want. The CV-safety
    guarantee still holds because every step is inside the Pipeline.
    """
    scaler = build_scaler(scaler_kind)
    # Preserve DataFrame column names through the scaler so downstream
    # estimators (LightGBM, XGBoost) don't warn about missing feature names.
    scaler.set_output(transform="pandas")
    steps: list[tuple[str, Any]] = [("scaler", scaler)]
    if extra_steps:
        steps.extend(extra_steps)
    steps.append(("clf", estimator))
    pipe = Pipeline(steps=steps)
    logger.debug(
        "Built pipeline: %s", " -> ".join(name for name, _ in pipe.steps)
    )
    return pipe


def get_classifier(pipeline: Pipeline) -> BaseEstimator:
    """Convenience -- return the ``clf`` step of a model pipeline."""
    return pipeline.named_steps["clf"]


def is_tree_based(estimator: BaseEstimator) -> bool:
    """Heuristic -- is this an estimator SHAP TreeExplainer can handle?

    Used by the SHAP layer to pick TreeExplainer (RF/XGBoost/LightGBM/
    CatBoost) vs KernelExplainer (MLP, Logistic). Uses ``isinstance`` so
    subclasses (e.g. ``BalancedXGBClassifier``) are recognised too.
    """
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
    )

    tree_types: tuple[type, ...] = (
        RandomForestClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
    )
    try:
        from xgboost import XGBClassifier
        tree_types = (*tree_types, XGBClassifier)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        tree_types = (*tree_types, LGBMClassifier)
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier
        tree_types = (*tree_types, CatBoostClassifier)
    except ImportError:
        pass
    return isinstance(estimator, tree_types)
