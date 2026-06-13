"""XGBoost wrapper.

Histogram-based gradient boosting. ``tree_method="hist"`` is the fastest
CPU option for the dataset sizes we deal with.

Uses ``BalancedXGBClassifier`` which auto-applies balanced sample weights
during ``fit()`` -- XGBoost lacks a native ``class_weight`` parameter, so
this is how we handle CICIDS2017's heavy class imbalance. Each CV fold
inside GridSearchCV gets per-fold sample weights computed from that
fold's ``y``.
"""

from __future__ import annotations

import logging
from typing import Any

from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config.constants import RANDOM_STATE
from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class BalancedXGBClassifier(XGBClassifier):
    """XGBClassifier that auto-applies ``class_weight='balanced'`` semantics.

    XGBoost lacks ``class_weight`` -- the canonical fix is to pass
    ``sample_weight`` to ``fit()``. This subclass computes it from ``y``
    so the wrapper Just Works inside sklearn Pipelines and GridSearchCV
    (each CV fold gets its own per-fold sample weights).
    """

    def fit(self, X, y, sample_weight=None, **kwargs):
        if sample_weight is None:
            sample_weight = compute_sample_weight("balanced", y)
        return super().fit(X, y, sample_weight=sample_weight, **kwargs)


class XGBoostModel(BaseModel):
    name = "xgboost"

    def _build_estimator(self) -> BalancedXGBClassifier:
        params: dict[str, Any] = self.baseline_params
        params.setdefault("random_state", RANDOM_STATE)
        params.setdefault("n_jobs", -1)
        params.setdefault("tree_method", "hist")
        # xgboost 2.x emits a warning if these aren't set explicitly.
        params.setdefault("eval_metric", "mlogloss")
        return BalancedXGBClassifier(**params)
