"""Logistic Regression wrapper.

Hyperparameters read from ``config.yaml::models.logistic_regression``.
Uses ``class_weight='balanced'`` by default (R-02 mitigation).
"""

from __future__ import annotations

import logging

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class LogisticRegressionModel(BaseModel):
    name = "logistic_regression"

    def _build_estimator(self):
        # TODO(phase 6): return sklearn.linear_model.LogisticRegression(**self.config["baseline"], random_state=RANDOM_STATE)
        raise NotImplementedError("Phase 6 implementation pending.")
