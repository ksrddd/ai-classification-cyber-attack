"""Logistic Regression wrapper -- optional linear baseline.

Not part of the primary five (RF / XGBoost / LightGBM / CatBoost / MLP)
but kept available as a sanity-check baseline. Add ``logistic_regression``
to ``config.yaml::models`` to use.
"""

from __future__ import annotations

import logging

from sklearn.linear_model import LogisticRegression

from src.config.constants import RANDOM_STATE
from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class LogisticRegressionModel(BaseModel):
    name = "logistic_regression"

    def _build_estimator(self) -> LogisticRegression:
        params = self.baseline_params
        params.setdefault("random_state", RANDOM_STATE)
        params.setdefault("n_jobs", -1)
        params.setdefault("max_iter", 1000)
        return LogisticRegression(**params)
