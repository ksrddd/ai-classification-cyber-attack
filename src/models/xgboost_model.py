"""XGBoost wrapper.

Histogram-based gradient boosting. ``tree_method="hist"`` is the fastest
CPU option for the dataset sizes we deal with.
"""

from __future__ import annotations

import logging
from typing import Any

from xgboost import XGBClassifier

from src.config.constants import RANDOM_STATE
from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    name = "xgboost"

    def _build_estimator(self) -> XGBClassifier:
        params: dict[str, Any] = self.baseline_params
        params.setdefault("random_state", RANDOM_STATE)
        params.setdefault("n_jobs", -1)
        params.setdefault("tree_method", "hist")
        # xgboost 2.x emits a warning if these aren't set explicitly.
        params.setdefault("eval_metric", "mlogloss")
        return XGBClassifier(**params)
