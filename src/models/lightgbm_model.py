"""LightGBM wrapper.

Microsoft's GBDT -- typically the fastest tree booster on tabular data
of this size. Supports ``class_weight='balanced'`` so handles CICIDS's
heavy class imbalance natively.
"""

from __future__ import annotations

import logging

from lightgbm import LGBMClassifier

from src.config.constants import RANDOM_STATE
from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class LightGBMModel(BaseModel):
    name = "lightgbm"

    def _build_estimator(self) -> LGBMClassifier:
        params = self.baseline_params
        params.setdefault("random_state", RANDOM_STATE)
        params.setdefault("n_jobs", -1)
        params.setdefault("verbosity", -1)
        return LGBMClassifier(**params)
