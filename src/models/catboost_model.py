"""CatBoost wrapper.

Yandex's GBDT -- strong out-of-the-box defaults. Handles imbalance via
``auto_class_weights="Balanced"``. Verbose output is suppressed to keep
CV logs readable.
"""

from __future__ import annotations

import logging

from catboost import CatBoostClassifier

from src.config.constants import RANDOM_STATE
from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class CatBoostModel(BaseModel):
    name = "catboost"

    def _build_estimator(self) -> CatBoostClassifier:
        params = self.baseline_params
        params.setdefault("random_seed", RANDOM_STATE)
        params.setdefault("verbose", 0)
        params.setdefault("allow_writing_files", False)
        return CatBoostClassifier(**params)
