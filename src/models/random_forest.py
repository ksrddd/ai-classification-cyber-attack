"""Random Forest wrapper.

Tree ensemble baseline. Supports ``class_weight='balanced'`` natively so
no SMOTE is needed for this model. SHAP TreeExplainer works directly.
"""

from __future__ import annotations

import logging

from sklearn.ensemble import RandomForestClassifier

from src.config.constants import RANDOM_STATE
from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    name = "random_forest"

    def _build_estimator(self) -> RandomForestClassifier:
        params = self.baseline_params
        params.setdefault("random_state", RANDOM_STATE)
        params.setdefault("n_jobs", -1)
        return RandomForestClassifier(**params)
