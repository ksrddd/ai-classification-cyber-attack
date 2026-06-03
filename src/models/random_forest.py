"""Random Forest wrapper.

The "main" model of the project — also the only one SHAP TreeExplainer
operates on (ADR-007).
"""

from __future__ import annotations

import logging

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    name = "random_forest"

    def _build_estimator(self):
        # TODO(phase 6): return sklearn.ensemble.RandomForestClassifier(**self.config["baseline"], random_state=RANDOM_STATE)
        raise NotImplementedError("Phase 6 implementation pending.")
