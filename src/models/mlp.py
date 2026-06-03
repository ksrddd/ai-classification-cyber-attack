"""MLP (multi-layer perceptron) wrapper.

Uses scikit-learn's ``MLPClassifier`` — no PyTorch/TensorFlow needed
(ADR scope: roadmap explicitly excludes deep-learning frameworks).

Class imbalance: MLPClassifier does NOT support ``class_weight``. For
this model the imbalance strategy is SMOTE (imbalanced-learn), wired in
the imblearn Pipeline rather than the sklearn one (Phase 6).
"""

from __future__ import annotations

import logging

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class MLPModel(BaseModel):
    name = "mlp"

    def _build_estimator(self):
        # TODO(phase 6): return sklearn.neural_network.MLPClassifier(**self.config["baseline"], random_state=RANDOM_STATE)
        raise NotImplementedError("Phase 6 implementation pending.")
