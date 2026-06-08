"""MLP (multi-layer perceptron) wrapper.

Uses scikit-learn's ``MLPClassifier`` -- no PyTorch/TensorFlow needed.

Class imbalance: MLPClassifier does NOT support ``class_weight``. Either
rely on the upstream subsample being class-balanced enough, or set
``imbalance_strategy: smote`` in the config (handled at preprocess time).
"""

from __future__ import annotations

import logging

from sklearn.neural_network import MLPClassifier

from src.config.constants import RANDOM_STATE
from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class MLPModel(BaseModel):
    name = "mlp"

    def _build_estimator(self) -> MLPClassifier:
        params = self.baseline_params
        params.setdefault("random_state", RANDOM_STATE)
        # MLPClassifier wants hidden_layer_sizes as a tuple, not list.
        hls = params.get("hidden_layer_sizes")
        if isinstance(hls, list):
            params["hidden_layer_sizes"] = tuple(hls)
        return MLPClassifier(**params)
