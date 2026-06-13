"""MLP (multi-layer perceptron) wrapper.

Uses scikit-learn's ``MLPClassifier`` inside an imbalanced-learn Pipeline
so SMOTE oversampling is applied only during training (never at predict time).
This compensates for MLPClassifier not supporting ``class_weight``.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.config.constants import RANDOM_STATE
from src.models.base import BaseModel

logger = logging.getLogger(__name__)


def _smote_strategy(y: np.ndarray) -> dict[int, int]:
    """Oversample each minority class to at most 30% of the majority count.

    Passing a float to sampling_strategy only works for binary targets.
    For multiclass we must return a dict {class_label: desired_n_samples}.
    Capping at 30% of majority prevents the dataset from ballooning to
    millions of rows and exhausting RAM on the 300K-row subsample.
    """
    counts = Counter(y)
    majority_count = max(counts.values())
    target = int(majority_count * 0.3)
    return {
        cls: target
        for cls, cnt in counts.items()
        if cnt < majority_count and cnt < target
    }


class MLPModel(BaseModel):
    name = "mlp"

    def _build_estimator(self) -> MLPClassifier:
        params = self.baseline_params
        params.setdefault("random_state", RANDOM_STATE)
        hls = params.get("hidden_layer_sizes")
        if isinstance(hls, list):
            params["hidden_layer_sizes"] = tuple(hls)
        return MLPClassifier(**params)

    def build(self) -> ImbPipeline:
        """Override: inject SMOTE between scaler and MLP.

        imblearn.Pipeline applies the SMOTE step only on fit(), not predict(),
        so the test set never sees synthetic samples.
        """
        estimator = self._build_estimator()
        self.pipeline = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote",  SMOTE(random_state=RANDOM_STATE, k_neighbors=5,
                             sampling_strategy=_smote_strategy)),
            ("clf",    estimator),
        ])
        logger.debug("Built MLP pipeline with SMOTE: scaler -> smote -> clf")
        return self.pipeline
