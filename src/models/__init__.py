"""Model wrappers + GridSearchCV tuner.

Each model is a thin class around the underlying sklearn-compatible
estimator that:
- Reads hyperparameters from ``config.yaml::models.<name>``.
- Builds itself wrapped in the standard Pipeline (scaler + clf).
- Exposes ``fit``, ``predict``, ``predict_proba``, ``save``, ``load``.
- Uses ``RANDOM_STATE`` for any stochastic component.

Public entry point: ``src.models.registry.build_model(name, cfg)``.
"""

from src.models.base import BaseModel, default_model_path
from src.models.registry import (
    ALIASES,
    MODEL_CLASSES,
    available_models,
    build_model,
    resolve_name,
)

__all__ = [
    "ALIASES",
    "BaseModel",
    "MODEL_CLASSES",
    "available_models",
    "build_model",
    "default_model_path",
    "resolve_name",
]
