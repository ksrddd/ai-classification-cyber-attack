"""Abstract base class for all model wrappers.

Defines the contract every concrete model (RF, XGBoost, LightGBM,
CatBoost, MLP) must satisfy. Subclasses implement ``_build_estimator()``
to return the bare sklearn-compatible classifier; ``build()`` wraps that
in the project's scaler+clf Pipeline so leakage protection comes for free.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from src.config.constants import MODELS_DIR
from src.features.pipeline import build_model_pipeline
from src.utils.io import load_joblib, save_joblib

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Common interface for all classifiers in this project."""

    name: str = "base"  # overridden by subclasses

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        scaler_kind: str = "standard",
    ) -> None:
        self.config = config or {"baseline": {}, "grid": {}}
        self.scaler_kind = scaler_kind
        self.pipeline: Pipeline | None = None

    # ------------------------------------------------------------------
    # Subclasses implement this -- the rest is shared.
    # ------------------------------------------------------------------
    @abstractmethod
    def _build_estimator(self) -> BaseEstimator:
        """Return a fresh classifier instance (no Pipeline wrapping)."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self) -> Pipeline:
        """Construct the scaler+clf Pipeline. Stores it on ``self``."""
        estimator = self._build_estimator()
        self.pipeline = build_model_pipeline(estimator, scaler_kind=self.scaler_kind)
        return self.pipeline

    def fit(self, X, y) -> "BaseModel":
        if self.pipeline is None:
            self.build()
        assert self.pipeline is not None
        self.pipeline.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        self._require_fitted()
        assert self.pipeline is not None
        return self.pipeline.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        self._require_fitted()
        assert self.pipeline is not None
        clf = self.pipeline.named_steps["clf"]
        if not hasattr(clf, "predict_proba"):
            raise AttributeError(
                f"{type(clf).__name__} does not support predict_proba; "
                "use decision_function or predict instead."
            )
        return self.pipeline.predict_proba(X)

    def save(self, path: Path | None = None) -> Path:
        """Persist the fitted pipeline. Default path: ``models/<name>.joblib``."""
        self._require_fitted()
        path = path or default_model_path(self.name)
        return save_joblib(self.pipeline, path)

    @classmethod
    def load(cls, path: Path) -> Pipeline:
        """Load a previously saved Pipeline. Returns the raw Pipeline.

        Callers needing the wrapper class (e.g. dashboard) typically only
        need the Pipeline, so we return that directly rather than re-wrapping.
        """
        pipe = load_joblib(path)
        if not isinstance(pipe, Pipeline):
            raise TypeError(f"Expected sklearn Pipeline at {path}, got {type(pipe)}")
        return pipe

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def baseline_params(self) -> dict[str, Any]:
        return dict(self.config.get("baseline", {}) or {})

    @property
    def grid(self) -> dict[str, list[Any]]:
        return dict(self.config.get("grid", {}) or {})

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", True))

    def _require_fitted(self) -> None:
        if self.pipeline is None:
            raise RuntimeError(
                f"{type(self).__name__} is not built/fit yet. "
                "Call .build() or .fit(X, y) first."
            )


def default_model_path(name: str) -> Path:
    """Standard on-disk location for a saved model pipeline."""
    return MODELS_DIR / f"{name}.joblib"
