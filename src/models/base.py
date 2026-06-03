"""Abstract base class for all model wrappers.

Defines the contract every concrete model (LR, RF, MLP) must satisfy.
Following the Template Method pattern — subclasses override
``_build_estimator()``; everything else is inherited.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Common interface for all classifiers in this project."""

    name: str = "base"  # overridden by subclasses

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.pipeline: Pipeline | None = None

    @abstractmethod
    def _build_estimator(self) -> Any:
        """Construct the underlying sklearn classifier (no Pipeline wrapping)."""

    # TODO(phase 6): concrete methods below
    #   def build(self) -> Pipeline                        # wrap _build_estimator in Pipeline
    #   def fit(self, X, y) -> "BaseModel"
    #   def predict(self, X) -> np.ndarray
    #   def predict_proba(self, X) -> np.ndarray
    #   def save(self, path: Path) -> Path
    #   @classmethod
    #   def load(cls, path: Path) -> "BaseModel"
