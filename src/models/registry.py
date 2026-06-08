"""Model registry -- one place that lists every model class.

The training pipeline asks the registry for ``available_models(config)``
and gets back a dict of ``name -> BaseModel`` instance built from the
per-model config block.
"""

from __future__ import annotations

import logging
from typing import Any

from src.models.base import BaseModel
from src.models.catboost_model import CatBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.mlp import MLPModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)

# Canonical name -> class. Order here is the order the comparison report
# uses, so put the primary models first.
MODEL_CLASSES: dict[str, type[BaseModel]] = {
    "random_forest":       RandomForestModel,
    "xgboost":             XGBoostModel,
    "lightgbm":            LightGBMModel,
    "catboost":            CatBoostModel,
    "mlp":                 MLPModel,
    "logistic_regression": LogisticRegressionModel,
}

# Short CLI aliases.
ALIASES: dict[str, str] = {
    "rf":   "random_forest",
    "xgb":  "xgboost",
    "lgbm": "lightgbm",
    "cat":  "catboost",
    "nn":   "mlp",
    "lr":   "logistic_regression",
}


def resolve_name(name: str) -> str:
    """Translate a CLI alias to a canonical model name."""
    return ALIASES.get(name, name)


def build_model(
    name: str,
    cfg: dict[str, Any],
    scaler_kind: str = "standard",
) -> BaseModel:
    """Return a built (but unfit) ``BaseModel`` for ``name``."""
    canonical = resolve_name(name)
    if canonical not in MODEL_CLASSES:
        raise KeyError(
            f"Unknown model {name!r}. Available: {sorted(MODEL_CLASSES)}."
        )
    model_cfg = cfg.get("models", {}).get(canonical, {})
    model = MODEL_CLASSES[canonical](
        config=model_cfg,
        scaler_kind=scaler_kind,
    )
    return model


def available_models(
    cfg: dict[str, Any],
    scaler_kind: str = "standard",
) -> dict[str, BaseModel]:
    """Build every enabled model in ``cfg.models``.

    A model is enabled if either ``enabled`` is true (default) or the key
    is present at all. Caller-facing order matches ``MODEL_CLASSES``.
    """
    out: dict[str, BaseModel] = {}
    for canonical in MODEL_CLASSES:
        mcfg = cfg.get("models", {}).get(canonical)
        if mcfg is None:
            continue
        if not mcfg.get("enabled", True):
            logger.info("Skipping disabled model: %s", canonical)
            continue
        out[canonical] = MODEL_CLASSES[canonical](
            config=mcfg,
            scaler_kind=scaler_kind,
        )
    return out
