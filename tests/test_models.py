"""Tests for src.models (RF, XGBoost, LightGBM, CatBoost, MLP)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config.loader import load_config
from src.models.base import BaseModel
from src.models.registry import (
    ALIASES,
    MODEL_CLASSES,
    available_models,
    build_model,
    resolve_name,
)


def test_model_registry_lists_all_five_primary_models() -> None:
    for name in ("random_forest", "xgboost", "lightgbm", "catboost", "mlp"):
        assert name in MODEL_CLASSES


def test_aliases_resolve_to_canonical_names() -> None:
    assert resolve_name("rf") == "random_forest"
    assert resolve_name("xgb") == "xgboost"
    assert resolve_name("lgbm") == "lightgbm"
    assert resolve_name("cat") == "catboost"
    assert resolve_name("nn") == "mlp"


def test_base_model_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseModel(config={})  # type: ignore[abstract]


def test_build_model_returns_buildable_pipeline() -> None:
    cfg = load_config()
    m = build_model("rf", cfg)
    pipe = m.build()
    assert isinstance(pipe, Pipeline)
    assert "clf" in pipe.named_steps


@pytest.mark.parametrize("name", list(ALIASES.keys()))
def test_every_alias_builds_a_pipeline(name: str) -> None:
    cfg = load_config()
    m = build_model(name, cfg)
    pipe = m.build()
    assert isinstance(pipe, Pipeline)


def test_available_models_returns_only_enabled() -> None:
    cfg = load_config()
    # Disable mlp via in-memory cfg copy.
    import copy
    cfg2 = copy.deepcopy(cfg)
    cfg2["models"]["mlp"]["enabled"] = False
    out = available_models(cfg2)
    assert "mlp" not in out
    assert "random_forest" in out


def test_random_forest_trains_and_predicts_on_synthetic_data() -> None:
    """End-to-end smoke: train a tiny RF on toy data and predict."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 4)), columns=list("abcd"))
    y = pd.Series(rng.integers(0, 3, size=60))
    cfg = load_config()
    m = build_model("rf", cfg)
    m.config = {**m.config, "baseline": {"n_estimators": 10, "n_jobs": 1}}
    m.build()
    m.fit(X, y)
    preds = m.predict(X)
    assert preds.shape == (60,)
    assert set(np.unique(preds)) <= set(np.unique(y))
