"""Smoke tests for the config layer."""

from __future__ import annotations

import pytest

from src.config.constants import (
    BINARY_LABELS,
    CLASSIFICATION_MODES,
    CONFIG_PATH,
    MULTICLASS_LABELS,
    RANDOM_STATE,
    get_target_labels,
)
from src.config.loader import (
    get_active_target_labels,
    get_classification_mode,
    load_config,
)


def test_random_state_is_42() -> None:
    """Single seed across the project."""
    assert RANDOM_STATE == 42


def test_binary_labels_are_two_classes() -> None:
    assert BINARY_LABELS == ("Normal", "Attack")


def test_multiclass_labels_are_ten_classes() -> None:
    assert len(MULTICLASS_LABELS) == 10
    assert "BENIGN" in MULTICLASS_LABELS
    assert "Other" in MULTICLASS_LABELS


def test_get_target_labels_dispatches_by_mode() -> None:
    assert get_target_labels("binary") == BINARY_LABELS
    assert get_target_labels("multiclass") == MULTICLASS_LABELS


def test_get_target_labels_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown classification_mode"):
        get_target_labels("triclass")


def test_config_yaml_loads_and_agrees_with_constants() -> None:
    cfg = load_config()
    assert cfg["project"]["random_state"] == RANDOM_STATE
    assert tuple(cfg["data"]["binary_labels"]) == BINARY_LABELS
    assert tuple(cfg["data"]["multiclass_labels"]) == MULTICLASS_LABELS
    assert cfg["classification"]["mode"] in CLASSIFICATION_MODES


def test_get_active_target_labels_matches_mode() -> None:
    cfg = load_config()
    mode = get_classification_mode(cfg)
    expected = BINARY_LABELS if mode == "binary" else MULTICLASS_LABELS
    assert get_active_target_labels(cfg) == expected


def test_config_path_is_resolvable() -> None:
    assert CONFIG_PATH.exists(), f"config.yaml should exist at {CONFIG_PATH}"
