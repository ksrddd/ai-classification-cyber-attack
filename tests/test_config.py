"""Smoke tests for the config layer."""

from __future__ import annotations

from src.config.constants import RANDOM_STATE, TARGET_LABELS, CONFIG_PATH
from src.config.loader import load_config


def test_random_state_is_42() -> None:
    """ADR-009 sanity check."""
    assert RANDOM_STATE == 42


def test_target_labels_are_four_classes() -> None:
    """ADR-002: the four classes are locked."""
    assert TARGET_LABELS == ("BENIGN", "DoS Hulk", "PortScan", "FTP-Patator")


def test_config_yaml_loads_and_agrees_with_constants() -> None:
    """config.yaml must parse and its target_labels must match the constants."""
    cfg = load_config()
    assert cfg["project"]["random_state"] == RANDOM_STATE
    assert tuple(cfg["data"]["target_labels"]) == TARGET_LABELS


def test_config_path_is_resolvable() -> None:
    assert CONFIG_PATH.exists(), f"config.yaml should exist at {CONFIG_PATH}"
