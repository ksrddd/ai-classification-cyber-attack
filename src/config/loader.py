"""YAML config loader.

Single entry point: ``load_config()`` returns a dict. Validates that
critical keys exist and that the per-mode label lists in YAML agree
with the source-of-truth tuples in ``src.config.constants``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from src.config.constants import (
    BINARY_LABELS,
    CLASSIFICATION_MODES,
    CONFIG_PATH,
    MULTICLASS_LABELS,
)

logger = logging.getLogger(__name__)

# Top-level YAML keys we require. Missing any of these = misconfiguration.
_REQUIRED_SECTIONS: tuple[str, ...] = (
    "project",
    "classification",
    "data",
    "preprocessing",
    "feature_selection",
    "cv",
    "models",
    "tuning",
    "evaluation",
    "shap",
    "dashboard",
    "logging",
)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the project YAML config.

    Parameters
    ----------
    path
        Optional override for the config file location. Defaults to
        ``src/config/config.yaml`` (via ``CONFIG_PATH``).

    Returns
    -------
    dict
        Parsed config. Keys mirror the YAML structure.

    Raises
    ------
    FileNotFoundError
        If the config file is missing.
    ValueError
        If a required section is absent or a label list diverges from
        ``src.config.constants``.
    """
    cfg_path = Path(path) if path else CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    _validate(cfg)
    logger.debug("Loaded config from %s", cfg_path)
    return cfg


def get_classification_mode(cfg: dict[str, Any]) -> str:
    """Return the active classification mode from a loaded config."""
    return cfg["classification"]["mode"]


def get_active_target_labels(cfg: dict[str, Any]) -> tuple[str, ...]:
    """Return the label tuple for the active classification mode."""
    mode = get_classification_mode(cfg)
    if mode == "binary":
        return tuple(cfg["data"]["binary_labels"])
    return tuple(cfg["data"]["multiclass_labels"])


def _validate(cfg: dict[str, Any]) -> None:
    missing = [s for s in _REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(f"config.yaml is missing required sections: {missing}")

    mode = cfg["classification"]["mode"]
    if mode not in CLASSIFICATION_MODES:
        raise ValueError(
            f"classification.mode={mode!r} is not in {CLASSIFICATION_MODES}"
        )

    yaml_binary = tuple(cfg["data"]["binary_labels"])
    if yaml_binary != BINARY_LABELS:
        raise ValueError(
            "config.yaml::data.binary_labels diverges from constants.BINARY_LABELS. "
            f"YAML={yaml_binary}  CONST={BINARY_LABELS}."
        )

    yaml_multi = tuple(cfg["data"]["multiclass_labels"])
    if yaml_multi != MULTICLASS_LABELS:
        raise ValueError(
            "config.yaml::data.multiclass_labels diverges from "
            "constants.MULTICLASS_LABELS. "
            f"YAML={yaml_multi}  CONST={MULTICLASS_LABELS}."
        )
