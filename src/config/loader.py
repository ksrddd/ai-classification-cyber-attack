"""YAML config loader.

Single entry point: ``load_config()`` returns a frozen dict-like object.
Validates that critical keys exist; raises early with a clear message
rather than letting a ``KeyError`` surface deep in a pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from src.config.constants import CONFIG_PATH, TARGET_LABELS

logger = logging.getLogger(__name__)

# Top-level YAML keys we require. Missing any of these = misconfiguration.
_REQUIRED_SECTIONS: tuple[str, ...] = (
    "project", "data", "preprocessing", "cv", "models",
    "evaluation", "shap", "dashboard", "logging",
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
        If a required section is absent or target_labels diverges from
        ``constants.TARGET_LABELS``.
    """
    cfg_path = Path(path) if path else CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    _validate(cfg)
    logger.debug("Loaded config from %s", cfg_path)
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    missing = [s for s in _REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(f"config.yaml is missing required sections: {missing}")

    # constants.TARGET_LABELS is the source of truth; config.yaml must agree.
    yaml_labels = tuple(cfg["data"]["target_labels"])
    if yaml_labels != TARGET_LABELS:
        raise ValueError(
            "config.yaml::data.target_labels diverges from "
            "src.config.constants.TARGET_LABELS. "
            f"YAML={yaml_labels}  CONST={TARGET_LABELS}. "
            "Edit both to keep them in sync."
        )
