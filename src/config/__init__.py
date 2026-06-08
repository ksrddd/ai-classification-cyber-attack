"""Project-wide configuration.

Exposes constants and the YAML config loader. All hard-coded knobs should
live in ``constants.py`` or ``config.yaml`` -- never inline in modules.
"""
from src.config.constants import (
    BINARY_LABELS,
    CLASSIFICATION_MODES,
    DEFAULT_CLASSIFICATION_MODE,
    MULTICLASS_LABELS,
    RANDOM_STATE,
    get_target_labels,
)
from src.config.loader import (
    get_active_target_labels,
    get_classification_mode,
    load_config,
)

__all__ = [
    "BINARY_LABELS",
    "CLASSIFICATION_MODES",
    "DEFAULT_CLASSIFICATION_MODE",
    "MULTICLASS_LABELS",
    "RANDOM_STATE",
    "get_active_target_labels",
    "get_classification_mode",
    "get_target_labels",
    "load_config",
]
