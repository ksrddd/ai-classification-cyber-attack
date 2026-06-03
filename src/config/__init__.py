"""Project-wide configuration.

Exposes constants and the YAML config loader. All hard-coded knobs should
live in ``constants.py`` or ``config.yaml`` — never inline in modules.
"""
from src.config.constants import RANDOM_STATE, TARGET_LABELS
from src.config.loader import load_config

__all__ = ["RANDOM_STATE", "TARGET_LABELS", "load_config"]
