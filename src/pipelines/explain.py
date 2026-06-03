"""Explainability pipeline stage.

Loads the trained Random Forest, runs SHAP, emits plots + ``shap_report.md``.
Phase 9.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run(config_path: Path) -> None:
    raise NotImplementedError("Phase 9 will implement SHAP pipeline.")
