"""Evaluation pipeline stage.

Loads each trained model, evaluates on the held-out test set, emits
the comparison report. Phase 8.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run(config_path: Path) -> None:
    raise NotImplementedError("Phase 8 will implement evaluation pipeline.")
