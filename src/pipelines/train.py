"""Training pipeline stage.

Reads config, loads cleaned data, builds the model Pipeline, runs
GridSearchCV, saves the best estimator + cv_results.
Used by ``python main.py --stage train --model {lr|rf|mlp|all}``.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run(config_path: Path, model: str = "all") -> None:
    """Train the requested model(s) end-to-end. Phase 6 implementation."""
    raise NotImplementedError("Phase 6 will implement training pipeline.")
