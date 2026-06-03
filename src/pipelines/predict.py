"""Inference pipeline — used by the Dashboard's "Predict New CSV" page (Page 6).

Takes a CSV path, applies the saved preprocessing Pipeline + label encoder,
runs the saved model, returns / saves predictions.

Phase 12 implementation.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run(input_csv: Path, model_name: str = "random_forest",
        output_csv: Path | None = None) -> Path:
    raise NotImplementedError("Phase 12 will implement predict pipeline.")
