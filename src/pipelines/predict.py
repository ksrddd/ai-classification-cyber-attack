"""Inference pipeline -- used by the Dashboard's Predict-New-CSV page
and ``python main.py --stage predict --input <path>``.

This file is a thin wrapper that picks a default output path and
delegates to :mod:`src.inference.predictor`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config.constants import PROCESSED_DIR
from src.inference.predictor import predict_csv

logger = logging.getLogger(__name__)


def run(
    input_csv: Path,
    model_name: str = "random_forest",
    output_csv: Path | None = None,
) -> Path:
    """Predict labels for every row in ``input_csv``; return the output path."""
    if output_csv is None:
        output_csv = PROCESSED_DIR / f"predictions_{Path(input_csv).stem}.csv"
    result = predict_csv(
        input_csv=Path(input_csv),
        model_name=model_name,
        output_csv=output_csv,
        include_probabilities=True,
    )
    logger.info(
        "Predicted %d rows with %s -> %s",
        len(result.predictions), model_name, output_csv,
    )
    return Path(output_csv)
