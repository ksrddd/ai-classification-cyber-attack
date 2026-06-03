"""EDA pipeline stage — loads raw data, runs EDA, prints + saves summary.

Invoked via ``python main.py --stage eda``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.config.constants import FIGURES_DIR, METRICS_DIR
from src.config.loader import load_config
from src.data.eda import run_eda
from src.data.loader import load_raw
from src.features.cleaning import clean, filter_target_classes
from src.utils.io import ensure_dir

logger = logging.getLogger(__name__)


def run(config_path: Path, raw_dir_override: Path | None = None) -> dict:
    """Run the EDA stage. Returns the summary dict (also written as JSON).

    Parameters
    ----------
    config_path
        Path to config.yaml.
    raw_dir_override
        If set, load from this directory and glob ALL CSVs there
        (ignoring config's ``required_files``). Use this to run against
        ``data/sample/`` without modifying the config.
    """
    cfg = load_config(config_path)
    target_labels = cfg["data"]["target_labels"]

    if raw_dir_override is not None:
        logger.info("Using raw_dir override: %s (ignoring required_files)", raw_dir_override)
        df = load_raw(
            raw_dir=raw_dir_override,
            subsample_n=cfg["data"].get("subsample_n"),
        )
    else:
        df = load_raw(
            files=cfg["data"].get("required_files"),
            subsample_n=cfg["data"].get("subsample_n"),
        )
    df = clean(df)
    df = filter_target_classes(df, target_labels)

    summary = run_eda(df, output_dir=FIGURES_DIR)
    ensure_dir(METRICS_DIR)
    summary_path = METRICS_DIR / "eda_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2, default=str)
    logger.info("EDA summary written to %s", summary_path)
    return summary


def _jsonable(obj):
    """Make pandas / numpy types JSON-serializable."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalars
        return obj.item()
    return obj
