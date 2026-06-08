"""EDA pipeline stage -- loads raw data, runs EDA, prints + saves summary.

Invoked via ``python main.py --stage eda``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.config.constants import (
    FIGURES_DIR,
    MAPPED_LABEL_COLUMN,
    METRICS_DIR,
)
from src.config.loader import get_classification_mode, load_config
from src.data.eda import run_eda
from src.data.label_mapping import add_mapped_column
from src.data.loader import load_raw
from src.features.cleaning import clean, drop_other_class
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
    mode = get_classification_mode(cfg)
    drop_other = cfg["data"].get("drop_other_class", False)

    raw_dir = raw_dir_override if raw_dir_override is not None else Path(cfg["data"]["raw_dir"])
    df = load_raw(
        files=cfg["data"].get("required_files"),
        raw_dir=raw_dir,
        subsample_n=cfg["data"].get("subsample_n"),
    )
    df = clean(df)
    df = add_mapped_column(df, mode=mode)
    if drop_other and mode == "multiclass":
        df = drop_other_class(df, label_col=MAPPED_LABEL_COLUMN)

    summary = run_eda(df, output_dir=FIGURES_DIR, label_col=MAPPED_LABEL_COLUMN)
    summary["classification_mode"] = mode

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
