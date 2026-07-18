"""fast_eda_from_cache.py

รัน EDA โดยใช้ cicids_clean.parquet (ไม่โหลด raw CSV ใหม่)
สร้าง:
  results/metrics/eda_summary.json
  results/figures/class_distribution.png
  results/figures/missing_value_audit.png
  results/figures/correlation_heatmap.png
  results/figures/feature_distributions.png
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fast_eda")

import pandas as pd  # noqa: E402

from src.config.constants import FIGURES_DIR, METRICS_DIR  # noqa: E402
from src.data.eda import run_eda  # noqa: E402
from src.utils.io import ensure_dir  # noqa: E402

CLEAN_CACHE  = ROOT / "data" / "processed" / "cicids_clean.parquet"
SUBSAMPLE_N  = 300_000   # เพียงพอสำหรับ EDA
RANDOM_STATE = 42
LABEL_COL    = "Label"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def main():
    if not CLEAN_CACHE.exists():
        log.error("cicids_clean.parquet not found at %s", CLEAN_CACHE)
        sys.exit(1)

    log.info("Loading %s ...", CLEAN_CACHE)
    df = pd.read_parquet(CLEAN_CACHE)
    log.info("  -> %d rows x %d cols", *df.shape)

    # Subsample for speed (EDA doesn't need all 13.9M rows)
    if len(df) > SUBSAMPLE_N:
        log.info("Subsampling to %d rows for EDA ...", SUBSAMPLE_N)
        df = df.sample(n=SUBSAMPLE_N, random_state=RANDOM_STATE).reset_index(drop=True)
        log.info("  -> %d rows", len(df))

    ensure_dir(FIGURES_DIR)
    ensure_dir(METRICS_DIR)

    log.info("Running EDA ...")
    summary = run_eda(df, output_dir=FIGURES_DIR, label_col=LABEL_COL)
    summary["classification_mode"] = "multiclass"

    summary_path = METRICS_DIR / "eda_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2, default=str)

    log.info("EDA summary written to %s", summary_path)
    log.info("Figures written to %s", FIGURES_DIR)
    log.info("Done! EDA page should now load correctly.")


if __name__ == "__main__":
    main()
