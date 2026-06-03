"""Cross-model comparison report.

Builds a single Markdown + CSV + PNG comparing the three models across
baseline and tuned variants. This is the artifact you'll show the panel.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# TODO(phase 8):
#   build_comparison_table(per_model_metrics: dict) -> pd.DataFrame
#   write_comparison_report(df, save_to) -> writes .md + .csv + .png
