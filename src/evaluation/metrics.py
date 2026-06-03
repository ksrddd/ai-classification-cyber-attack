"""Per-model metric computation.

Computes the metric set from ``config.yaml::evaluation.metrics``:
accuracy, precision/recall/F1 (weighted + macro + per class),
ROC-AUC (One-vs-Rest), Matthews correlation coefficient.

Returns a single dict that becomes a row in the comparison CSV.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# TODO(phase 8):
#   compute_metrics(y_true, y_pred, y_proba=None, labels=None) -> dict[str, float]
#   classification_report_df(y_true, y_pred, labels) -> pd.DataFrame
