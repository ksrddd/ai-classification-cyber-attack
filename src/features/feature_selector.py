"""Feature selection — four methods compared (Phase 5).

Methods (build prompt §5):
  1. Correlation analysis — drop one of any pair above threshold.
  2. Random Forest feature importance — top-K by mean decrease in impurity.
  3. SelectKBest — ANOVA F-statistic for classification.
  4. Recursive Feature Elimination — RFE wrapping a base estimator.

Output: ``reports/feature_selection_report.md`` comparing the methods'
agreement (Jaccard overlap matrix) plus the union/intersection sets.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# TODO(phase 5):
#   - select_by_correlation(X, threshold) -> list[str]
#   - select_by_rf_importance(X, y, top_k) -> list[str]
#   - select_by_k_best(X, y, k) -> list[str]
#   - select_by_rfe(X, y, n_features) -> list[str]
#   - compare_methods(results: dict[str, list[str]]) -> pd.DataFrame
#   - generate_report(comparison) -> writes feature_selection_report.md
