"""SHAP analysis driver.

Phase 9 implementation:
- Build ``shap.TreeExplainer`` for the trained Random Forest.
- Compute shap_values on a stratified test sample (N from config).
- Save: summary_plot (per class), bar plot, force_plot for top-K samples.
- Extract top-5 features per attack class -> ``reports/shap_report.md``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# TODO(phase 9):
#   class ShapAnalyzer:
#       __init__(model, X_background, X_analyze, feature_names, class_names)
#       compute() -> None
#       plot_summary(save_to)
#       plot_bar(save_to)
#       plot_force(sample_idx, save_to)
#       top_features_per_class(k=5) -> dict[str, list[tuple[str, float]]]
#       write_report(save_to)
