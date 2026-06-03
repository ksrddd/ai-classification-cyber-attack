"""Exploratory data analysis.

All plots are saved under ``results/figures/`` so the dashboard and reports
can pull them without re-running EDA.

Phase 4 will implement:
- ``describe_dataset(df)`` -> shape, dtypes, class counts, missing/inf audit.
- ``plot_class_distribution(df)`` -> bar chart.
- ``plot_correlation_heatmap(df)`` -> top-K correlated features.
- ``plot_feature_distributions(df)`` -> histogram grid for top features.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# TODO(phase 4): functions listed in module docstring.
