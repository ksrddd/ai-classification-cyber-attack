"""Generic matplotlib/seaborn plot helpers.

Project-wide visual conventions live here so every chart looks consistent.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# TODO(phase 4 onwards):
#   set_style() -> apply seaborn defaults + figure size
#   save_fig(fig, path) -> tight_layout + savefig + close
#   bar_chart, heatmap, hist_grid helpers
