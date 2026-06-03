"""Data cleaning steps applied BEFORE the sklearn Pipeline.

Why outside the Pipeline: these operations change row count (drop NaN/Inf,
drop duplicates), which sklearn transformers don't support cleanly.
Anything that preserves row count (scaling, encoding) goes inside the Pipeline.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def clean(df: pd.DataFrame, drop_dup: bool = True, drop_inf: bool = True,
          drop_na: bool = True) -> pd.DataFrame:
    """Apply row-count-changing cleaning steps.

    Order (Phase 4 implementation):
      1. Strip column-name whitespace (CICIDS bug).
      2. Replace ``±inf`` with ``NaN``.
      3. Drop rows with any NaN.
      4. Drop exact duplicate rows.

    Returns a *new* DataFrame; does not mutate input.
    """
    # TODO(phase 4): implement and log row deltas at each step.
    raise NotImplementedError("Phase 4 implementation pending.")
