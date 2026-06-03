"""Schema validation for user-uploaded CSVs (Dashboard Page 6).

When a user uploads a CSV via the Streamlit dashboard, we must check
that it has the same feature columns the training Pipeline expects.
Mismatch -> friendly error, no inference attempt.

Phase 10 implementation:
- ``validate_inference_csv(df, expected_features)`` -> tuple[bool, list[str]]
  Returns (ok, list_of_missing_or_extra_columns).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# TODO(phase 10): implement.
