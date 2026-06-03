"""Label encoding helpers.

Wraps ``sklearn.preprocessing.LabelEncoder`` with project-aware defaults:
- Asserts the encoded label set matches ``TARGET_LABELS``.
- Persists the fitted encoder via joblib to ``models/label_encoder.joblib``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# TODO(phase 4):
#   - fit_label_encoder(y) -> LabelEncoder
#   - save_label_encoder(le, path)
#   - load_label_encoder(path) -> LabelEncoder
