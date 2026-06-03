"""sklearn Pipeline assembly — the leak-proof core (ADR-006).

Every model is wrapped as::

    Pipeline([
        ("scaler", StandardScaler()),  # or MinMax via config
        ("clf",    <model>),
    ])

When this Pipeline is handed to ``GridSearchCV(cv=5)``, the scaler refits
on each training fold — test data never influences scaler statistics.
That's the structural defence against data leakage (R-03).

Phase 5/6 will implement:
- ``build_preprocessing(scaler_kind)`` -> single transformer.
- ``build_model_pipeline(model_name)`` -> full ``Pipeline``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# TODO(phase 5/6): see module docstring.
