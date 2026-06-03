"""Deterministic seeding for reproducibility (ADR-009).

Any code path that touches randomness should call ``seed_everything()``
at the top. ``RANDOM_STATE`` from ``src.config.constants`` is the single
source of truth.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

from src.config.constants import RANDOM_STATE

logger = logging.getLogger(__name__)


def seed_everything(seed: int = RANDOM_STATE) -> None:
    """Seed Python ``random``, numpy, and the ``PYTHONHASHSEED`` env var.

    Note: scikit-learn estimators take ``random_state=`` directly — that
    parameter is wired in the model classes (Phase 6), not here.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    logger.debug("Seeded random, numpy, PYTHONHASHSEED to %d", seed)
