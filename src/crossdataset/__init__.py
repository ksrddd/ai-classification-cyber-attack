"""Shared schema and label vocabulary for cross-dataset 2017 <-> 2018 work.

The 2017 and 2018 pipelines were built separately and share no preprocessing:
their column names differ, their label sets differ, and their class counts
differ (9 against 15). This package holds the translation layer that lets a
model trained on one be evaluated on the other.

* :mod:`src.crossdataset.schema` -- the 77 shared feature columns.
* :mod:`src.crossdataset.labels` -- the 7 shared classes.
"""

from src.crossdataset.labels import (
    SHARED_CLASSES,
    map_to_shared,
    shared_class_counts,
)
from src.crossdataset.schema import (
    CANONICAL_FEATURES,
    DST_PORT,
    canonical_features,
    to_canonical,
)

__all__ = [
    "CANONICAL_FEATURES",
    "DST_PORT",
    "SHARED_CLASSES",
    "canonical_features",
    "map_to_shared",
    "shared_class_counts",
    "to_canonical",
]
