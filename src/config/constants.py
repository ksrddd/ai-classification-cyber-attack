"""Project-wide constants.

Every place in the codebase that needs a random seed, a target class list,
or a canonical path reads it from this module. One place to change.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# Single seed across the project. Pass to every scikit-learn estimator,
# every numpy RandomState, every train_test_split.
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Classification modes + target labels
# ---------------------------------------------------------------------------
# Two modes the project supports end-to-end:
#   - "binary":     BENIGN -> "Normal", everything else -> "Attack"
#   - "multiclass": 10-class grouping of CICIDS2017 attack families
#
# Both label lists are tuples (immutable, hashable) so they can be used as
# class-order keys in sklearn LabelEncoder without surprise mutation.

CLASSIFICATION_MODES: tuple[str, ...] = ("binary", "multiclass")
DEFAULT_CLASSIFICATION_MODE: str = "multiclass"

BINARY_LABELS: tuple[str, ...] = (
    "Normal",
    "Attack",
)

MULTICLASS_LABELS: tuple[str, ...] = (
    "BENIGN",
    "DoS",
    "DDoS",
    "PortScan",
    "Bot",
    "Web Attack",
    "Brute Force",
    "Infiltration",
    "Heartbleed",
    "Other",
)


def get_target_labels(mode: str) -> tuple[str, ...]:
    """Return the canonical label tuple for the given classification mode."""
    if mode == "binary":
        return BINARY_LABELS
    if mode == "multiclass":
        return MULTICLASS_LABELS
    raise ValueError(
        f"Unknown classification_mode: {mode!r}. "
        f"Must be one of {CLASSIFICATION_MODES}."
    )


# ---------------------------------------------------------------------------
# Canonical project paths
# ---------------------------------------------------------------------------
# constants.py lives at <root>/src/config/constants.py -- go up 2 levels.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR:       Path = PROJECT_ROOT / "data"
RAW_DIR:        Path = DATA_DIR / "raw"
INTERIM_DIR:    Path = DATA_DIR / "interim"
PROCESSED_DIR:  Path = DATA_DIR / "processed"
SAMPLE_DIR:     Path = DATA_DIR / "sample"

MODELS_DIR:     Path = PROJECT_ROOT / "models"
REPORTS_DIR:    Path = PROJECT_ROOT / "reports"
RESULTS_DIR:    Path = PROJECT_ROOT / "results"
METRICS_DIR:    Path = RESULTS_DIR / "metrics"
FIGURES_DIR:    Path = RESULTS_DIR / "figures"
SHAP_DIR:       Path = RESULTS_DIR / "shap"
LOGS_DIR:       Path = PROJECT_ROOT / "logs"

CONFIG_PATH:    Path = PROJECT_ROOT / "src" / "config" / "config.yaml"

# The cleaned CICIDS2017 corpus. Built by `main.py --stage preprocess` and
# consumed by training, the audit stage, and SHAP explainability.
CLEAN_CACHE_PATH: Path = PROCESSED_DIR / "cicids2017_clean.parquet"

# Chronological split manifest that defines the delivery protocol.
SPLIT_MANIFEST_PATH: Path = (
    PROJECT_ROOT / "configs" / "splits" / "cicids2017_temporal_70_30.json"
)

# ---------------------------------------------------------------------------
# Schema knobs (used by data/schema.py)
# ---------------------------------------------------------------------------
# The CICIDS2017 CSVs ship with a leading space on every column name -- a
# well-known bug. The loader strips it; tests assert it stays stripped.
LABEL_COLUMN: str = "Label"

# After label normalization the mapped column is appended alongside the
# raw column. Downstream code reads from this name, not from LABEL_COLUMN.
MAPPED_LABEL_COLUMN: str = "Label_mapped"

# ---------------------------------------------------------------------------
# Metrics payload schema
# ---------------------------------------------------------------------------
# Keys in a per-model metrics payload whose values are dicts or lists rather
# than scalars. Anything listed here must be dropped before the payload is
# flattened into a comparison DataFrame, because those tables format every
# cell as a number. Lives here rather than in train.py so the dashboard can
# read it without importing the training stack.
NON_SCALAR_METRIC_KEYS: tuple[str, ...] = (
    "best_params",
    "cv_f1_macro_scores",
    "reportable_classes",
    "per_class_precision",
    "per_class_recall",
    "per_class_f1",
    "per_class_fpr",
    "per_class_fnr",
    "per_class_support",
    "per_class_false_positives",
    "per_class_false_negatives",
)
