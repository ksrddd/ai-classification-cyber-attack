"""Project-wide constants.

Every place in the codebase that needs a random seed, a target class list,
or a canonical path reads it from this module. One place to change.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# ADR-009: single seed across the project. Pass to every scikit-learn estimator,
# every numpy RandomState, every train_test_split.
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Target classes
# ---------------------------------------------------------------------------
# ADR-002: scoped to four CICIDS2017 labels. Extending = edit this list AND
# config.yaml::data.target_labels (they must stay in sync; see validator).
TARGET_LABELS: tuple[str, ...] = (
    "BENIGN",
    "DoS Hulk",
    "PortScan",
    "FTP-Patator",
)

# ---------------------------------------------------------------------------
# Canonical project paths
# ---------------------------------------------------------------------------
# constants.py lives at <root>/src/config/constants.py — go up 2 levels.
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

# ---------------------------------------------------------------------------
# Schema knobs (used by data/schema.py)
# ---------------------------------------------------------------------------
# The CICIDS2017 CSVs ship with a leading space on every column name — a
# well-known bug. The loader strips it; tests assert it stays stripped.
LABEL_COLUMN: str = "Label"
