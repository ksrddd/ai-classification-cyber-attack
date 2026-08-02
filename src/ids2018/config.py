"""Constants and hyper-parameters for the CSE-CIC-IDS2018 pipeline.

Everything that a user might reasonably want to change lives here, so the
other modules stay free of magic numbers. Nothing in this file imports from
the CICIDS2017 side of the project.
"""

from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
# Root of the 10 raw daily CSV exports shipped by the Canadian Institute
# for Cybersecurity. Override with --raw-dir on the CLI.
RAW_DIR = Path("D:/CSE-CIC-IDS2018")

# All generated artefacts land under these two directories so the 2017
# outputs are never overwritten.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "ids2018"
MODELS_DIR = PROJECT_ROOT / "models" / "ids2018"
CACHE_DIR = PROJECT_ROOT / "data" / "ids2018"

# Pass 1 produces one int16 label code per corpus row (~26 MB for 13M rows).
# Caching it means the 300k / 500k / 1M runs pay the 6.7 GB CSV scan once
# between them instead of once each.
INDEX_CACHE = CACHE_DIR / "corpus_label_index.npz"


def size_tag(n_rows: int) -> str:
    """Short, filesystem-safe label for a sample size: 300000 -> '300k'."""
    if n_rows % 1_000_000 == 0:
        return f"{n_rows // 1_000_000}m"
    if n_rows % 1_000 == 0:
        return f"{n_rows // 1_000}k"
    return str(n_rows)


def sample_cache_path(n_rows: int) -> Path:
    """Per-size Parquet cache, so each rung of the ladder keeps its own copy."""
    return CACHE_DIR / f"sample_{size_tag(n_rows)}.parquet"

# ----------------------------------------------------------------------
# Sampling / splitting
# ----------------------------------------------------------------------
RANDOM_STATE = 42

# Total rows drawn from the ~13M-row corpus via stratified sampling.
SAMPLE_SIZE = 300_000

# 70 / 30 train-test split -> 210,000 / 90,000 rows.
TEST_SIZE = 0.30

# Rows read per chunk during both scan passes. 500k x 84 float columns is
# roughly 350 MB peak, which is safe on a 16 GB machine. Lower it if the
# machine is tight on RAM.
CHUNKSIZE = 500_000

# Proportional stratification would give ultra-rare classes (SQL Injection
# has ~87 rows in 13M) only ~2 sampled rows, and a stratified 70/30 split
# needs at least 2 members per class or scikit-learn raises. This floor
# guarantees every attack class survives both the sample and the split.
# Set to 0 for strictly proportional sampling with no adjustment.
MIN_PER_CLASS = 2

# How the sample is drawn. Both modes are stratified; neither ever uses
# simple random sampling.
#
#   "nested"       Per-stratum proportional allocation over a seed-fixed
#                  shuffle of each class. Sample sizes are *nested*:
#                  300k subset of 500k subset of 1M. Use this when comparing
#                  several sample sizes, so the only thing that changes
#                  between runs is how much data there is.
#   "independent"  scikit-learn's StratifiedShuffleSplit. Each size is an
#                  independent draw -- correct for a single run, but two
#                  sizes then differ in *which* rows they got as well as
#                  how many.
SAMPLING_MODE = "nested"

# ----------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------
LABEL_COL = "Label"

# Flow identifiers. Present only in 02-20-2018.csv, which was exported with
# a wider schema than the other nine files. They are pure identifiers: an IP
# address memorises "which host attacked" rather than "what an attack looks
# like", so keeping them inflates accuracy and destroys generalisation.
IDENTITY_COLS = ["Flow ID", "Src IP", "Src Port", "Dst IP"]

# The subset of the above that holds text rather than numbers. Src Port is a
# number and is excluded, so it is never mangled by numeric coercion.
TEXT_IDENTITY_COLS = ["Flow ID", "Src IP", "Dst IP"]

# Wall-clock capture time. Since each attack was run in its own time window,
# Timestamp is a perfect label proxy -- textbook target leakage.
TIMESTAMP_COL = "Timestamp"

# Destination port is high-cardinality and encodes the lab setup (e.g. HOIC
# always hit port 80). Dropped by default per the project spec; pass
# --keep-dst-port to retain it as a numeric feature.
DST_PORT_COL = "Dst Port"

# Canonical spelling for every label in the corpus. The raw CSVs use
# inconsistent casing and spacing across days ("DDOS attack-HOIC" vs
# "DDoS attacks-LOIC-HTTP"), and "Infilteration" is misspelt upstream --
# we keep the upstream spelling as the canonical key so it matches the
# published class distribution, and only normalise whitespace/casing noise.
LABEL_ALIASES: dict[str, str] = {
    "benign": "Benign",
    "ftp-bruteforce": "FTP-BruteForce",
    "ssh-bruteforce": "SSH-Bruteforce",
    "dos attacks-goldeneye": "DoS attacks-GoldenEye",
    "dos attacks-slowloris": "DoS attacks-Slowloris",
    "dos attacks-slowhttptest": "DoS attacks-SlowHTTPTest",
    "dos attacks-hulk": "DoS attacks-Hulk",
    "ddos attacks-loic-http": "DDoS attacks-LOIC-HTTP",
    "ddos attack-hoic": "DDOS attack-HOIC",
    "ddos attack-loic-udp": "DDOS attack-LOIC-UDP",
    "brute force -web": "Brute Force -Web",
    "brute force -xss": "Brute Force -XSS",
    "sql injection": "SQL Injection",
    "infilteration": "Infilteration",
    "infiltration": "Infilteration",
    "bot": "Bot",
}

# Rows whose Label equals this are repeated CSV headers embedded mid-file --
# an artefact of how CIC concatenated the daily captures. Every one of the
# ten files contains them.
EMBEDDED_HEADER_TOKEN = "label"

# ----------------------------------------------------------------------
# Model hyper-parameters
# ----------------------------------------------------------------------
# How class imbalance is handled. This is a *global* switch on purpose:
# scikit-learn's MLPClassifier supports neither `class_weight` nor
# `sample_weight`, so weighting only six of the seven models would make the
# comparison table meaningless -- the unweighted model would post a higher
# accuracy purely because it was allowed to favour the majority class.
#
#   "none"      Every model sees the identical training set with identical
#               (uniform) weights. The only fair 7-way comparison, and the
#               setup most published CICIDS results use.
#   "balanced"  Inverse-frequency weighting for the six models that support
#               it. Better macro recall on rare attacks, but MLP is then
#               NOT comparable with the rest, and linear models degrade
#               badly because the Benign:SQL-Injection weight ratio exceeds
#               100,000:1.
CLASS_WEIGHTING = "none"

# Applied on top of MODEL_PARAMS only when CLASS_WEIGHTING == "balanced".
CLASS_WEIGHT_PARAMS: dict[str, dict] = {
    "random_forest": {"class_weight": "balanced_subsample"},
    "lightgbm": {"class_weight": "balanced"},
    "catboost": {"auto_class_weights": "Balanced"},
    "logistic_regression": {"class_weight": "balanced"},
}

# Deliberately modest depths/estimator counts: the goal is a fair 7-way
# comparison on 210k training rows, not a squeeze for the last 0.1% F1.
MODEL_PARAMS: dict[str, dict] = {
    "random_forest": {
        "n_estimators": 300,
        "max_depth": 30,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    },
    "xgboost": {
        "n_estimators": 400,
        "max_depth": 8,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        # Stated explicitly rather than left to the library default, so
        # XGBoost and LightGBM are regularised identically -- see the note
        # on LightGBM's reg_lambda below.
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "eval_metric": "mlogloss",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    },
    "lightgbm": {
        "n_estimators": 400,
        "num_leaves": 63,
        "max_depth": -1,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        # REQUIRED, not a tuning choice. LightGBM defaults reg_lambda to 0
        # while XGBoost defaults it to 1, and with an unregularised leaf the
        # optimal value -g/h is unbounded. SQL Injection contributes a single
        # training row, so its leaves blow up, the softmax overflows, and the
        # damage spreads to every class: accuracy climbs to 0.978 by round 10,
        # then collapses to 0.678 -- worse than always predicting Benign --
        # and LightGBM stops finding splits at round 59 of 400. reg_lambda=1
        # bounds the leaf value, restores all 6000 trees, and puts LightGBM on
        # the same regularisation footing as XGBoost.
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "verbose": -1,
        "random_state": RANDOM_STATE,
    },
    "catboost": {
        "iterations": 400,
        "depth": 8,
        "learning_rate": 0.1,
        "loss_function": "MultiClass",
        "verbose": 0,
        "allow_writing_files": False,
        "random_seed": RANDOM_STATE,
    },
    "mlp": {
        "hidden_layer_sizes": (128, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-4,
        "batch_size": 512,
        "learning_rate_init": 1e-3,
        "max_iter": 60,
        "early_stopping": True,
        "n_iter_no_change": 8,
        "validation_fraction": 0.1,
        "random_state": RANDOM_STATE,
    },
    "logistic_regression": {
        # lbfgs, not saga. The first run used saga/max_iter=300 and hit
        # ConvergenceWarning, so its numbers described an unfinished fit
        # rather than what the model can do. lbfgs is a quasi-Newton method
        # and converges reliably on this problem (210k x 69, dense, already
        # standardised), while saga's stochastic updates need far more
        # epochs at this scale.
        "solver": "lbfgs",
        "max_iter": 1000,
        "C": 1.0,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    },
    "stacking": {
        # Base learners are the three cheapest strong models; the meta
        # learner sees their out-of-fold probabilities.
        "cv": 3,
        "n_jobs": 1,  # base learners already use n_jobs=-1 internally
        "passthrough": False,
        "final_estimator_max_iter": 500,
    },
}

# Canonical run order for the 7-model comparison.
MODEL_NAMES = [
    "xgboost",
    "lightgbm",
    "catboost",
    "random_forest",
    "mlp",
    "logistic_regression",
    "stacking",  # last: it refits its own base learners
]
