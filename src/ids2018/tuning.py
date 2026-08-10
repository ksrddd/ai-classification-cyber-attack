"""Hyper-parameter search for the CSE-CIC-IDS2018 benchmark.

The 2018 pipeline shipped untuned on purpose -- ``MODEL_PARAMS`` are
"deliberately modest depths/estimator counts" chosen for a fair 7-way
comparison, and ``registry.py`` recorded ``hp_tuned = False`` to say so. This
module is what makes ``hp_tuned = True`` an honest claim instead of a flag.

Three decisions here are not defaults and are the reason this file exists.

1. **Rare classes never enter a validation fold.**
   Under the temporal split, SQL Injection contributes a single training row.
   Plain ``StratifiedKFold(3)`` warns and then puts that row in *validation* for
   one fold, where the model has by construction never seen the class -- its F1
   is 0 for reasons that have nothing to do with the hyper-parameters. Averaged
   into ``f1_macro`` that is not a weak signal, it is a wrong one, and it varies
   with the fold seed. :class:`RareAwareStratifiedKFold` pins those rows to
   every training fold and keeps them out of every validation fold, so the
   search score only ranges over classes the search can actually influence.

2. **The search runs on a stratified subsample; the winner is refit on all of
   it.** Sixty fits per model over 210k x 69 rows is hours of MLP alone. The
   subsample keeps every row of every rare class and thins the bulk classes, so
   the search sees the same class vocabulary at a fraction of the cost. This
   trades some fidelity for tractability and is stated in the run metadata
   rather than buried.

3. **``reg_lambda`` never goes below 1.0 for LightGBM.** ``config.py`` documents
   why at length: at ``reg_lambda=0`` a leaf holding SQL Injection's single row
   has unbounded optimal value, the softmax overflows, and accuracy collapses to
   below the majority-class baseline. That is a correctness bound, not a
   hyper-parameter, so the search space starts above it.

4. **Every model's search space is the same size.** Equal candidate counts over
   unequal spaces are not an equal budget, and the resulting table compares
   search effort as much as it compares models. See ``SEARCH_SPACES``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.ids2018.config import RANDOM_STATE

logger = logging.getLogger(__name__)

# Scoring metric for the search. Macro F1, not accuracy: Benign is ~83% of the
# corpus, so accuracy cannot separate a working detector from a constant.
SCORING = "f1_macro"

# Search runs on at most this many training rows unless overridden.
DEFAULT_TUNE_SUBSAMPLE = 60_000


# ----------------------------------------------------------------------
# Cross-validation that tolerates single-row classes
# ----------------------------------------------------------------------
class RareAwareStratifiedKFold:
    """Stratified k-fold that pins under-populated classes to train.

    Any class with fewer than ``n_splits`` members is *pinned*: all of its rows
    join every training fold and none ever appear in validation. Remaining
    classes are split by an ordinary :class:`StratifiedKFold`.

    The consequence is explicit and intended: the search score says nothing
    about the pinned classes. That is strictly better than the alternative,
    where a class absent from a fold's training data still contributes a
    guaranteed-zero F1 to the mean and adds seed-dependent noise to the ranking.
    """

    def __init__(self, n_splits: int = 3, *, shuffle: bool = True,
                 random_state: int = RANDOM_STATE) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2; got {n_splits}")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: ARG002
        return self.n_splits

    def pinned_classes(self, y) -> list:
        labels, counts = np.unique(np.asarray(y), return_counts=True)
        return labels[counts < self.n_splits].tolist()

    def split(self, X, y, groups=None):  # noqa: ARG002
        y = np.asarray(y)
        labels, counts = np.unique(y, return_counts=True)
        rare = set(labels[counts < self.n_splits].tolist())

        rare_mask = np.isin(y, list(rare)) if rare else np.zeros(len(y), dtype=bool)
        rare_idx = np.flatnonzero(rare_mask)
        dense_idx = np.flatnonzero(~rare_mask)

        if dense_idx.size == 0:
            raise ValueError(
                "Every class has fewer members than n_splits; cannot build folds"
            )

        inner = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state if self.shuffle else None,
        )
        for train_pos, val_pos in inner.split(dense_idx.reshape(-1, 1), y[dense_idx]):
            # Pinned rows ride along in train, never in validation.
            train_idx = np.concatenate([dense_idx[train_pos], rare_idx])
            yield np.sort(train_idx), np.sort(dense_idx[val_pos])


# ----------------------------------------------------------------------
# Search spaces
# ----------------------------------------------------------------------
# Every space below holds exactly SPACE_SIZE configurations. That is the point
# of this block, not an accident of it.
#
# The CICIDS2017 pipeline fixed its grids at 144 apiece and states why in
# ``train.py:hp_grids``: the search budget has to "stay identical across models,
# or the imbalance treatment stops being a controlled variable". This module
# shipped without that discipline. Every model drew the same 20 candidates, but
# from spaces ranging over three orders of magnitude, so what each model
# actually received was:
#
#     logistic_regression   10 configs, 10 drawn   -> 100%  (exhaustive)
#     mlp                  108 configs, 20 drawn   -> 18.5%
#     catboost             144 configs, 20 drawn   -> 13.9%
#     random_forest        432 configs, 20 drawn   ->  4.6%
#     lightgbm           1,296 configs, 20 drawn   ->  1.5%
#     xgboost            5,184 configs, 20 drawn   ->  0.4%
#
# A table built on that cannot separate a better model from a better-searched
# one, which is the only question the table exists to answer. Logistic
# regression was handed its global optimum; XGBoost was shown 0.4% of its own
# space and asked to compete with it.
#
# Bringing every space to 144 cuts both ways, and which way is decided by
# evidence rather than taste. The 300k and 500k temporal searches agreed on some
# winners and disagreed on others; a parameter both runs settled on is one the
# search has already answered, so it is pinned -- to the value that won, not to
# the old default the search rejected -- and written as a single-element list so
# ``tuning.json`` still records what was held and where. Parameters the two runs
# disagreed on stay open, because those are the ones still doing work. Spaces
# that were *under* 144 are widened along their least-settled axis instead.
#
# Pinning is a real cost, not a free tidy-up: XGBoost loses the ability to
# revisit subsample and colsample_bytree here. It is accepted because both runs
# independently chose the same value for them, which is weak evidence that the
# axis is spent -- and because the alternative, scaling n_iter to the space
# (780 candidates for XGBoost against 10 for logistic regression), costs days
# of compute to answer a question this project is not asking.
SPACE_SIZE = 144

SEARCH_SPACES: dict[str, dict[str, list]] = {
    # 4 x 4 x 3 x 3 x 1. Both runs agreed on max_features=0.3, beating the
    # 'sqrt' default; the estimator count is the only axis they split on.
    "random_forest": {
        "n_estimators": [200, 300, 400, 600],
        "max_depth": [15, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [0.3],
    },
    # 3 x 4 x 4 x 3 x 1 x 1 x 1. learning_rate and reg_lambda are the two axes
    # the runs disagreed on, so they keep their full range; depth and estimator
    # count stay open as the parameters that dominate a boosted tree's capacity.
    "xgboost": {
        "n_estimators": [200, 400, 600],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.03, 0.05, 0.1, 0.2],
        "reg_lambda": [0.5, 3.0, 10.0],
        "subsample": [0.8],
        "colsample_bytree": [1.0],
        "min_child_weight": [1],
    },
    # 4 x 4 x 3 x 3 x 1 x 1. Five of six axes disagreed between runs, so this
    # space is pinned only where it must be: n_estimators=600 won twice, and
    # colsample_bytree takes the midpoint of the two winners (0.6 and 0.8).
    "lightgbm": {
        "num_leaves": [31, 63, 127, 255],
        "learning_rate": [0.03, 0.05, 0.1, 0.2],
        # Floor of 1.0 is a correctness bound -- see the module docstring.
        "reg_lambda": [1.0, 3.0, 10.0],
        "min_child_samples": [5, 20, 50],
        "n_estimators": [600],
        "colsample_bytree": [0.8],
    },
    # 3 x 4 x 3 x 4. Already at 144 before this change, and left untouched:
    # it is the space the other five are being matched to.
    "catboost": {
        "iterations": [300, 500, 800],
        "depth": [4, 6, 8, 10],
        "learning_rate": [0.03, 0.05, 0.1],
        "l2_leaf_reg": [1, 3, 5, 9],
    },
    # 4 x 3 x 4 x 3. Widened, not pinned: at 108 this space was over-covered
    # relative to the others. learning_rate_init takes the fourth value, since
    # both runs converged on 1e-3 and a step size bracketed on one side only
    # cannot show that.
    "mlp": {
        "hidden_layer_sizes": [(128, 64), (256, 128), (128,), (256, 128, 64)],
        "alpha": [1e-5, 1e-4, 1e-3],
        "learning_rate_init": [5e-4, 1e-3, 3e-3, 5e-3],
        "batch_size": [256, 512, 1024],
    },
    # 24 x 3 x 2. The largest change in the file: this space was searched
    # exhaustively before, so logistic regression is the one model that stands
    # to score *worse* under an equal budget. That is what the unequal budget
    # was hiding. C is filled out densely rather than padding the space with
    # new axes, following the 2017 grid -- C is the parameter that governs a
    # logistic regression, and solver stays lbfgs because config.py records
    # that saga did not converge at this scale.
    "logistic_regression": {
        "C": [
            0.001, 0.002, 0.003, 0.005,
            0.01, 0.02, 0.03, 0.05,
            0.1, 0.2, 0.3, 0.5,
            1.0, 2.0, 3.0, 5.0,
            10.0, 20.0, 30.0, 50.0,
            100.0, 200.0, 300.0, 500.0,
        ],
        # At or above the sklearn default: a looser tolerance stops earlier, so
        # none of these needs more iterations than max_iter already allows.
        "tol": [1e-4, 5e-4, 1e-3],
        "fit_intercept": [True, False],
    },
}

# Stacking is not searched directly: StackingClassifier refits three base
# learners cv+1 times per candidate fit, so 20 candidates x 3 folds is roughly
# 240 base-learner fits per hyper-parameter it could possibly learn. Instead the
# ensemble inherits the tuned parameters of its base learners -- see
# apply_tuned_stacking_params in train_ids2018.
UNTUNED = frozenset({"stacking"})


def space_size(space: dict[str, list]) -> int:
    """Number of distinct candidates in a fully finite search space."""
    total = 1
    for values in space.values():
        total *= len(values)
    return total


def grid_fingerprint(name: str) -> str:
    """Content hash of a model's search space.

    ``--resume`` reuses any search result already in ``tuning.json`` rather than
    repeating it, which is right when the run was interrupted and wrong when the
    grid has been edited since -- the second case would silently pair new code
    with a winner drawn from a space that no longer exists. The size alone
    cannot catch it: a grid whose values changed but whose shape did not still
    hashes differently here. Borrowed from ``train.py:hp_grid_fingerprint``.
    """
    space = SEARCH_SPACES.get(name)
    if not space:
        return ""
    encoded = json.dumps(
        {key: list(values) for key, values in sorted(space.items())},
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


# ----------------------------------------------------------------------
# Subsampling
# ----------------------------------------------------------------------
def stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_rows: int,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Thin the bulk classes, keep every row of the rare ones.

    Proportional stratification would drop the ultra-rare classes entirely --
    SQL Injection's one row is 0.0005% of the split. Keeping them whole costs
    almost nothing and preserves the class vocabulary the search sees.
    """
    n = len(y)
    if n <= max_rows:
        return X, y, n

    rng = np.random.default_rng(random_state)
    labels, counts = np.unique(y, return_counts=True)

    # Classes small enough to keep in full; everything else is thinned to fit.
    keep_whole = counts <= 1000
    protected = np.flatnonzero(np.isin(y, labels[keep_whole]))
    budget = max(0, max_rows - protected.size)

    thinned: list[np.ndarray] = [protected]
    bulk_labels = labels[~keep_whole]
    bulk_total = int(counts[~keep_whole].sum())
    for label in bulk_labels:
        idx = np.flatnonzero(y == label)
        take = max(1, int(round(budget * idx.size / bulk_total)))
        take = min(take, idx.size)
        thinned.append(rng.choice(idx, size=take, replace=False))

    selected = np.sort(np.concatenate(thinned))
    return X[selected], y[selected], selected.size


# ----------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------
@dataclass
class TuneResult:
    """Everything a run needs to record about one model's search."""

    model: str
    tuned: bool
    best_params: dict[str, Any] = field(default_factory=dict)
    best_score: float | None = None
    n_candidates: int = 0
    n_splits: int = 0
    search_rows: int = 0
    grid_size: int = 0
    grid_fingerprint: str = ""
    pinned_classes: list = field(default_factory=list)
    seconds: float = 0.0
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "tuned": self.tuned,
            "best_params": {k: list(v) if isinstance(v, tuple) else v
                            for k, v in self.best_params.items()},
            "best_score_f1_macro": self.best_score,
            "scoring": SCORING,
            "n_candidates": self.n_candidates,
            # Recorded next to the candidate count because one without the
            # other is what made the old runs unreadable: 20 of 144 and 20 of
            # 5,184 are the same number in a table and different experiments.
            "grid_size": self.grid_size,
            "grid_coverage": (
                round(self.n_candidates / self.grid_size, 4) if self.grid_size else None
            ),
            "grid_fingerprint": self.grid_fingerprint,
            "cv_folds": self.n_splits,
            "search_rows": self.search_rows,
            "classes_pinned_to_train": [str(c) for c in self.pinned_classes],
            "seconds": round(self.seconds, 1),
            "note": self.note,
        }


def tune_model(
    name: str,
    estimator: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_iter: int = 20,
    n_splits: int = 3,
    max_rows: int = DEFAULT_TUNE_SUBSAMPLE,
    random_state: int = RANDOM_STATE,
    class_names: list[str] | None = None,
) -> TuneResult:
    """Randomised search for ``name``; returns the winner and its provenance.

    A failure here is reported and swallowed: the caller falls back to the
    untuned defaults for that model rather than losing the other six results.
    """
    if name in UNTUNED or name not in SEARCH_SPACES:
        return TuneResult(
            model=name,
            tuned=False,
            note="not searched; inherits tuned base-learner parameters"
            if name in UNTUNED
            else "no search space defined",
        )

    space = SEARCH_SPACES[name]
    # Asking for more candidates than the space holds makes RandomizedSearchCV
    # sample with replacement and waste fits on duplicates.
    candidates = min(n_iter, space_size(space))

    X_s, y_s, rows = stratified_subsample(
        X_train, y_train, max_rows=max_rows, random_state=random_state
    )
    cv = RareAwareStratifiedKFold(n_splits=n_splits, random_state=random_state)
    # Recorded as names, not encoder codes. This field is the honest caveat on
    # the search score -- "SQL Injection was never validated" is a sentence a
    # reader can check; "class 13 was never validated" is not.
    codes = cv.pinned_classes(y_s)
    pinned = [class_names[int(c)] if class_names else c for c in codes]

    logger.info(
        "  tuning %s: %d/%d config(s) = %.1f%% of grid x %d fold(s) on %s rows%s",
        name,
        candidates,
        space_size(space),
        candidates / space_size(space) * 100,
        n_splits,
        f"{rows:,}",
        f" ({len(pinned)} class(es) pinned to train)" if pinned else "",
    )

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=space,
        n_iter=candidates,
        scoring=SCORING,
        cv=cv,
        # The estimators already parallelise internally with n_jobs=-1;
        # parallelising the search on top oversubscribes the CPU and is
        # reliably slower than running candidates one at a time.
        n_jobs=1,
        refit=False,
        random_state=random_state,
        error_score=np.nan,
        verbose=0,
    )

    t0 = time.perf_counter()
    try:
        search.fit(X_s, y_s)
    except Exception:
        elapsed = time.perf_counter() - t0
        logger.exception("  tuning %s failed -- falling back to untuned defaults", name)
        return TuneResult(
            model=name, tuned=False, seconds=elapsed,
            note="search raised; untuned defaults used",
        )
    elapsed = time.perf_counter() - t0

    best_score = float(search.best_score_) if np.isfinite(search.best_score_) else None
    if best_score is None:
        logger.warning("  %s: every candidate errored -- using untuned defaults", name)
        return TuneResult(
            model=name, tuned=False, seconds=elapsed,
            note="all candidates errored; untuned defaults used",
        )

    logger.info(
        "  tuned %s: %s=%.4f in %.1f min | %s",
        name, SCORING, best_score, elapsed / 60, search.best_params_,
    )
    return TuneResult(
        model=name,
        tuned=True,
        best_params=dict(search.best_params_),
        best_score=best_score,
        n_candidates=candidates,
        n_splits=n_splits,
        search_rows=rows,
        grid_size=space_size(space),
        grid_fingerprint=grid_fingerprint(name),
        pinned_classes=pinned,
        seconds=elapsed,
    )
