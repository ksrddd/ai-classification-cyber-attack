"""Hyperparameter tuning driver.

Reads the per-model ``grid`` block from ``config.yaml`` and runs either
``GridSearchCV`` or ``RandomizedSearchCV`` over the (scaler + clf)
Pipeline. The chosen strategy and CV settings come from the top-level
``tuning:`` and ``cv:`` sections.

Outputs are returned in a ``TuneResult`` dataclass; the caller decides
where to persist them (the training pipeline writes them to
``results/metrics/`` and ``models/``).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline

from src.config.constants import RANDOM_STATE

logger = logging.getLogger(__name__)


@dataclass
class TuneResult:
    """Output of a single model's hyperparameter search."""

    model_name: str
    strategy: str
    best_estimator: Pipeline
    best_params: dict[str, Any]
    best_score: float
    scoring: str
    cv_results: pd.DataFrame = field(repr=False)
    n_candidates: int = 0
    duration_seconds: float = 0.0


def tune_model(
    model_name: str,
    pipeline: Pipeline,
    param_grid: dict[str, list[Any]],
    X,
    y,
    *,
    strategy: str = "grid",
    n_iter: int = 20,
    cv_splits: int = 5,
    scoring: str = "f1_weighted",
    n_jobs: int = -1,
    verbose: int = 1,
) -> TuneResult:
    """Run hyperparameter search and return a populated ``TuneResult``.

    Parameters
    ----------
    model_name
        Used for logging + return value -- no functional effect.
    pipeline
        The (scaler + clf) Pipeline; ``param_grid`` keys must use the
        ``clf__`` prefix.
    param_grid
        Mapping of parameter name to candidate list.
    strategy
        ``"grid"`` -> exhaustive ``GridSearchCV``.
        ``"random"`` -> ``RandomizedSearchCV`` with ``n_iter`` samples.
    cv_splits
        K for ``StratifiedKFold``.
    scoring
        sklearn scoring metric, e.g. ``"f1_weighted"`` or ``"roc_auc_ovr"``.
    """
    if not param_grid:
        logger.warning(
            "Model %s has an empty grid; skipping tuning (returning the unfit pipeline).",
            model_name,
        )
        pipeline.fit(X, y)
        return TuneResult(
            model_name=model_name,
            strategy="none",
            best_estimator=pipeline,
            best_params={},
            best_score=float("nan"),
            scoring=scoring,
            cv_results=pd.DataFrame(),
            n_candidates=0,
            duration_seconds=0.0,
        )

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    search = _build_search(
        estimator=pipeline,
        param_grid=param_grid,
        strategy=strategy,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    logger.info(
        "Tuning %s via %s (cv=%d, scoring=%s)",
        model_name, strategy, cv_splits, scoring,
    )
    t0 = time.time()
    search.fit(X, y)
    duration = time.time() - t0

    cv_df = pd.DataFrame(search.cv_results_)
    result = TuneResult(
        model_name=model_name,
        strategy=strategy,
        best_estimator=search.best_estimator_,
        best_params=dict(search.best_params_),
        best_score=float(search.best_score_),
        scoring=scoring,
        cv_results=cv_df,
        n_candidates=len(cv_df),
        duration_seconds=duration,
    )
    logger.info(
        "Tuned %s: best_score=%.4f params=%s (%.1fs over %d candidates)",
        model_name, result.best_score, result.best_params,
        duration, result.n_candidates,
    )
    return result


def _build_search(
    *,
    estimator: BaseEstimator,
    param_grid: dict[str, list[Any]],
    strategy: str,
    n_iter: int,
    cv: StratifiedKFold,
    scoring: str,
    n_jobs: int,
    verbose: int,
):
    common = dict(
        estimator=estimator,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
        return_train_score=False,
    )
    if strategy == "grid":
        return GridSearchCV(param_grid=param_grid, **common)
    if strategy == "random":
        # RandomizedSearchCV expects param_distributions, not param_grid.
        return RandomizedSearchCV(
            param_distributions=param_grid,
            n_iter=min(n_iter, _grid_size(param_grid)),
            random_state=RANDOM_STATE,
            **common,
        )
    raise ValueError(
        f"Unknown tuning strategy {strategy!r}. Use 'grid' or 'random'."
    )


def _grid_size(grid: dict[str, list[Any]]) -> int:
    """Number of distinct combos in ``grid`` (for capping random search)."""
    size = 1
    for v in grid.values():
        size *= max(len(v), 1)
    return max(size, 1)


def cv_results_summary(cv_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Trim ``cv_results_`` to the columns worth reporting."""
    if cv_df.empty:
        return cv_df
    cols = [c for c in cv_df.columns if c.startswith("param_") or c in
            ("mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time")]
    out = cv_df.loc[:, cols].copy()
    if "rank_test_score" in out.columns:
        out = out.sort_values("rank_test_score").head(top_n)
    out = out.reset_index(drop=True)
    # numpy floats -> python floats for clean JSON / Markdown serialisation
    for c in out.columns:
        if np.issubdtype(out[c].dtype, np.floating):
            out[c] = out[c].astype(float)
    return out
