"""Feature selection -- four methods compared.

Methods:
  1. Correlation analysis -- drop one of any pair above threshold.
  2. Random Forest feature importance -- top-K by mean decrease in impurity.
  3. SelectKBest -- ANOVA F-statistic for classification.
  4. Recursive Feature Elimination -- RFE wrapping a base estimator.

Output: ``reports/feature_selection_report.md`` comparing the methods'
agreement (Jaccard overlap matrix) plus the union/intersection sets.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

from src.config.constants import RANDOM_STATE, REPORTS_DIR
from src.utils.io import ensure_dir

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    """Output of a single feature-selection method."""

    method: str
    selected: list[str]
    scores: dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.selected)


# ---------------------------------------------------------------------------
# Individual methods
# ---------------------------------------------------------------------------


def select_by_correlation(
    X: pd.DataFrame,
    threshold: float = 0.95,
) -> SelectionResult:
    """Drop one column of any pair correlated above ``threshold``.

    Walks columns left-to-right; the *first* occurrence is kept and any
    later column too correlated with an already-kept column is dropped.
    Deterministic given a stable column order.
    """
    numeric = X.select_dtypes(include=np.number)
    corr = numeric.corr().abs()
    kept: list[str] = []
    dropped: list[str] = []
    for col in numeric.columns:
        if any(corr.loc[col, k] > threshold for k in kept):
            dropped.append(col)
        else:
            kept.append(col)
    logger.info(
        "Correlation filter (>%.2f): kept %d / %d (dropped %d)",
        threshold, len(kept), numeric.shape[1], len(dropped),
    )
    return SelectionResult(method="correlation", selected=kept)


def select_by_rf_importance(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    top_k: int = 30,
    n_estimators: int = 200,
) -> SelectionResult:
    """Top-K columns by Random Forest mean-decrease-in-impurity importance.

    Trained on the full ``(X, y)`` -- this is a *selection* step, separate
    from model training, so it's allowed to peek at all rows.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    selected = importances.head(top_k).index.tolist()
    logger.info("RF importance: top-%d features selected", top_k)
    return SelectionResult(
        method="random_forest_importance",
        selected=selected,
        scores=importances.to_dict(),
    )


def select_by_k_best(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    k: int = 30,
) -> SelectionResult:
    """Top-K columns by ANOVA F-statistic (univariate, fast).

    Scales features first because F-statistic is scale-sensitive in practice
    (large-variance columns can dominate).
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    skb = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    skb.fit(Xs, y)
    mask = skb.get_support()
    selected = X.columns[mask].tolist()
    scores = pd.Series(skb.scores_, index=X.columns).to_dict()
    logger.info("SelectKBest: %d features selected (k=%d)", len(selected), k)
    return SelectionResult(method="select_k_best", selected=selected, scores=scores)


def select_by_rfe(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    n_features: int = 25,
    step: int = 5,
) -> SelectionResult:
    """RFE around a lightweight Random Forest base estimator.

    Uses a small RF (50 trees, depth 10) to keep this affordable on the
    full CICIDS corpus -- a full-depth RFE is impractical here.
    """
    base = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rfe = RFE(estimator=base, n_features_to_select=n_features, step=step)
    rfe.fit(X, y)
    selected = X.columns[rfe.support_].tolist()
    ranks = pd.Series(rfe.ranking_, index=X.columns).to_dict()
    logger.info("RFE: %d features selected (n_features=%d)", len(selected), n_features)
    return SelectionResult(method="rfe", selected=selected, scores=ranks)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_all_methods(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    methods: Iterable[str],
    k_best: int = 30,
    rfe_n_features: int = 25,
    correlation_threshold: float = 0.95,
) -> dict[str, SelectionResult]:
    """Run every requested method, return results keyed by method name."""
    results: dict[str, SelectionResult] = {}
    methods = list(methods)
    if "correlation" in methods:
        results["correlation"] = select_by_correlation(X, correlation_threshold)
    if "random_forest_importance" in methods:
        results["random_forest_importance"] = select_by_rf_importance(X, y, top_k=k_best)
    if "select_k_best" in methods:
        results["select_k_best"] = select_by_k_best(X, y, k=k_best)
    if "rfe" in methods:
        results["rfe"] = select_by_rfe(X, y, n_features=rfe_n_features)
    return results


def compare_methods(results: dict[str, SelectionResult]) -> pd.DataFrame:
    """Jaccard-overlap matrix between every pair of methods."""
    names = list(results.keys())
    sets = {n: set(results[n].selected) for n in names}
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            inter = sets[a] & sets[b]
            union = sets[a] | sets[b]
            mat.loc[a, b] = len(inter) / max(len(union), 1)
    return mat


def consensus_features(
    results: dict[str, SelectionResult],
    min_methods: int = 2,
) -> list[str]:
    """Features selected by at least ``min_methods`` of the methods."""
    counts: dict[str, int] = {}
    for r in results.values():
        for f in r.selected:
            counts[f] = counts.get(f, 0) + 1
    return sorted(f for f, c in counts.items() if c >= min_methods)


def write_report(
    results: dict[str, SelectionResult],
    save_to: Path | None = None,
) -> Path:
    """Write a Markdown summary comparing the methods."""
    save_to = save_to or (REPORTS_DIR / "feature_selection_report.md")
    ensure_dir(save_to.parent)

    overlap = compare_methods(results)
    consensus = consensus_features(results, min_methods=2)

    lines = ["# Feature Selection Report", ""]
    lines.append("## Selected count per method")
    for name, r in results.items():
        lines.append(f"- **{name}**: {len(r)} features")
    lines.append("")
    lines.append("## Jaccard overlap matrix")
    lines.append("")
    lines.append(overlap.round(3).to_markdown())
    lines.append("")
    lines.append(f"## Consensus features (>= 2 methods, {len(consensus)} total)")
    for f in consensus:
        lines.append(f"- {f}")
    lines.append("")
    save_to.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote feature selection report -> %s", save_to)
    return save_to
