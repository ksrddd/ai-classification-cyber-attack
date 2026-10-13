"""SHAP analysis driver.

For tree-based models (RF / XGBoost / LightGBM / CatBoost) we use
``shap.TreeExplainer`` -- exact, fast.

For MLPClassifier we fall back to ``shap.KernelExplainer`` with
configurably small background + analysis sets. KernelExplainer is slow
(O(2^n_features)) but works model-agnostically. With ``linear_explainer``
a LogisticRegression gets the exact ``shap.LinearExplainer`` instead.

Given labels ``y`` and ``per_class_samples``, rows are sampled per class
and each class's importance is averaged over that class's own rows only,
so a rare class is explained on its own flows rather than on Benign ones.

Output artefacts (under ``results/shap/<model>/``):
- ``summary.png``      -- beeswarm-style summary plot.
- ``summary_bar.png``  -- mean |SHAP value| bar plot.
- ``top_features.json``-- top-K features overall + per-class.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.config.constants import SHAP_DIR
from src.features.pipeline import is_tree_based
from src.utils.io import ensure_dir

logger = logging.getLogger(__name__)

_TREE_EXPLAINER = "tree"
_KERNEL_EXPLAINER = "kernel"
_LINEAR_EXPLAINER = "linear"

#: What one SHAP unit means per explainer: margin-space tree models (XGBoost,
#: LightGBM, CatBoost) explain the log-odds, RandomForest and KernelExplainer on
#: predict_proba explain probability, LinearExplainer explains the per-class
#: logit. Raw magnitudes are only comparable within one unit; shares are unit-free.
_MARGIN_TREES = {"XGBClassifier", "LGBMClassifier", "CatBoostClassifier"}


@dataclass
class ShapResult:
    """SHAP outputs for one model."""

    model_name: str
    explainer_kind: str
    feature_names: list[str]
    class_names: list[str]
    shap_values: np.ndarray | list[np.ndarray]
    top_features_overall: list[tuple[str, float]]
    top_features_per_class: dict[str, list[tuple[str, float]]]
    artefacts: dict[str, Path]
    # Full-length mean |SHAP| vectors, for aggregating across seeds.
    importance_overall: np.ndarray | None = None
    importance_per_class: dict[str, np.ndarray] | None = None
    # Rows each class's importance was averaged over.
    per_class_n: dict[str, int] | None = None
    units: str | None = None


def analyze_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    class_names: Iterable[str],
    *,
    model_name: str = "model",
    background_samples: int = 200,
    analysis_samples: int = 1000,
    top_k: int = 10,
    save_dir: Path | None = None,
    random_state: int = 42,
    y: np.ndarray | None = None,
    per_class_samples: int | None = None,
    kernel_nsamples: int | str = 100,
    kernel_l1_reg: str | bool | None = None,
    background_kmeans: int | None = None,
    linear_explainer: bool = False,
    write_plots: bool = True,
    write_json: bool = True,
    n_jobs: int = 1,
) -> ShapResult:
    """Compute SHAP values + write summary plots and a JSON report.

    The pipeline's scaler is applied to ``X`` first; SHAP sees the same
    feature space the classifier sees, which keeps feature names but
    scales values.

    ``y`` (encoded labels aligned with ``X``) plus ``per_class_samples``
    switches to class-stratified sampling: up to that many rows per class,
    every row of a smaller class. Per-class importance is then averaged over
    the class's own rows, and the overall ranking is class-balanced.
    ``kernel_nsamples="auto"`` and ``background_kmeans`` make KernelExplainer
    estimates less noisy; the defaults keep the original behaviour.
    """
    classifier = pipeline.named_steps["clf"]
    scaler = pipeline.named_steps.get("scaler")

    rng = np.random.default_rng(random_state)
    feature_names = list(X.columns)
    classes = [str(c) for c in class_names]

    # Sample analysis + background sets.
    stratified = y is not None and per_class_samples is not None
    if stratified:
        y = np.asarray(y)
        analyze_idx = np.concatenate([
            rng.choice(rows, size=min(per_class_samples, len(rows)), replace=False)
            for rows in (np.flatnonzero(y == ci) for ci in range(len(classes)))
            if len(rows)
        ])
        y_analyze = y[analyze_idx]
    else:
        analyze_idx = rng.choice(len(X), size=min(analysis_samples, len(X)), replace=False)
        y_analyze = None
    n_analyze = len(analyze_idx)
    X_analyze = X.iloc[analyze_idx]
    X_a_scaled = pd.DataFrame(
        scaler.transform(X_analyze) if scaler is not None else X_analyze.values,
        columns=feature_names,
    )

    is_tree = is_tree_based(classifier)
    is_linear = (
        linear_explainer and not is_tree
        and classifier.__class__.__name__ == "LogisticRegression"
    )
    explainer_kind = (
        _TREE_EXPLAINER if is_tree else _LINEAR_EXPLAINER if is_linear else _KERNEL_EXPLAINER
    )
    logger.info(
        "SHAP for %s using %s (analyze=%d, background=%d)",
        model_name, explainer_kind, n_analyze, background_samples,
    )

    if is_tree:
        # shap.TreeExplainer + CatBoost can segfault on Windows with multiclass
        # models (catboost issue #2474). Use CatBoost's native ShapValues API
        # instead -- returns (n_samples, n_classes, n_features+1) where the
        # last column on axis 2 is the bias term.
        if classifier.__class__.__name__ == "CatBoostClassifier":
            from catboost import Pool
            raw = classifier.get_feature_importance(
                Pool(X_a_scaled), type="ShapValues",
            )
            shap_values = np.transpose(raw[:, :, :-1], (1, 0, 2))
        elif n_jobs > 1:
            # Each row's attribution is independent, so row chunks run in
            # parallel and concatenate to exactly the serial result.
            from joblib import Parallel, delayed
            chunks = np.array_split(np.arange(n_analyze), n_jobs)
            parts = Parallel(n_jobs=n_jobs)(
                delayed(_tree_shap)(classifier, X_a_scaled.iloc[c]) for c in chunks if len(c)
            )
            shap_values = np.concatenate(parts, axis=0)
        else:
            shap_values = _tree_shap(classifier, X_a_scaled)
    else:
        n_bg = min(background_samples, len(X))
        bg_idx = rng.choice(len(X), size=n_bg, replace=False)
        X_bg = X.iloc[bg_idx]
        X_bg_scaled = scaler.transform(X_bg) if scaler is not None else X_bg.values
        if is_linear:
            explainer = shap.LinearExplainer(classifier, X_bg_scaled)
            shap_values = explainer.shap_values(X_a_scaled.values)
        else:
            background = (
                shap.kmeans(X_bg_scaled, background_kmeans) if background_kmeans else X_bg_scaled
            )
            explainer = shap.KernelExplainer(classifier.predict_proba, background)
            # shap defaults l1_reg to "num_features(10)": every row keeps only 10
            # non-zero features, so rankings past the top few are an artefact.
            # Pass False for full estimates; None keeps shap's default.
            extra = {} if kernel_l1_reg is None else {"l1_reg": kernel_l1_reg}
            shap_values = explainer.shap_values(
                X_a_scaled.values, nsamples=kernel_nsamples, **extra
            )

    # Normalize shap_values shape so the rest is uniform across explainers.
    sv = _normalize_shap_shape(shap_values, n_classes=len(list(class_names)))

    # Top-K features.
    imp_overall = np.mean(np.abs(sv), axis=(0, 1))
    imp_per_class, per_class_n = _class_importance(sv, classes, y_analyze)
    overall = _rank(imp_overall, feature_names, top_k)
    per_class = {c: _rank(v, feature_names, top_k) for c, v in imp_per_class.items()}
    units = (
        "logit" if is_linear
        else "log-odds" if classifier.__class__.__name__ in _MARGIN_TREES
        else "probability"
    )

    # Plots + JSON.
    save_dir = save_dir or (SHAP_DIR / model_name)
    ensure_dir(save_dir)
    plots = (
        _write_plots(sv, X_a_scaled, classes, save_dir, model_name, y_analyze)
        if write_plots else {}
    )

    if write_json:
        top_path = save_dir / "top_features.json"
        payload = {
            "overall": [[name, float(v)] for name, v in overall],
            "per_class": {
                cls: [[name, float(v)] for name, v in items]
                for cls, items in per_class.items()
            },
        }
        if stratified:
            payload.update(
                overall_share=[
                    [n, float(v)] for n, v in _rank(_share(imp_overall), feature_names, top_k)
                ],
                per_class_share={
                    c: [[n, float(x)] for n, x in _rank(_share(v), feature_names, top_k)]
                    for c, v in imp_per_class.items()
                },
                per_class_n=per_class_n,
                units=units,
                explainer=explainer_kind,
            )
        top_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        plots["top_features_json"] = top_path

    return ShapResult(
        model_name=model_name,
        explainer_kind=explainer_kind,
        feature_names=feature_names,
        class_names=list(class_names),
        shap_values=shap_values,
        top_features_overall=overall,
        top_features_per_class=per_class,
        artefacts=plots,
        importance_overall=imp_overall,
        importance_per_class=imp_per_class,
        per_class_n=per_class_n,
        units=units,
    )


def write_shap_report(
    results: dict[str, ShapResult],
    save_to: Path | None = None,
) -> Path:
    """Cross-model SHAP markdown summary."""
    save_to = save_to or (SHAP_DIR / "shap_report.md")
    ensure_dir(save_to.parent)

    lines = ["# SHAP Explainability Report", ""]
    for name, r in results.items():
        lines.append(f"## {name} (explainer: {r.explainer_kind})")
        lines.append("")
        lines.append("### Top features overall")
        for feat, val in r.top_features_overall[:10]:
            lines.append(f"- {feat}: {val:.4f}")
        lines.append("")
        lines.append("### Top features per class")
        for cls, items in r.top_features_per_class.items():
            top3 = ", ".join(f"{f} ({v:.3f})" for f, v in items[:3])
            lines.append(f"- **{cls}**: {top3}")
        lines.append("")
    save_to.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote SHAP report -> %s", save_to)
    return save_to


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _normalize_shap_shape(shap_values, n_classes: int) -> np.ndarray:
    """Return an array of shape ``(n_classes, n_samples, n_features)``.

    SHAP's return shape varies by explainer + model:
    - Binary tree models -> (n_samples, n_features)
    - Multi-class tree models -> list of (n_samples, n_features) per class,
      OR a 3-D array (n_samples, n_features, n_classes) in newer SHAP.
    - KernelExplainer -> list of arrays per class.
    """
    if isinstance(shap_values, list):
        return np.stack(shap_values, axis=0)
    arr = np.asarray(shap_values)
    if arr.ndim == 2:
        # Binary, or a regressor -- treat as single class.
        return arr[np.newaxis, ...]
    if arr.ndim == 3:
        # (n_samples, n_features, n_classes) -> (n_classes, n_samples, n_features)
        if arr.shape[-1] == n_classes:
            return np.transpose(arr, (2, 0, 1))
        # Already (n_classes, n_samples, n_features).
        if arr.shape[0] == n_classes:
            return arr
    raise ValueError(f"Unexpected SHAP value shape: {arr.shape}")


def _tree_shap(classifier, X: pd.DataFrame) -> np.ndarray:
    """TreeExplainer values as one (n_samples, n_features, n_classes) array."""
    values = shap.TreeExplainer(classifier).shap_values(X)
    return np.stack(values, axis=-1) if isinstance(values, list) else np.asarray(values)


def _rank(importance: np.ndarray, feature_names: list[str], k: int) -> list[tuple[str, float]]:
    order = np.argsort(importance)[::-1][:k]
    return [(feature_names[i], float(importance[i])) for i in order]


def _share(importance: np.ndarray) -> np.ndarray:
    """Mean |SHAP| as a fraction of the model's total, so units cancel."""
    total = importance.sum()
    return importance / total if total > 0 else importance


def _class_importance(
    sv: np.ndarray,
    classes: list[str],
    y_analyze: np.ndarray | None,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Mean |SHAP| toward each class's score.

    Without labels it averages over every analysed row. With labels it uses
    only the class's own rows; a class with none in the sample is left out
    rather than explained on other classes' rows.
    """
    out: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for ci, cls in enumerate(classes):
        if ci >= sv.shape[0]:
            break
        if y_analyze is None:
            out[cls] = np.mean(np.abs(sv[ci]), axis=0)
            counts[cls] = sv.shape[1]
            continue
        mask = y_analyze == ci
        if mask.any():
            out[cls] = np.mean(np.abs(sv[ci][mask]), axis=0)
            counts[cls] = int(mask.sum())
    return out, counts


def _write_plots(
    sv: np.ndarray,
    X_scaled: pd.DataFrame,
    class_names: list[str],
    save_dir: Path,
    model_name: str,
    y_analyze: np.ndarray | None = None,
) -> dict[str, Path]:
    """Write summary + bar plots. Returns {name: path}.

    With labels, each class's beeswarm shows only that class's rows.
    """
    from src.visualization.plots import set_style
    set_style()

    paths: dict[str, Path] = {}

    # Per-class beeswarm for the most common explanation.
    for ci, cls in enumerate(class_names):
        if ci >= sv.shape[0]:
            break
        mask = np.ones(sv.shape[1], bool) if y_analyze is None else y_analyze == ci
        if not mask.any():
            continue
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            sv[ci][mask], X_scaled[mask], feature_names=list(X_scaled.columns),
            show=False, max_display=15,
        )
        plt.title(f"SHAP summary -- {model_name} / {cls}")
        path = save_dir / f"summary_{_slug(cls)}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        paths[f"summary_{cls}"] = path

    # Mean |shap| bar across classes -> single overall importance plot.
    importance = np.mean(np.abs(sv), axis=(0, 1))
    order = np.argsort(importance)[::-1][:20]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [X_scaled.columns[i] for i in order][::-1],
        importance[order][::-1],
        color="steelblue",
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top features by mean |SHAP| -- {model_name}")
    bar_path = save_dir / "summary_bar.png"
    plt.tight_layout()
    fig.savefig(bar_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths["summary_bar"] = bar_path

    return paths


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_") or "class"
