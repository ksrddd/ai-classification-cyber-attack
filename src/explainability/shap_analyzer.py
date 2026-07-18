"""SHAP analysis driver.

For tree-based models (RF / XGBoost / LightGBM / CatBoost) we use
``shap.TreeExplainer`` -- exact, fast.

For MLPClassifier we fall back to ``shap.KernelExplainer`` with
configurably small background + analysis sets. KernelExplainer is slow
(O(2^n_features)) but works model-agnostically.

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
) -> ShapResult:
    """Compute SHAP values + write summary plots and a JSON report.

    The pipeline's scaler is applied to ``X`` first; SHAP sees the same
    feature space the classifier sees, which keeps feature names but
    scales values.
    """
    classifier = pipeline.named_steps["clf"]
    scaler = pipeline.named_steps.get("scaler")

    rng = np.random.default_rng(random_state)
    feature_names = list(X.columns)

    # Sample analysis + background sets.
    n_analyze = min(analysis_samples, len(X))
    analyze_idx = rng.choice(len(X), size=n_analyze, replace=False)
    X_analyze = X.iloc[analyze_idx]
    X_a_scaled = pd.DataFrame(
        scaler.transform(X_analyze) if scaler is not None else X_analyze.values,
        columns=feature_names,
    )

    is_tree = is_tree_based(classifier)
    explainer_kind = _TREE_EXPLAINER if is_tree else _KERNEL_EXPLAINER
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
        else:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_a_scaled)
    else:
        n_bg = min(background_samples, len(X))
        bg_idx = rng.choice(len(X), size=n_bg, replace=False)
        X_bg = X.iloc[bg_idx]
        X_bg_scaled = scaler.transform(X_bg) if scaler is not None else X_bg.values
        predict_fn = classifier.predict_proba
        explainer = shap.KernelExplainer(predict_fn, X_bg_scaled)
        shap_values = explainer.shap_values(X_a_scaled.values, nsamples=100)

    # Normalize shap_values shape so the rest is uniform across explainers.
    sv = _normalize_shap_shape(shap_values, n_classes=len(list(class_names)))

    # Top-K features.
    overall = _top_features_overall(sv, feature_names, k=top_k)
    per_class = _top_features_per_class(sv, feature_names, class_names, k=top_k)

    # Plots + JSON.
    save_dir = save_dir or (SHAP_DIR / model_name)
    ensure_dir(save_dir)
    plots = _write_plots(sv, X_a_scaled, list(class_names), save_dir, model_name)

    top_path = save_dir / "top_features.json"
    with top_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "overall": [[name, float(v)] for name, v in overall],
                "per_class": {
                    cls: [[name, float(v)] for name, v in items]
                    for cls, items in per_class.items()
                },
            },
            f, indent=2,
        )
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


def _top_features_overall(
    sv: np.ndarray,
    feature_names: list[str],
    k: int,
) -> list[tuple[str, float]]:
    importance = np.mean(np.abs(sv), axis=(0, 1))  # average over classes + samples
    order = np.argsort(importance)[::-1][:k]
    return [(feature_names[i], float(importance[i])) for i in order]


def _top_features_per_class(
    sv: np.ndarray,
    feature_names: list[str],
    class_names: Iterable[str],
    k: int,
) -> dict[str, list[tuple[str, float]]]:
    classes = list(class_names)
    out: dict[str, list[tuple[str, float]]] = {}
    for ci, cls in enumerate(classes):
        if ci >= sv.shape[0]:
            break
        importance = np.mean(np.abs(sv[ci]), axis=0)
        order = np.argsort(importance)[::-1][:k]
        out[str(cls)] = [(feature_names[i], float(importance[i])) for i in order]
    return out


def _write_plots(
    sv: np.ndarray,
    X_scaled: pd.DataFrame,
    class_names: list[str],
    save_dir: Path,
    model_name: str,
) -> dict[str, Path]:
    """Write summary + bar plots. Returns {name: path}."""
    from src.visualization.plots import set_style
    set_style()

    paths: dict[str, Path] = {}

    # Per-class beeswarm for the most common explanation.
    for ci, cls in enumerate(class_names):
        if ci >= sv.shape[0]:
            break
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            sv[ci], X_scaled, feature_names=list(X_scaled.columns),
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
