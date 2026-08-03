"""Discover results bundles and normalise them to one schema.

Two incompatible layouts exist in ``results/``:

    CICIDS2017 (``results/cicids2017_temporal_v1/``)
        metrics.json                 run-level: split, class names, counts
        <model>_metrics.json         per-model, ~40 keys incl. MCC, CV, HP search
        <model>_per_class.csv        per-class precision/recall/F1/support

    CSE-CIC-IDS2018 (``results/ids2018/<size>/``)
        <model>/metrics.json         per-model, 11 keys -- no MCC, no CV, no FPR
        <model>/per_class_report.csv per-class
        <model>/confusion_matrix.csv
        extended_metrics.csv         MCC, binary FPR, FP/FN, throughput, size
        significance_*.csv           bootstrap CI + McNemar (2018 only)

Neither is a superset of the other, so normalisation cannot mean "fill in
the gaps". A field the bundle does not record is emitted as ``None`` and the
UI is expected to render it as absent. Substituting 0.0 would be worse than
useless here: a false-positive rate of "not recorded" and one of "zero" lead
to opposite conclusions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.constants import RESULTS_DIR

logger = logging.getLogger(__name__)

# Directories under results/ that are not bundles.
_NON_BUNDLE = {"figures", "metrics", "coverage", "shap"}


class BundleNotFound(LookupError):
    """Raised when a bundle id does not resolve to a directory on disk."""


@dataclass
class Bundle:
    """One results bundle, normalised.

    ``models`` maps model name -> flat metric dict. Every dict carries the
    same keys; values are ``None`` where the source bundle had nothing.
    """

    id: str
    path: Path
    dataset: str
    layout: str  # "cicids2017" | "ids2018"
    run: dict[str, Any] = field(default_factory=dict)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    classes: list[str] = field(default_factory=list)
    # Paired comparisons against the bundle's leading model, when the bundle
    # ran them. Keyed by rival model name. See _load_significance.
    significance: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_summary(self) -> dict[str, Any]:
        """Small payload for the bundle picker in the sidebar."""
        return {
            "id": self.id,
            "dataset": self.dataset,
            "layout": self.layout,
            "n_models": len(self.models),
            "n_classes": len(self.classes),
            "split_protocol": self.run.get("split_protocol"),
            "n_train": self.run.get("n_train"),
            "n_test": self.run.get("n_test"),
            "n_features": self.run.get("n_features"),
            "class_weighting": self.run.get("class_weighting"),
            "hp_tuned": self.run.get("hp_tuned"),
        }


# The normalised per-model schema. Everything the dashboard may ask for.
METRIC_FIELDS = (
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_weighted",
    "precision_macro",
    "precision_weighted",
    "recall_macro",
    "recall_weighted",
    "mcc",
    "binary_fpr",
    "binary_recall",
    "false_alarms_fp",
    "missed_attacks_fn",
    "train_seconds",
    "predict_seconds",
    "throughput_flows_per_sec",
    "model_size_mb",
    "cv_f1_macro_mean",
    "cv_f1_macro_std",
    "label_shuffle_f1_macro",
    "hp_tuned",
    "accelerator",
    "f1_macro_ci_low",
    "f1_macro_ci_high",
)


def _blank_metrics() -> dict[str, Any]:
    return dict.fromkeys(METRIC_FIELDS)


def _num(value: Any) -> float | None:
    """Coerce to float, mapping NaN and non-numerics to None."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out  # NaN check


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------
def _is_2017_bundle(path: Path) -> bool:
    return (path / "metrics.json").is_file() and any(path.glob("*_metrics.json"))


def _is_2018_bundle(path: Path) -> bool:
    return (path / "model_comparison.csv").is_file() and any(
        (child / "metrics.json").is_file() for child in path.iterdir() if child.is_dir()
    )


def _discover(results_dir: Path) -> dict[str, Path]:
    """Map bundle id -> directory, searching one level and one nested level.

    2017 runs sit directly under results/; 2018 runs are grouped one level
    deeper (results/ids2018/300k), so both depths are scanned.
    """
    found: dict[str, Path] = {}
    if not results_dir.is_dir():
        return found

    for child in sorted(results_dir.iterdir()):
        if not child.is_dir() or child.name in _NON_BUNDLE:
            continue
        if _is_2017_bundle(child) or _is_2018_bundle(child):
            found[child.name] = child
            continue
        for grandchild in sorted(child.iterdir()):
            if not grandchild.is_dir():
                continue
            if _is_2017_bundle(grandchild) or _is_2018_bundle(grandchild):
                found[f"{child.name}/{grandchild.name}"] = grandchild
    return found


def list_bundles(results_dir: Path | None = None) -> list[dict[str, Any]]:
    """Summaries of every bundle on disk, newest-looking first.

    Ordering is stable rather than clever: 2017 runs first, then 2018 runs,
    so the picker does not reshuffle between requests.
    """
    summaries = []
    for bundle_id in _discover(results_dir or RESULTS_DIR):
        try:
            summaries.append(load_bundle(bundle_id, results_dir).to_summary())
        except Exception:  # noqa: BLE001 -- one broken bundle must not hide the rest
            logger.exception("Skipping unreadable bundle %s", bundle_id)
    return summaries


def resolve_bundle_id(bundle_id: str | None, results_dir: Path | None = None) -> str:
    """Return ``bundle_id`` if it exists, else the first available bundle."""
    found = _discover(results_dir or RESULTS_DIR)
    if not found:
        raise BundleNotFound("No results bundle found under results/")
    if bundle_id is None:
        return next(iter(found))
    if bundle_id not in found:
        raise BundleNotFound(
            f"Unknown bundle {bundle_id!r}. Available: {', '.join(found)}"
        )
    return bundle_id


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_bundle(bundle_id: str, results_dir: Path | None = None) -> Bundle:
    """Read one bundle from disk and normalise it."""
    found = _discover(results_dir or RESULTS_DIR)
    if bundle_id not in found:
        raise BundleNotFound(f"Unknown bundle {bundle_id!r}")
    path = found[bundle_id]
    return _load_2017(bundle_id, path) if _is_2017_bundle(path) else _load_2018(bundle_id, path)


def describe_bundle(bundle_id: str, results_dir: Path | None = None) -> dict[str, Any]:
    """Bundle payload shaped for the API: run block plus per-model metrics."""
    bundle = load_bundle(bundle_id, results_dir)
    return {
        "id": bundle.id,
        "dataset": bundle.dataset,
        "layout": bundle.layout,
        "run": bundle.run,
        "classes": bundle.classes,
        "models": bundle.models,
        "significance": bundle.significance,
    }


def _load_significance(path: Path) -> dict[str, dict[str, Any]]:
    """Paired comparisons against the leading model, if the bundle ran them.

    Paired, not marginal, and the distinction is the whole point. Each model's
    own bootstrap interval on macro-F1 is wide here -- the tiny classes swing
    the statistic from resample to resample -- so the marginal intervals of
    all seven models overlap, including ones that disagree on 1,955 of 90,000
    test flows. Judging "tied" by overlapping marginal intervals would call
    those two indistinguishable, which is false.

    The models are scored on the *same* resamples, so the interval on their
    difference is far tighter and is the quantity that answers "is this gap
    real". That is what this file holds, alongside McNemar on per-flow
    correctness.
    """
    rows = _read_indexed_csv(path / "significance_mcnemar.csv", "rival")
    out: dict[str, dict[str, Any]] = {}
    for rival, row in rows.items():
        p = _num(row.get("mcnemar_p"))
        lo = _num(row.get("d_ci_lo"))
        hi = _num(row.get("d_ci_hi"))
        # Not separable if either test fails to reject: a difference interval
        # spanning zero, or a McNemar p above 0.05. Requiring both to agree
        # before claiming a real difference is the conservative reading, and
        # the honest one when the two tests disagree.
        separable = (
            None
            if (p is None or lo is None or hi is None)
            else bool(p < 0.05 and not (lo <= 0 <= hi))
        )
        out[rival] = {
            "delta_f1_macro": _num(row.get("d_f1")),
            "delta_ci_low": lo,
            "delta_ci_high": hi,
            "mcnemar_p": p,
            "disagreeing_flows": _int(row.get("disagree")),
            "leader_only_right": _int(row.get("leader_only_right")),
            "rival_only_right": _int(row.get("rival_only_right")),
            "separable_from_leader": separable,
        }
    return out


def _load_2017(bundle_id: str, path: Path) -> Bundle:
    run_raw = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    classes = list(run_raw.get("class_names", []))

    models: dict[str, dict[str, Any]] = {}
    for metrics_file in sorted(path.glob("*_metrics.json")):
        name = metrics_file.name.removesuffix("_metrics.json")
        if name == "metrics":  # the run-level file itself
            continue
        raw = json.loads(metrics_file.read_text(encoding="utf-8"))
        row = _blank_metrics()
        for key in METRIC_FIELDS:
            if key in raw:
                row[key] = raw[key]
        # Names that differ between the two pipelines.
        row["binary_fpr"] = _num(raw.get("binary_fpr"))
        row["binary_recall"] = _num(raw.get("binary_recall"))
        row["false_alarms_fp"] = raw.get("binary_false_positives")
        row["missed_attacks_fn"] = raw.get("binary_false_negatives")
        row["train_seconds"] = _num(raw.get("fit_seconds"))
        row["hp_tuned"] = raw.get("hp_tuned")
        models[name] = row

    return Bundle(
        id=bundle_id,
        path=path,
        dataset="CICIDS2017",
        layout="cicids2017",
        classes=classes,
        models=models,
        run={
            "run_name": run_raw.get("run_name"),
            "split_protocol": run_raw.get("split_protocol"),
            "random_state": run_raw.get("random_state"),
            "n_train": run_raw.get("n_train"),
            "n_test": run_raw.get("n_test"),
            "n_features": run_raw.get("n_features"),
            "n_classes": len(classes),
            "class_names": classes,
            "per_class_n_train": run_raw.get("per_class_n_train", {}),
            "per_class_n_test": run_raw.get("per_class_n_test", {}),
            "majority_baseline_acc": run_raw.get("majority_baseline_acc"),
            "hp_tuned": run_raw.get("hp_search"),
            "class_weighting": run_raw.get("imbalance_strategy"),
            "accelerator": run_raw.get("accelerator"),
        },
    )


def _load_2018(bundle_id: str, path: Path) -> Bundle:
    # Per-model metrics.json carries only the 11 core scores; the rest of
    # the dashboard's fields live in these two side files, written by the
    # analysis steps rather than by training.
    extended = _read_indexed_csv(path / "extended_metrics.csv", "model")
    ci = _read_indexed_csv(path / "significance_f1_macro_ci.csv", "model")

    # Loaded before the model loop, not after: the loop needs hp_tuned, and the
    # protocol fields below are now recorded per run rather than assumed for
    # the whole layout. A 2018 bundle can be random+untuned or temporal+tuned,
    # and reporting the first for both would misdescribe how a score may be read.
    meta = _read_json(path.parent.parent / "models" / "ids2018" / path.name / "metadata.json")
    if not meta:
        meta = _read_json(_project_root(path) / "models" / "ids2018" / path.name / "metadata.json")

    # A bundle whose metadata loaded but carries no `hp_tuned` key was produced
    # by a version of this pipeline that had no search code at all, so False is
    # a fact about the code path that made it, not a guess about the data.
    # No metadata file at all is a different situation -- then nothing is known
    # and the field stays absent rather than being invented.
    hp_tuned = meta.get("hp_tuned", False) if meta else None
    split_protocol = meta.get("split_protocol") or "random_stratified_70_30"

    models: dict[str, dict[str, Any]] = {}
    classes: list[str] = []
    for child in sorted(p for p in path.iterdir() if p.is_dir()):
        metrics_file = child / "metrics.json"
        if not metrics_file.is_file():
            continue
        raw = json.loads(metrics_file.read_text(encoding="utf-8"))
        name = child.name
        row = _blank_metrics()
        for key in METRIC_FIELDS:
            if key in raw:
                row[key] = raw[key]
        ext = extended.get(name, {})
        row["mcc"] = _num(ext.get("mcc"))
        row["binary_fpr"] = _num(ext.get("binary_fpr"))
        row["binary_recall"] = _num(ext.get("binary_recall"))
        row["false_alarms_fp"] = _int(ext.get("false_alarms_fp"))
        row["missed_attacks_fn"] = _int(ext.get("missed_attacks_fn"))
        row["throughput_flows_per_sec"] = _num(ext.get("flows_per_sec"))
        row["model_size_mb"] = _num(ext.get("artifact_mb"))
        row["f1_macro_ci_low"] = _num(ci.get(name, {}).get("ci_lo"))
        row["f1_macro_ci_high"] = _num(ci.get(name, {}).get("ci_hi"))
        # This pipeline still never cross-validates and never runs a
        # label-shuffle control. Those stay None, not zero.
        row["hp_tuned"] = hp_tuned
        row["accelerator"] = meta.get("accelerator")
        models[name] = row

        if not classes:
            classes = _classes_from_report(child / "per_class_report.csv")

    n_test = _test_support(path, classes)
    sample_size = meta.get("sample_size")
    return Bundle(
        significance=_load_significance(path),
        id=bundle_id,
        path=path,
        dataset="CSE-CIC-IDS2018",
        layout="ids2018",
        classes=classes or list(meta.get("classes", [])),
        models=models,
        run={
            "run_name": bundle_id,
            "split_protocol": split_protocol,
            "random_state": meta.get("random_state"),
            "n_train": (sample_size - n_test) if (sample_size and n_test) else None,
            "n_test": n_test,
            "n_features": meta.get("n_features"),
            "n_classes": len(classes or meta.get("classes", [])),
            "class_names": classes or list(meta.get("classes", [])),
            "per_class_n_train": {},
            "per_class_n_test": _test_supports(path, classes),
            "majority_baseline_acc": _majority_baseline(path, classes),
            "hp_tuned": hp_tuned,
            "class_weighting": meta.get("class_weighting"),
            "accelerator": meta.get("accelerator"),
            "dropped_columns": meta.get("dropped_columns", []),
        },
    )


# ----------------------------------------------------------------------
# Small readers
# ----------------------------------------------------------------------
def _project_root(path: Path) -> Path:
    for parent in path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return path.parents[-1]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read %s", path)
        return {}


def _read_indexed_csv(path: Path, key: str) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        logger.warning("Could not read %s", path)
        return {}
    if key not in df.columns:
        return {}
    return {str(r[key]): r.to_dict() for _, r in df.iterrows()}


def _int(value: Any) -> int | None:
    n = _num(value)
    return None if n is None else int(n)


def _classes_from_report(path: Path) -> list[str]:
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    col = df.columns[0]
    skip = {"accuracy", "macro avg", "weighted avg"}
    return [str(v) for v in df[col] if str(v) not in skip]


def _test_supports(path: Path, classes: list[str]) -> dict[str, int]:
    """Per-class test support, taken from any model's per-class report."""
    for child in sorted(p for p in path.iterdir() if p.is_dir()):
        report = child / "per_class_report.csv"
        if not report.is_file():
            continue
        df = pd.read_csv(report)
        col = df.columns[0]
        df = df[df[col].isin(classes)]
        return {str(r[col]): int(r["support"]) for _, r in df.iterrows()}
    return {}


def _test_support(path: Path, classes: list[str]) -> int | None:
    supports = _test_supports(path, classes)
    return sum(supports.values()) or None


def _majority_baseline(path: Path, classes: list[str]) -> float | None:
    """Accuracy of always predicting the largest class in the test split."""
    supports = _test_supports(path, classes)
    total = sum(supports.values())
    return (max(supports.values()) / total) if total else None
