"""Discover results bundles and normalise them to one schema.

Three incompatible layouts exist in ``results/``:

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

    Cross-dataset (``results/crossdataset/<run>/``)
        results.csv                  one row per (mode, seed, train, test, model)
        summary_gap.csv              ceiling / transfer / recovered, per model
        summary_leakage.csv          random-vs-chronological ceiling inflation
        summary_per_class.csv        per-class F1 for all four combinations
        summary_support.csv          test rows per class, per corpus
        corpora.json                 what entered the run, per corpus
        split.json                   train/test shape (absent = legacy 60/10/30)

    Audits (``results/crossdataset/<audit>/``)
        findings.json + ranking.csv  adversarial validation
        findings.json + report.md    77-feature mapping verification
        ids2017.json / ids2018.json  stacking data-leakage audit
        summary.csv + per_model.csv  feature-ablation experiment

Neither is a superset of the other, so normalisation cannot mean "fill in
the gaps". A field the bundle does not record is emitted as ``None`` and the
UI is expected to render it as absent. Substituting 0.0 would be worse than
useless here: a false-positive rate of "not recorded" and one of "zero" lead
to opposite conclusions.

An audit is not a training bundle either, and for a stronger reason: it does
not score models at all. It asks whether a *result* can be believed -- do the
columns mean the same thing on both sides, can the stacking ensemble see its
own test set, does anything in the feature matrix name the corpus a row came
from. There is no accuracy to report and no model to rank, so ``models`` is
empty here too and the evidence lives in ``audit``, whose shape depends on
which question the audit asked (``audit["kind"]`` says which).

The cross-dataset layout is not a training bundle and is deliberately not
bent into one. A training bundle answers "which of these seven models is
best"; a cross-dataset run answers "how much of what a model learned on one
corpus survives the move to the other", and its unit of result is a
(train corpus, test corpus) pair rather than a model. There is no champion to
name and no single score per model to rank, so ``models`` stays empty for
this layout and the four-way payload lives in ``crossdataset`` instead. A
consumer written for the other two layouts therefore reads nothing here,
which is the intended failure: an empty table is recoverable, a table of
numbers drawn from whichever of the four combinations happened to sort first
is not.
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
    layout: str  # "cicids2017" | "ids2018" | "crossdataset" | "audit"
    run: dict[str, Any] = field(default_factory=dict)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    classes: list[str] = field(default_factory=list)
    # Paired comparisons against the bundle's leading model, when the bundle
    # ran them. Keyed by rival model name. See _load_significance.
    significance: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Four-way transfer payload, present only on the crossdataset layout and
    # None everywhere else. See _load_crossdataset.
    crossdataset: dict[str, Any] | None = None
    # Evidence payload, present only on the audit layout and None everywhere
    # else. Its shape varies by ``audit["kind"]``. See _load_audit.
    audit: dict[str, Any] | None = None

    def to_summary(self) -> dict[str, Any]:
        """Small payload for the bundle picker in the sidebar."""
        return {
            "id": self.id,
            "dataset": self.dataset,
            "layout": self.layout,
            # Overridable because the crossdataset layout evaluates models
            # without producing one metric row per model: seven models ran,
            # and reporting zero there would understate the run rather than
            # describe it. Layouts that do fill ``models`` never set the key.
            "n_models": self.run.get("n_models", len(self.models)),
            "n_classes": len(self.classes),
            "split_protocol": self.run.get("split_protocol"),
            "n_train": self.run.get("n_train"),
            "n_test": self.run.get("n_test"),
            "n_features": self.run.get("n_features"),
            "class_weighting": self.run.get("class_weighting"),
            "hp_tuned": self.run.get("hp_tuned"),
            # None on every other layout. The picker uses it to name what an
            # audit asked, since "0 models, 0 classes" describes none of them.
            "audit_kind": self.audit.get("kind") if self.audit else None,
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


def _is_crossdataset_bundle(path: Path) -> bool:
    """A cross-dataset run: the raw rows plus the gap table derived from them.

    Both files are required. ``results.csv`` alone is what a run that died
    before ``scripts/analyze_crossdataset.py`` leaves behind, and the picker
    should not offer a run whose summary tables were never computed.
    """
    return (path / "results.csv").is_file() and (path / "summary_gap.csv").is_file()


#: Audit kind -> the files that identify it. The sets are disjoint, so at
#: most one can match a directory; ``findings.json`` appears twice and is
#: disambiguated by its companion (a ranking CSV for the adversarial run, a
#: written report for the mapping verification).
_AUDIT_FILES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("adversarial_validation", ("findings.json", "ranking.csv")),
    ("feature_mapping", ("findings.json", "report.md")),
    ("stacking_leakage", ("ids2017.json",)),
    ("ablation", ("summary.csv", "per_model.csv")),
)


def _audit_kind(path: Path) -> str | None:
    """Which question this directory holds the answer to, or None."""
    for kind, required in _AUDIT_FILES:
        if all((path / name).is_file() for name in required):
            return kind
    return None


def _is_audit_bundle(path: Path) -> bool:
    return _audit_kind(path) is not None


def _is_bundle(path: Path) -> bool:
    return (
        _is_2017_bundle(path)
        or _is_2018_bundle(path)
        or _is_crossdataset_bundle(path)
        or _is_audit_bundle(path)
    )


def _discover(results_dir: Path) -> dict[str, Path]:
    """Map bundle id -> directory, searching one level and one nested level.

    2017 runs sit directly under results/; 2018 and cross-dataset runs are
    grouped one level deeper (results/ids2018/300k,
    results/crossdataset/protocol_v1), so both depths are scanned.

    Directories whose name starts with an underscore are skipped at both
    depths. That prefix marks a superseded run kept on disk only long enough
    to diff against its successor -- ``results/crossdataset/_*`` is gitignored
    for the same reason. Those runs hold numbers the project has already
    retracted, and a picker entry is an invitation to quote one.
    """
    found: dict[str, Path] = {}
    if not results_dir.is_dir():
        return found

    for child in sorted(results_dir.iterdir()):
        if not child.is_dir() or child.name in _NON_BUNDLE or child.name.startswith("_"):
            continue
        if _is_bundle(child):
            found[child.name] = child
            continue
        for grandchild in sorted(child.iterdir()):
            if not grandchild.is_dir() or grandchild.name.startswith("_"):
                continue
            if _is_bundle(grandchild):
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
    if _is_2017_bundle(path):
        return _load_2017(bundle_id, path)
    # Checked before the 2018 probe, which walks every subdirectory looking
    # for metrics.json and would otherwise be the one to answer for a
    # cross-dataset run that grew a subdirectory later.
    if _is_crossdataset_bundle(path):
        return _load_crossdataset(bundle_id, path)
    kind = _audit_kind(path)
    if kind is not None:
        return _load_audit(bundle_id, path, kind)
    return _load_2018(bundle_id, path)


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
        # None for the two training layouts. The UI branches on `layout`, so
        # a page that cannot render transfer results never reaches this key.
        "crossdataset": bundle.crossdataset,
        # Likewise None unless layout == "audit".
        "audit": bundle.audit,
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
            "per_class_n_train": _train_supports(path, classes, meta),
            "per_class_n_test": _test_supports(path, classes),
            "majority_baseline_acc": _majority_baseline(path, classes),
            "hp_tuned": hp_tuned,
            "class_weighting": meta.get("class_weighting"),
            "accelerator": meta.get("accelerator"),
            "dropped_columns": meta.get("dropped_columns", []),
        },
    )


def _load_crossdataset(bundle_id: str, path: Path) -> Bundle:
    """Read a cross-dataset run into the four-way transfer payload.

    Built from the summary tables rather than from ``results.csv``, so the
    dashboard shows the same aggregation the write-up cites instead of a
    second one computed here that could drift from it. ``results.csv`` stays
    on disk as the record those tables were derived from.

    ``models`` is left empty -- see the module docstring. ``run`` carries only
    what genuinely describes the whole run; the fields a training bundle
    would fill (a single train/test row count, one split protocol, a majority
    baseline) differ per corpus or per mode here and are reported inside the
    payload where they can be labelled, not flattened into run-level keys
    that would silently describe one corpus as if it were both.
    """
    gap = _read_csv(path / "summary_gap.csv")
    per_class = _read_csv(path / "summary_per_class.csv")
    leakage = _read_csv(path / "summary_leakage.csv")
    support = _read_csv(path / "summary_support.csv")
    corpora = _read_json(path / "corpora.json")
    # Runs made before the calibration part was dropped carry no split.json and
    # were all 60/10/30, so that is what an absent file means.
    shape = _read_json(path / "split.json").get("shape") or "60/10/30"

    modes = sorted({str(r["mode"]) for r in gap if r.get("mode")})
    models = sorted({str(r["model"]) for r in gap if r.get("model")})
    classes = _crossdataset_classes(per_class)

    return Bundle(
        id=bundle_id,
        path=path,
        # ASCII on purpose: this string reaches Windows consoles through
        # logging and the Streamlit path, and cp1252 cannot encode an arrow.
        dataset=" <-> ".join(_corpus_label(name) for name in corpora) or "cross-dataset",
        layout="crossdataset",
        classes=classes,
        run={
            "run_name": path.name,
            "split_protocol": f"{shape}, " + " + ".join(modes) if modes else None,
            "random_state": None,
            "n_train": None,
            "n_test": None,
            "n_features": _first_int(corpora, "n_features"),
            "n_classes": len(classes),
            "class_names": classes,
            "per_class_n_train": {},
            "per_class_n_test": {},
            "majority_baseline_acc": None,
            # The protocol runs library defaults on both sides on purpose:
            # tuning moves these models by 0.01-0.05 against a gap of
            # 0.4-0.6, and importing one corpus's tuned parameters would
            # hand that corpus an advantage shaped like a dataset effect.
            "hp_tuned": False,
            "class_weighting": None,
            "accelerator": None,
            "n_models": len(models),
        },
        crossdataset={
            "modes": modes,
            "models": models,
            "classes": classes,
            "corpora": corpora,
            "cells": _transfer_cells(gap),
            "per_class": [_per_class_row(r, classes) for r in per_class],
            "leakage": [
                {
                    "test": str(r.get("test")),
                    "model": str(r.get("model")),
                    "chronological": _num(r.get("chronological")),
                    "random": _num(r.get("random")),
                    "inflation": _num(r.get("inflation")),
                }
                for r in leakage
            ],
            "support": [
                {
                    "mode": str(r.get("mode")),
                    "test": str(r.get("test")),
                    "per_class": {c: _int(r.get(f"n__{c}")) for c in classes},
                }
                for r in support
            ],
        },
    )


def _transfer_cells(gap: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """All four (train, test) cells per mode and model, from the gap table.

    ``summary_gap.csv`` holds only the two transfer rows per (mode, model),
    but each carries the ``ceiling`` for the corpus it was tested on -- that
    ceiling *is* the diagonal cell, the score of a model trained and tested
    on the same corpus. Deriving the diagonal from it keeps the matrix
    internally consistent: the number the UI prints on the diagonal is the
    same one the recovered-percentage in the off-diagonal was measured
    against, rather than a second reading of it from another file.

    Diagonal cells carry no ``recovered_pct``. It would be 100% by
    construction there, and a tautology printed in the same column as a
    measurement reads as the best result in the table.
    """
    cells: list[dict[str, Any]] = []
    seen_self: set[tuple[str, str, str]] = set()

    for row in gap:
        mode = str(row.get("mode"))
        model = str(row.get("model"))
        train = str(row.get("train"))
        test = str(row.get("test"))
        floor = _num(row.get("floor"))
        ceiling = _num(row.get("ceiling"))

        cells.append(
            {
                "mode": mode,
                "model": model,
                "train": train,
                "test": test,
                "self": False,
                "f1_macro": _num(row.get("transfer")),
                "floor": floor,
                "ceiling": ceiling,
                "gap": _num(row.get("gap")),
                "recovered_pct": _num(row.get("recovered_pct")),
                "sd": _num(row.get("f1_macro_all_sd")),
            }
        )

        key = (mode, model, test)
        if key not in seen_self:
            seen_self.add(key)
            cells.append(
                {
                    "mode": mode,
                    "model": model,
                    "train": test,
                    "test": test,
                    "self": True,
                    "f1_macro": ceiling,
                    "floor": floor,
                    "ceiling": ceiling,
                    "gap": None,
                    "recovered_pct": None,
                    "sd": None,
                }
            )
    return cells


def _per_class_row(row: dict[str, Any], classes: list[str]) -> dict[str, Any]:
    train, test = str(row.get("train")), str(row.get("test"))
    return {
        "mode": str(row.get("mode")),
        "model": str(row.get("model")),
        "train": train,
        "test": test,
        "self": train == test,
        "f1": {c: _num(row.get(f"f1__{c}")) for c in classes},
    }


def _crossdataset_classes(per_class: list[dict[str, Any]]) -> list[str]:
    """Shared class names, in the column order the analysis wrote them."""
    if not per_class:
        return []
    return [k[len("f1__") :] for k in per_class[0] if str(k).startswith("f1__")]


def _corpus_label(name: str) -> str:
    return {"ids2017": "CICIDS2017", "ids2018": "CSE-CIC-IDS2018"}.get(name, name)


def _first_int(corpora: dict[str, Any], key: str) -> int | None:
    """A run-level value that every corpus agrees on, or None if they differ.

    The shared schema gives both corpora the same feature count, so reporting
    it once is accurate. Returning None when they disagree keeps a future run
    with a per-corpus schema from being described by whichever corpus the
    JSON happened to list first.
    """
    values = {_int(c.get(key)) for c in corpora.values() if isinstance(c, dict)}
    values.discard(None)
    return values.pop() if len(values) == 1 else None


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


def _read_csv(path: Path) -> list[dict[str, Any]]:
    """Rows of a CSV as plain dicts, or an empty list if it cannot be read.

    Missing cells arrive as NaN from pandas; they are left alone here and
    normalised by ``_num``/``_int`` at the point of use, which map NaN to
    None. Doing it here instead would have to guess whether a column is
    numeric.
    """
    if not path.is_file():
        return []
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        logger.warning("Could not read %s", path)
        return []
    return [r.to_dict() for _, r in df.iterrows()]


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


def _train_supports(path: Path, classes: list[str], meta: dict[str, Any] | None = None) -> dict[str, int]:
    """Per-class *train* counts, from whichever record the run left behind.

    Two sources, checked in this order:

    ``split_manifest.json``
        Written by the temporal protocol only, and the older of the two. It is
        preferred because it is derived from the split itself rather than from
        the encoded labels, so a bundle carrying both is read from the record
        that the analysis step also validates.

    ``metadata.json`` -> ``per_class_n_train``
        Written by every run from the commit that added it, under both
        protocols. Bundles trained before that have no such key and fall
        through to ``{}``; those are backfilled by
        ``scripts/backfill_ids2018_counts.py`` rather than guessed at read time.

    Deriving the counts from the test supports and the 70/30 ratio is
    deliberately not done. It would be an inference presented in the same
    typeface as the measured columns, and for the rarest classes -- where the
    split floor at ``min_test_per_class`` moves the ratio away from 70/30 --
    it would be wrong in exactly the rows a reader is most likely to check.
    """
    wanted = set(classes)

    manifest = _read_json(path / "split_manifest.json")
    counts = manifest.get("expected_counts") or {}
    if counts:
        return {
            str(name): int(side["train"])
            for name, side in counts.items()
            if str(name) in wanted and isinstance(side, dict) and "train" in side
        }

    recorded = (meta or {}).get("per_class_n_train") or {}
    return {str(name): int(n) for name, n in recorded.items() if str(name) in wanted}


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


# ----------------------------------------------------------------------
# Audits
# ----------------------------------------------------------------------
#: How many per-class permutation rows to carry. The overall ranking below
#: is complete; this is the per-class detail beside it, where everything past
#: the first handful sits in the noise floor the run itself reports.
_AUDIT_TOP_FEATURES = 10


def _load_audit(bundle_id: str, path: Path, kind: str) -> Bundle:
    """Read one audit directory into an evidence payload.

    The four audits answer different questions and have no schema in common,
    so ``audit["kind"]`` discriminates and the UI branches on it. What they do
    share is the reason they are here at all: each one can *fail*, and a check
    whose result is only ever read by whoever ran it is not much of a check.

    Numbers are passed through close to the shape the audit script wrote,
    rather than re-aggregated. A second aggregation computed here would be a
    second answer to the same question, free to drift from the one the report
    cites, and the whole point of an audit is that there is exactly one.
    """
    builders = {
        "adversarial_validation": _audit_adversarial,
        "feature_mapping": _audit_feature_mapping,
        "stacking_leakage": _audit_stacking_leakage,
        "ablation": _audit_ablation,
    }
    payload, classes, n_features, dataset = builders[kind](path)
    payload["kind"] = kind

    return Bundle(
        id=bundle_id,
        path=path,
        dataset=dataset,
        layout="audit",
        classes=classes,
        run={
            "run_name": path.name,
            "split_protocol": None,
            "random_state": None,
            "n_train": None,
            "n_test": None,
            "n_features": n_features,
            "n_classes": len(classes),
            "class_names": classes,
            "per_class_n_train": {},
            "per_class_n_test": {},
            "majority_baseline_acc": None,
            "hp_tuned": None,
            "class_weighting": None,
            "accelerator": None,
            # An audit scores no models. Zero is the honest answer here, and
            # unlike the cross-dataset layout there is nothing to override it
            # with -- see to_summary.
            "n_models": 0,
        },
        audit=payload,
    )


_BOTH_CORPORA = " <-> ".join(_corpus_label(n) for n in ("ids2017", "ids2018"))


def _audit_adversarial(path: Path) -> tuple[dict[str, Any], list[str], int | None, str]:
    """Can a classifier name the corpus a row came from, within one class?

    Carries three layers, because the headline AUC alone is not readable: the
    per-class score with its ablation curve (how much of the feature space has
    to go before separability breaks), the permutation ranking (which columns
    carry it), and the diagnostics that tried to break the result (a model
    ladder down to a single threshold, a null control, near-duplicate and
    disjoint-column checks). The diagnostics are optional -- an older run may
    predate ``scripts/diagnose_av.py`` -- and arrive as None when absent,
    which the UI renders as "not run" rather than as "passed".
    """
    findings = _read_json(path / "findings.json")
    raw_classes = findings.get("classes", []) or []

    ks = sorted(
        {
            int(k)
            for row in raw_classes
            for k in (row.get("auc_after_dropping_top_k") or {})
        }
    )

    classes = [str(row.get("shared_class")) for row in raw_classes]
    payload = {
        "seeds": [_int(s) for s in findings.get("seeds", []) or []],
        "n_features": _int(findings.get("n_features")),
        "min_per_side": _int(findings.get("min_per_side")),
        "max_per_side": _int(findings.get("max_per_side")),
        "ablation_k": ks,
        "classes": [
            {
                "shared_class": str(row.get("shared_class")),
                "n_per_side": _int(row.get("n_per_side")),
                "n_seeds": _int(row.get("n_seeds")),
                "auc": _num(row.get("auc")),
                "auc_sd": _num(row.get("auc_sd")),
                "auc_per_seed": [_num(v) for v in row.get("auc_per_seed", []) or []],
                "ablation": {
                    str(k): _num(v)
                    for k, v in (row.get("auc_after_dropping_top_k") or {}).items()
                },
                "top_features": [
                    {
                        "feature": str(f.get("feature")),
                        "importance": _num(f.get("importance")),
                        "sd": _num(f.get("sd")),
                    }
                    for f in (row.get("importance") or [])[:_AUDIT_TOP_FEATURES]
                ],
            }
            for row in raw_classes
        ],
        "ranking": [
            {
                "feature": str(r.get("feature")),
                "mean": _num(r.get("mean")),
                "max": _num(r.get("max")),
                "count": _int(r.get("count")),
            }
            for r in _read_csv(path / "ranking.csv")
        ],
        "diagnostics": _audit_adversarial_diagnostics(path),
        "control": _audit_adversarial_control(path),
    }
    return payload, classes, payload["n_features"], _BOTH_CORPORA


def _audit_adversarial_control(path: Path) -> dict[str, Any] | None:
    """The within-dataset control, or None if it was never run.

    Passed through almost verbatim. The shape is a per-class dict of named
    contrasts, and flattening it to a single "within-dataset AUC" here would
    throw away which boundary produced it -- a time split and a capture-file
    split of the same class are different claims and can disagree.
    """
    raw = _read_json(path / "within_dataset_control.json")
    if not raw:
        return None
    return {
        # The follow-up that made two of the control's rows readable. Kept
        # nested rather than beside it because on its own it answers nothing:
        # it exists to say what the group contrast in DoS and DDoS was really
        # comparing.
        "capture_day": _audit_capture_day(path),
        # The strongest control the data supports: the attack tool held fixed
        # while the lab and the year change.
        "tool_matched": _audit_tool_matched(path),
        "question": raw.get("question"),
        "verdict": raw.get("verdict"),
        "reason": raw.get("reason"),
        "seeds": [_int(s) for s in raw.get("seeds", []) or []],
        "min_per_side": _int(raw.get("min_per_side")),
        "separable_threshold": _num(raw.get("separable_threshold")),
        "n_classes_judged": _int(raw.get("n_classes_judged")),
        "min_gap": _num(raw.get("min_gap")),
        "max_within_auc": _num(raw.get("max_within_auc")),
        # The pre-registered rule re-applied to each model's own numbers. A
        # robustness check reported beside the headline verdict, never in place
        # of it: the rule was registered against one model, and picking the
        # friendliest of eight afterwards is the freedom it exists to remove.
        "per_model": {
            name: {
                "verdict": str(v.get("verdict")),
                "n_classes_judged": _int(v.get("n_classes_judged")),
                "min_gap": _num(v.get("min_gap")),
                "max_within_auc": _num(v.get("max_within_auc")),
                "max_null_auc": _num(v.get("max_null_auc")),
            }
            for name, v in (raw.get("per_model") or {}).items()
        },
        "classes": [
            {
                "shared_class": str(c.get("shared_class")),
                "n_per_side": _int(c.get("n_per_side")),
                "cross_auc": _num(c.get("cross_auc")),
                "within_auc_max": _num(c.get("within_auc_max")),
                "gap": _num(c.get("gap")),
                # Per model, so a reader can see whether the comparison holds
                # across the study's seven or only for the one that produced
                # the headline. Each entry pairs a model's own cross figure
                # with its own within figure rather than mixing two models.
                "per_model": {
                    name: {
                        "cross_auc": _num(v.get("cross_auc")),
                        "within_auc_max": _num(v.get("within_auc_max")),
                        "gap": _num(v.get("gap")),
                        "null_auc_max": _num(v.get("null_auc_max")),
                    }
                    for name, v in (c.get("per_model") or {}).items()
                },
                "contrasts": {
                    key: {
                        "auc": _num(v.get("auc")),
                        "auc_sd": _num(v.get("auc_sd")),
                        "stump": _num(v.get("stump")),
                        "kind": str(v.get("kind")),
                        "boundary": str(v.get("boundary")),
                        "n_available_per_side": _int(v.get("n_available_per_side")),
                    }
                    for key, v in (c.get("contrasts") or {}).items()
                },
            }
            for c in raw.get("classes", []) or []
        ],
    }


def _audit_adversarial_diagnostics(path: Path) -> dict[str, Any] | None:
    """The attempt to break the AUC, or None if it was never run."""
    raw = _read_json(path / "diagnostics.json")
    if not raw:
        return None
    return {
        "split": raw.get("split"),
        "preprocessing": raw.get("preprocessing"),
        "seeds": [_int(s) for s in raw.get("seeds", []) or []],
        "max_per_side": _int(raw.get("max_per_side")),
        "near_dup_quantile": _num(raw.get("near_dup_quantile")),
        "classes": [
            {
                "shared_class": str(c.get("shared_class")),
                "n_per_side": _int(c.get("n_per_side")),
                "auc_mean": _num(c.get("auc_mean")),
                "auc_sd": _num(c.get("auc_sd")),
                "best_single_feature": c.get("best_single_feature"),
                "best_single_feature_auc": _num(c.get("best_single_feature_auc")),
                "best_single_feature_median": {
                    k: _num(v)
                    for k, v in (c.get("best_single_feature_median") or {}).items()
                },
                "disjoint_columns": [str(x) for x in c.get("disjoint_columns", []) or []],
                "rows_identical_across_corpora": _int(
                    c.get("rows_identical_across_corpora")
                ),
                "rows_identical_across_split": _int(c.get("rows_identical_across_split")),
                "near_duplicate_test_share": _num(c.get("near_duplicate_test_share")),
                "model_ladder": {
                    k: _num(v) for k, v in (c.get("model_ladder") or {}).items()
                },
                "null_ladder": {
                    corpus: {k: _num(v) for k, v in rungs.items()}
                    for corpus, rungs in (c.get("null_ladder") or {}).items()
                },
            }
            for c in raw.get("classes", []) or []
        ],
    }


def _audit_capture_day(path: Path) -> dict[str, Any] | None:
    """Two capture days of the same attack, against two different attacks.

    The within-dataset control found a *perfect* within-2018 separation in DoS
    and DDoS, which would have refuted the fingerprint reading had it meant
    what it appeared to. It did not: the 2018 corpus gives each attack tool its
    own capture day and the shared schema collapses several tools into one
    class, so that contrast compared Hulk against GoldenEye rather than one
    Thursday against another. These two ranges separate the explanations.
    """
    raw = _read_json(path / "capture_day_control.json")
    if not raw:
        return None

    def rows(key: str) -> list[dict[str, Any]]:
        return [
            {
                "label": str(r.get("label")),
                "kind": str(r.get("kind")),
                "boundary": str(r.get("boundary")),
                "n_per_side": _int(r.get("n_per_side")),
                "auc": _num(r.get("auc")),
                "auc_sd": _num(r.get("auc_sd")),
                "stump": _num(r.get("stump")),
            }
            for r in raw.get(key, []) or []
        ]

    return {
        "question": raw.get("question"),
        "corpus": raw.get("corpus"),
        "min_per_side": _int(raw.get("min_per_side")),
        "same_tool_other_day": rows("same_tool_other_day"),
        "other_tool_same_class": rows("other_tool_same_class"),
    }


def _audit_tool_matched(path: Path) -> dict[str, Any] | None:
    """The same attack program, run in two different labs two years apart.

    The capture-day follow-up found that two attack tools the shared schema
    collapses into one class separate at AUC 1.0. That raises the same question
    about the cross-dataset result itself, since "DoS" names a different
    mixture of tools on each side -- so part of the headline AUC could be tool
    composition rather than anything about the labs.

    Each tool carries its class-level contrast re-run at the same rows per
    side, because the tool-matched number only means something against a
    like-for-like baseline rather than against a figure from another draw.
    """
    raw = _read_json(path / "tool_matched.json")
    if not raw:
        return None

    def scored(block: dict[str, Any] | None) -> dict[str, Any]:
        block = block or {}
        return {
            "auc": _num(block.get("auc")),
            "auc_sd": _num(block.get("auc_sd")),
            "stump": _num(block.get("stump")),
        }

    return {
        "question": raw.get("question"),
        "seeds": [_int(v) for v in raw.get("seeds", []) or []],
        "tool_pairs": [
            {
                "tool": str(r.get("tool")),
                "shared_class": str(r.get("shared_class")),
                "ids2017_label": str(r.get("ids2017_label")),
                "ids2018_label": str(r.get("ids2018_label")),
                # True where the two labels do not say the same word and the
                # pairing rests on CIC's documentation instead.
                "identity_from_documentation": bool(
                    r.get("identity_from_documentation")
                ),
                "n_per_side": _int(r.get("n_per_side")),
                "same_tool": scored(r.get("same_tool_across_corpora")),
                "whole_class": scored(r.get("whole_class_across_corpora")),
                "delta": _num(r.get("delta")),
            }
            for r in raw.get("tool_pairs", []) or []
        ],
        "different_tools_within_ids2017": [
            {
                "shared_class": str(r.get("shared_class")),
                "tools": [str(t) for t in r.get("tools", []) or []],
                "n_per_side": _int(r.get("n_per_side")),
                "auc": _num(r.get("auc")),
                "stump": _num(r.get("stump")),
            }
            for r in raw.get("different_tools_within_ids2017", []) or []
        ],
    }


def _audit_feature_mapping(
    path: Path,
) -> tuple[dict[str, Any], list[str], int | None, str]:
    """Do the 77 paired columns mean the same thing on both sides?

    The invariants are the load-bearing part: each is an algebraic identity
    evaluated on each corpus *separately*, so a pass does not depend on the
    two corpora agreeing about anything. The scale audit is reported beside
    them and must not be read as a verdict -- a column can shift by decades
    between corpora and still be the same measurement, which is the whole
    distinction the invariants exist to draw.
    """
    findings = _read_json(path / "findings.json")
    definitions = findings.get("invariant_definitions", {}) or {}

    payload = {
        "n_rows": {
            "ids2017": _int(findings.get("n_rows_ids2017")),
            "ids2018": _int(findings.get("n_rows_ids2018")),
        },
        "n_features": _int(findings.get("n_features")),
        "n_distinct_measurements": _int(findings.get("n_distinct_measurements")),
        "invariants": [
            {
                "invariant": str(r.get("invariant")),
                "verdict": str(r.get("verdict")),
                "rate_ids2017": _num(r.get("rate_ids2017")),
                "rate_ids2018": _num(r.get("rate_ids2018")),
                "columns": [str(c) for c in r.get("columns", []) or []],
                "note": r.get("note")
                or (definitions.get(str(r.get("invariant")), {}) or {}).get("note"),
            }
            for r in findings.get("invariants", []) or []
        ],
        "duplicate_groups": [
            [str(c) for c in group] for group in findings.get("duplicate_groups", []) or []
        ],
        "scale_audit": [
            {
                "feature": str(r.get("feature")),
                "flag": str(r.get("flag")),
                "reason": r.get("reason") or "",
                "ids2017": {k: _num(v) for k, v in (r.get("ids2017") or {}).items()},
                "ids2018": {k: _num(v) for k, v in (r.get("ids2018") or {}).items()},
            }
            for r in findings.get("scale_audit", []) or []
        ],
    }
    return payload, [], payload["n_features"], _BOTH_CORPORA


def _audit_stacking_leakage(
    path: Path,
) -> tuple[dict[str, Any], list[str], int | None, str]:
    """Can the stacking ensemble see the test set it is scored on?

    Run once per corpus and reported that way rather than merged. Three of the
    checks are protocol-level and cannot differ between corpora, but the
    duplicate-flow one can: the raw 2018 corpus is 29% duplicates, so it is
    the side where getting the order of deduplication and splitting wrong
    would do real damage, and a merged verdict would hide which side proved it.
    """
    corpora = []
    for name in ("ids2017", "ids2018"):
        raw = _read_json(path / f"{name}.json")
        if not raw:
            continue
        corpora.append(
            {
                "dataset": str(raw.get("dataset", name)),
                "label": _corpus_label(str(raw.get("dataset", name))),
                "run_utc": raw.get("run_utc"),
                "audit_rows": _int(raw.get("audit_rows")),
                "seed": _int(raw.get("seed")),
                "passed": bool(raw.get("passed")),
                "n_checks": _int(raw.get("n_checks")),
                "n_failed": _int(raw.get("n_failed")),
                "checks": [
                    {
                        "section": str(c.get("section")),
                        "check": str(c.get("check")),
                        "passed": bool(c.get("passed")),
                        "detail": str(c.get("detail", "")),
                    }
                    for c in raw.get("checks", []) or []
                ],
            }
        )

    payload = {"corpora": corpora}
    dataset = " + ".join(c["label"] for c in corpora) or _BOTH_CORPORA
    return payload, [], None, dataset


def _audit_ablation(path: Path) -> tuple[dict[str, Any], list[str], int | None, str]:
    """Does dropping the columns that name the corpus improve transfer?

    ``per_model.csv`` is wide -- one column per K -- and stays that way here.
    Melting it into (model, k, value) triples would lose the one thing the
    table is for: reading across a row to see whether a model's recovery moves
    monotonically with K, which is what separates a real effect from one
    K-value that happened to look good.
    """
    summary = _read_csv(path / "summary.csv")
    per_model = _read_csv(path / "per_model.csv")

    ks = [c for c in (per_model[0] if per_model else {}) if str(c).isdigit()]
    payload = {
        "summary": [
            {
                "k": _int(r.get("k")),
                "mode": str(r.get("mode")),
                "ceiling": _num(r.get("ceiling")),
                "transfer": _num(r.get("transfer")),
                "gap": _num(r.get("gap")),
                "recovered_pct": _num(r.get("recovered_pct")),
            }
            for r in summary
        ],
        "ks": [int(k) for k in ks],
        "per_model": [
            {
                "model": str(r.get("model")),
                "recovered_pct": {str(k): _num(r.get(k)) for k in ks},
                "change": _num(r.get("change")),
            }
            for r in per_model
        ],
        # The decision rule, written down before any of these runs existed.
        # Shipping it beside the result is the only thing that makes the
        # result a test rather than a search.
        "preregistration": _read_text(path / "PREREGISTRATION.md"),
    }
    return payload, [], None, _BOTH_CORPORA


def _read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("Could not read %s", path)
        return None
