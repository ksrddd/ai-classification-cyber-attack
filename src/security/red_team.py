"""Offline adversarial-ML report runner.

This module deliberately operates only on local CSV/DataFrame inputs.  It
does not send traffic, scan hosts, call APIs, or modify the source data.

Example
-------
``python -m src.security.red_team --reference train.csv --candidate test.csv
--label-column Label --model-path results/demo/model.joblib
--output results/demo/red_team.json``

Only pass a trusted, locally-created ``--model-path``: joblib files are Python
pickle data and must never be accepted from an untrusted source.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

from src.security.evasion import generate_bounded_perturbations, prediction_flip_rate
from src.security.ood import detect_ood, fit_iqr_profile
from src.security.poisoning import find_conflicting_labels, label_distribution_shift

REPORT_VERSION = "red-team-offline-v1"
DEFAULT_EPSILONS = (0.01, 0.05, 0.10)

Predictor = Callable[[pd.DataFrame], Sequence[object]]


def run_red_team_report(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    label_column: str,
    predictor: Predictor | None = None,
    feature_columns: list[str] | None = None,
    epsilons: Sequence[float] = DEFAULT_EPSILONS,
    ood_max_feature_fraction: float = 0.1,
) -> dict[str, Any]:
    """Run copy-only poisoning, OOD, and optional model-evasion checks.

    ``reference`` should be the clean training/calibration sample. ``candidate``
    is the local evaluation sample.  The original DataFrames are never changed.
    When a predictor is supplied, it receives feature-only copies and is used
    solely to compare clean and bounded-perturbation predictions.
    """
    _validate_inputs(reference, candidate, label_column, feature_columns, epsilons)
    features = feature_columns or [
        column
        for column in reference.columns
        if column != label_column and pd.api.types.is_numeric_dtype(reference[column])
    ]
    numeric_features = [
        column for column in features if pd.api.types.is_numeric_dtype(reference[column])
    ]
    if not numeric_features:
        raise ValueError("At least one numeric feature column is required")

    reference_features = reference.loc[:, numeric_features].copy(deep=True)
    candidate_features = candidate.loc[:, numeric_features].copy(deep=True)
    profile = fit_iqr_profile(reference_features, columns=numeric_features)
    ood = detect_ood(
        candidate_features,
        profile,
        max_feature_fraction=ood_max_feature_fraction,
    )

    poisoning = {
        "candidate_label_conflicts": _as_json(
            find_conflicting_labels(candidate, label_column=label_column, feature_columns=features)
        ),
        "label_distribution_shift": _as_json(
            label_distribution_shift(reference, candidate, label_column=label_column)
        ),
    }
    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "scope": "offline-local-data-only",
        "rows": {"reference": len(reference), "candidate": len(candidate)},
        "feature_columns": numeric_features,
        "poisoning": poisoning,
        "ood": _as_json(ood),
        "evasion": {"performed": False, "reason": "No predictor supplied."},
    }
    if predictor is not None:
        report["evasion"] = _evasion_report(
            candidate_features,
            candidate[label_column],
            predictor,
            epsilons,
        )
    return report


def write_red_team_report(report: dict[str, Any], output_path: Path) -> Path:
    """Atomically write a JSON report, creating its parent directory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, output_path)
    return output_path


def _evasion_report(
    features: pd.DataFrame,
    labels: pd.Series,
    predictor: Predictor,
    epsilons: Sequence[float],
) -> dict[str, Any]:
    clean = _predict(predictor, features)
    y_true = labels.to_numpy(copy=True)
    clean_metrics = _classification_metrics(y_true, clean)
    scenarios: list[dict[str, Any]] = []
    for epsilon in epsilons:
        perturbed = generate_bounded_perturbations(features, relative_epsilon=float(epsilon))
        predictions = _predict(predictor, perturbed)
        metrics = _classification_metrics(y_true, predictions)
        flip = prediction_flip_rate(clean, predictions)
        recall_drop = _worst_recall_drop(clean_metrics["recall_by_label"], metrics["recall_by_label"])
        robust_f1 = metrics["macro_f1"] >= 0.90 * clean_metrics["macro_f1"]
        robust_recall = recall_drop <= 0.15
        scenarios.append(
            {
                "epsilon": float(epsilon),
                "prediction_stability": _as_json(flip),
                "macro_f1": metrics["macro_f1"],
                "worst_class_recall_drop": recall_drop,
                "policy_pass": bool(robust_f1 and robust_recall),
            }
        )
    policy_scenario = next((item for item in scenarios if item["epsilon"] == 0.05), None)
    return {
        "performed": True,
        "clean_macro_f1": clean_metrics["macro_f1"],
        "clean_recall_by_label": clean_metrics["recall_by_label"],
        "scenarios": scenarios,
        "five_percent_policy_pass": None if policy_scenario is None else policy_scenario["policy_pass"],
    }


def _classification_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    labels = sorted({str(value) for value in y_true} | {str(value) for value in predictions})
    y_true_text = [str(value) for value in y_true]
    prediction_text = [str(value) for value in predictions]
    recalls = recall_score(y_true_text, prediction_text, labels=labels, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true_text, prediction_text, labels=labels, average="macro", zero_division=0)),
        "recall_by_label": dict(zip(labels, (float(value) for value in recalls), strict=True)),
    }


def _worst_recall_drop(clean: dict[str, float], perturbed: dict[str, float]) -> float:
    labels = set(clean) | set(perturbed)
    return max((clean.get(label, 0.0) - perturbed.get(label, 0.0) for label in labels), default=0.0)


def _predict(predictor: Predictor, features: pd.DataFrame) -> np.ndarray:
    values = np.asarray(list(predictor(features.copy(deep=True))))
    if len(values) != len(features):
        raise ValueError("Predictor must return exactly one prediction per candidate row")
    return values


def _validate_inputs(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    label_column: str,
    feature_columns: list[str] | None,
    epsilons: Sequence[float],
) -> None:
    if reference.empty or candidate.empty:
        raise ValueError("Reference and candidate data must both contain at least one row")
    if label_column not in reference.columns or label_column not in candidate.columns:
        raise ValueError(f"Missing label column: {label_column}")
    selected = feature_columns or [column for column in reference.columns if column != label_column]
    missing_reference = [column for column in selected if column not in reference.columns]
    missing_candidate = [column for column in selected if column not in candidate.columns]
    if missing_reference or missing_candidate:
        raise ValueError(
            f"Missing feature columns: reference={missing_reference}, candidate={missing_candidate}"
        )
    if not epsilons or any(float(epsilon) < 0 for epsilon in epsilons):
        raise ValueError("At least one non-negative epsilon is required")


def _as_json(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _as_json(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): _as_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_as_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parse_columns(value: str | None) -> list[str] | None:
    if value is None:
        return None
    columns = [column.strip() for column in value.split(",") if column.strip()]
    if not columns:
        raise argparse.ArgumentTypeError("--feature-columns cannot be empty")
    return columns


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline local red-team checks on CSV data")
    parser.add_argument("--reference", type=Path, required=True, help="Clean reference CSV")
    parser.add_argument("--candidate", type=Path, required=True, help="Evaluation CSV")
    parser.add_argument("--label-column", default="Label", help="Ground-truth label column")
    parser.add_argument("--feature-columns", type=_parse_columns, default=None,
                        help="Comma-separated numeric feature columns (default: all numeric)")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Trusted local joblib model/pipeline for evasion checks")
    parser.add_argument("--output", type=Path, required=True, help="JSON report output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reference = pd.read_csv(args.reference, low_memory=False)
    candidate = pd.read_csv(args.candidate, low_memory=False)
    predictor: Predictor | None = None
    if args.model_path is not None:
        model = joblib.load(args.model_path)
        if not hasattr(model, "predict"):
            raise ValueError("--model-path must resolve to an object with predict(DataFrame)")
        predictor = model.predict
    report = run_red_team_report(
        reference,
        candidate,
        label_column=args.label_column,
        predictor=predictor,
        feature_columns=args.feature_columns,
    )
    output = write_red_team_report(report, args.output)
    print(f"Offline red-team report written to: {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
