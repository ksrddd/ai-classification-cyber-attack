#!/usr/bin/env python
"""Recalibrate deployed Infiltration thresholds for balanced FP/FN behavior."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train  # noqa: E402

MODEL_ORDER = (
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "mlp",
    "logistic_regression",
    "stacking",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="latest")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Persist thresholds, metrics, reports, and confusion matrices.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(path)


def main() -> int:
    args = parse_args()
    run_dir = ROOT / "results" / args.run
    aggregate_path = run_dir / "metrics.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if aggregate.get("training_protocol_version") != train.TRAINING_PROTOCOL_VERSION:
        raise RuntimeError(
            "This run predates strict budget/fingerprint tracking. Retrain it "
            "before recalibrating thresholds."
        )

    cfg = dict(train.CONFIG)
    cfg.update(train.RAM_PRESETS["8gb"])
    df = train.load_and_clean_cached(cfg, force=False)
    current_fingerprint = train._dataset_fingerprint(df, cfg)
    if aggregate.get("data_fingerprint") != current_fingerprint:
        raise RuntimeError(
            "The cleaned dataset no longer matches this model run; retrain "
            "instead of recalibrating on different data."
        )
    total_rows = int(
        aggregate["n_train"]
        + aggregate["n_calibration"]
        + aggregate["n_test"]
    )
    train_df, calibration_df, test_df = train.budgeted_train_test_split(
        df,
        label_col=cfg["label_column"],
        total_budget=total_rows,
        test_size=cfg["test_size"],
        min_test_per_class=cfg["min_test_per_class"],
        rare_threshold=cfg["rare_threshold"],
        train_sampling="natural",
        target_class="Infiltration",
        target_ratio=1.0,
        random_state=int(aggregate["random_state"]),
        calibration_size=aggregate["n_calibration"] / total_rows,
    )
    del df, train_df
    gc.collect()

    feature_columns = json.loads(
        (run_dir / "feature_columns.json").read_text(encoding="utf-8")
    )
    label_encoder = joblib.load(run_dir / "label_encoder.joblib")
    target_idx = int(label_encoder.transform(["Infiltration"])[0])
    benign_idx = int(label_encoder.transform(["BENIGN"])[0])
    y_calibration = label_encoder.transform(
        calibration_df[cfg["label_column"]].to_numpy()
    )
    y_test = label_encoder.transform(test_df[cfg["label_column"]].to_numpy())
    X_calibration = calibration_df[feature_columns].astype(np.float32)
    X_test = test_df[feature_columns].astype(np.float32)
    del calibration_df, test_df
    gc.collect()

    aggregate_entries = {
        item["model"]: item for item in aggregate.get("models", [])
    }
    comparison_rows: list[dict[str, Any]] = []

    for model_name in MODEL_ORDER:
        model_path = run_dir / f"{model_name}.joblib"
        metrics_path = run_dir / f"{model_name}_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        model = joblib.load(model_path)
        old_threshold = float(metrics.get(
            "high_recall_threshold",
            model.target_threshold,
        ))
        old_fp = int(metrics.get(
            "high_recall_target_false_positives",
            metrics["target_false_positives"],
        ))
        old_fn = int(metrics.get(
            "high_recall_target_false_negatives",
            metrics["target_false_negatives"],
        ))

        probabilities = model.predict_proba(X_calibration)[:, target_idx]
        calibration = train.calibrate_target_threshold(
            y_calibration,
            probabilities,
            target_class_index=target_idx,
            max_false_positive_rate=float(metrics["target_max_fpr"]),
        )
        model.high_recall_threshold = old_threshold
        model.target_threshold = calibration.threshold
        accuracy, balanced_accuracy, macro_f1, weighted_f1, per_class, cm = (
            train.evaluate(model, X_test, y_test, list(label_encoder.classes_))
        )
        target = train.target_metrics_from_confusion(
            cm,
            target_class_index=target_idx,
            benign_class_index=benign_idx,
        )
        target_precision = float(target["target_precision"])
        target_recall = float(target["target_recall"])
        target_f1 = (
            2.0 * target_precision * target_recall
            / max(target_precision + target_recall, np.finfo(float).eps)
        )

        comparison_rows.append({
            "model": model_name,
            "high_recall_threshold": old_threshold,
            "high_recall_fp": old_fp,
            "high_recall_fn": old_fn,
            "balanced_threshold": calibration.threshold,
            "balanced_fp": target["target_false_positives"],
            "balanced_fn": target["target_false_negatives"],
            "balanced_target_f1": target_f1,
            "balanced_accuracy": accuracy,
            "balanced_macro_f1": macro_f1,
            "total_cross_errors_reduced": (
                old_fp + old_fn
                - int(target["target_false_positives"])
                - int(target["target_false_negatives"])
            ),
        })
        print(
            f"{model_name}: FP {old_fp} -> {target['target_false_positives']}, "
            f"FN {old_fn} -> {target['target_false_negatives']}, "
            f"threshold {old_threshold:.6f} -> {calibration.threshold:.6f}"
        )

        if args.apply:
            train.atomic_joblib_dump(model, model_path)
            per_class.to_csv(run_dir / f"{model_name}_per_class.csv", index=True)
            train.plot_confusion_matrix(
                cm,
                list(label_encoder.classes_),
                run_dir / f"{model_name}_confusion_matrix.png",
                title=(
                    f"{model_name} -- CICIDS2017 + "
                    "CSE-CIC-IDS2018 test set"
                ),
            )
            metrics.update({
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "f1_macro": macro_f1,
                "f1_weighted": weighted_f1,
                "threshold_objective": "target_f1_balanced",
                "high_recall_threshold": old_threshold,
                "high_recall_target_false_positives": old_fp,
                "high_recall_target_false_negatives": old_fn,
                "target_threshold": calibration.threshold,
                "target_f1": target_f1,
                "calibration_recall": calibration.recall,
                "calibration_fpr": calibration.false_positive_rate,
                **target,
            })
            write_json(metrics_path, metrics)
            aggregate_entries[model_name] = {
                **aggregate_entries.get(model_name, {}),
                **metrics,
            }

        del model, probabilities, per_class, cm
        gc.collect()

    comparison = pd.DataFrame(comparison_rows)
    if args.apply:
        comparison.to_csv(run_dir / "threshold_tradeoff.csv", index=False)
        aggregate.update({
            "threshold_objective": "target_f1_balanced",
            "models": [aggregate_entries[name] for name in MODEL_ORDER],
        })
        write_json(aggregate_path, aggregate)
        report_lines = [
            "# Balanced Infiltration decision thresholds",
            "",
            "Thresholds maximize Infiltration F1 on the natural calibration set.",
            "",
            "| model | threshold | precision | recall | F1 | FPR | FP | FN | FP + FN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for model_name in MODEL_ORDER:
            item = aggregate_entries[model_name]
            report_lines.append(
                f"| {model_name} | {item['target_threshold']:.6f} | "
                f"{item['target_precision']:.4f} | {item['target_recall']:.4f} | "
                f"{item['target_f1']:.4f} | {item['target_fpr']:.4f} | "
                f"{item['target_false_positives']} | "
                f"{item['target_false_negatives']} | "
                f"{item['target_false_positives'] + item['target_false_negatives']} |"
            )
        (run_dir / "report.md").write_text(
            "\n".join(report_lines) + "\n",
            encoding="utf-8",
        )
        manifest_path = run_dir / "final_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for model_name in MODEL_ORDER:
                if model_name in manifest.get("models", {}):
                    manifest["models"][model_name]["target_threshold"] = (
                        aggregate_entries[model_name]["target_threshold"]
                    )
                    manifest["models"][model_name]["threshold_objective"] = (
                        "target_f1_balanced"
                    )
            write_json(manifest_path, manifest)
    print(comparison.to_string(index=False))
    if not args.apply:
        print("Dry run only; pass --apply to persist balanced thresholds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
