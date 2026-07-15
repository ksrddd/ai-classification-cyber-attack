#!/usr/bin/env python
"""Atomically promote one validated all-model run into results/latest."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODEL_ORDER = (
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "mlp",
    "logistic_regression",
    "stacking",
)
SUFFIXES = (".joblib", "_metrics.json", "_per_class.csv", "_confusion_matrix.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--destination-run", default="latest")
    parser.add_argument("--baseline-run", default="latest")
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required artifact is missing or empty: {path}")


def class_metrics(run_dir: Path, model: str) -> dict[str, float | int] | None:
    path = run_dir / f"{model}_per_class.csv"
    if not path.exists():
        return None
    row = pd.read_csv(path, index_col=0).loc["Infiltration"]
    support = int(row["support"])
    true_positives = int(round(support * float(row["recall"])))
    return {
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1": float(row["f1-score"]),
        "support": support,
        "false_negatives": support - true_positives,
    }


def main() -> int:
    args = parse_args()
    source = RESULTS / args.source_run
    destination = RESULTS / args.destination_run
    baseline = RESULTS / args.baseline_run
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination runs must differ")

    source_metrics_path = source / "metrics.json"
    require_file(source_metrics_path)
    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    # A recovery pass may deliberately use a different FPR ceiling for a
    # difficult model.  Build the final aggregate from the authoritative
    # per-model JSON files instead of assuming every model shares the
    # top-level run configuration.
    source_entries: dict[str, dict] = {}
    for model in MODEL_ORDER:
        metrics_path = source / f"{model}_metrics.json"
        require_file(metrics_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("model") != model:
            raise ValueError(
                f"Expected model={model!r} in {metrics_path}; "
                f"got {metrics.get('model')!r}"
            )
        source_entries[model] = metrics

    baseline_metrics = {
        model: class_metrics(baseline, model) for model in MODEL_ORDER
    }
    for model in MODEL_ORDER:
        for suffix in SUFFIXES:
            require_file(source / f"{model}{suffix}")
    for common_name in ("label_encoder.joblib", "feature_columns.json"):
        require_file(source / common_name)

    staging = RESULTS / f".{args.destination_run}_staging"
    backup = RESULTS / f".{args.destination_run}_previous"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    final_entries: list[dict] = []
    comparison_rows: list[dict] = []
    manifest_models: dict[str, dict] = {}
    for model in MODEL_ORDER:
        for suffix in SUFFIXES:
            shutil.copy2(source / f"{model}{suffix}", staging / f"{model}{suffix}")
        metrics = json.loads(
            (source / f"{model}_metrics.json").read_text(encoding="utf-8")
        )
        current = class_metrics(source, model)
        if current is None:
            raise ValueError(f"Missing Infiltration report for {model}")
        previous = baseline_metrics[model]
        entry = dict(source_entries[model])
        entry.update({
            "source_run": args.source_run,
            "infiltration_precision": current["precision"],
            "infiltration_recall": current["recall"],
            "infiltration_f1": current["f1"],
        })
        if previous is not None:
            old_fn = int(previous["false_negatives"])
            new_fn = int(entry["target_false_negatives"])
            entry.update({
                "previous_target_recall": previous["recall"],
                "previous_target_false_negatives": old_fn,
                "target_fn_reduction": old_fn - new_fn,
                "target_fn_reduction_pct": (old_fn - new_fn) / max(old_fn, 1),
            })
            comparison_rows.append({
                "model": model,
                "previous_recall": previous["recall"],
                "new_recall": entry["target_recall"],
                "previous_false_negatives": old_fn,
                "new_false_negatives": new_fn,
                "false_negatives_reduced": old_fn - new_fn,
                "reduction_pct": entry["target_fn_reduction_pct"],
                "new_false_positives": entry["target_false_positives"],
                "new_fpr": entry["target_fpr"],
            })
        final_entries.append(entry)
        manifest_models[model] = {
            "source_run": args.source_run,
            "strategy": metrics.get("imbalance_strategy"),
            "target_threshold": metrics.get("target_threshold"),
            "model_bytes": (source / f"{model}.joblib").stat().st_size,
        }

    for common_name in ("label_encoder.joblib", "feature_columns.json"):
        shutil.copy2(source / common_name, staging / common_name)

    aggregate = dict(source_metrics)
    aggregate.update({
        "run_name": args.destination_run,
        "source_run": args.source_run,
        "finalized_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "per_model_threshold_policy": True,
        "selection_policy": (
            "Minimize Infiltration false negatives using genuine-target sampling, "
            "target-F2 tuning, and a natural-validation FPR ceiling."
        ),
        "models": final_entries,
    })
    (staging / "metrics.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    (staging / "final_manifest.json").write_text(
        json.dumps({
            "created_at": aggregate["finalized_at"],
            "source_run": args.source_run,
            "models": manifest_models,
        }, indent=2),
        encoding="utf-8",
    )
    if comparison_rows:
        pd.DataFrame(comparison_rows).to_csv(
            staging / "before_after_fn.csv", index=False
        )

    lines = [
        "# FN-aware final dashboard model set",
        "",
        f"Source run: `{args.source_run}`",
        "",
        "| model | threshold | recall | F2 | FPR | FN | FN to BENIGN | FP | FN reduced |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(final_entries, key=lambda item: item["target_false_negatives"]):
        reduction = row.get("target_fn_reduction")
        reduction_text = str(reduction) if reduction is not None else "n/a"
        lines.append(
            f"| {row['model']} | {row['target_threshold']:.6f} | "
            f"{row['target_recall']:.4f} | {row['target_f2']:.4f} | "
            f"{row['target_fpr']:.4f} | {row['target_false_negatives']} | "
            f"{row['target_to_benign_fn']} | {row['target_false_positives']} | "
            f"{reduction_text} |"
        )
    (staging / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for model in MODEL_ORDER:
        for suffix in SUFFIXES:
            require_file(staging / f"{model}{suffix}")
    for name in ("metrics.json", "report.md", "final_manifest.json"):
        require_file(staging / name)

    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.rename(backup)
    try:
        staging.rename(destination)
    except Exception:
        if backup.exists() and not destination.exists():
            backup.rename(destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)

    print(f"Promoted {len(MODEL_ORDER)} models from {source} to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
