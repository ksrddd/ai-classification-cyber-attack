"""Unified command-line entry point for the cyber-attack classification project.

Canonical usage
---------------
    python main.py --stage train
    python main.py --stage evaluate
    python main.py --stage dashboard
    python main.py --stage predict --input my_traffic.csv

Training writes the dashboard-ready artefacts under ``results/latest``. The
older modular stages remain available for development, but README-facing
workflows should use the commands above so there is only one path to remember.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import configure_logging, default_log_file  # noqa: E402

STAGES = (
    "train",
    "evaluate",
    "dashboard",
    "predict",
    "eda",
    "preprocess",
    "explain",
    "all",
    "audit",
    "red-team",
    "promote",
)
MODELS = (
    "all",
    "rf", "random_forest",
    "xgb", "xgboost",
    "lgbm", "lightgbm",
    "cat", "catboost",
    "nn", "mlp",
    "lr", "logistic_regression",
    "stack", "stacking",
)
RAM_PRESETS = ("8gb", "16gb", "32gb", "full")
RF_CLASS_WEIGHTS = ("none", "balanced", "balanced_subsample")
IMBALANCE_STRATEGIES = (
    "class_weight", "targeted", "random_over", "borderline_smote", "smoteenn",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cyberml",
        description="AI-Based Cyber Attack Classification -- unified driver",
    )
    parser.add_argument("--stage", choices=STAGES, required=True,
                        help="Which stage to run")
    parser.add_argument("--model", choices=MODELS, default="all",
                        help="Model for train/predict/explain (default: all for train)")
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "src" / "config" / "config.yaml",
                        help="Path to YAML config for legacy/development stages")
    parser.add_argument("--input", type=Path, default=None,
                        help="CSV file for --stage predict")
    parser.add_argument("--reference", type=Path, default=None,
                        help="Clean local CSV reference for --stage red-team")
    parser.add_argument("--candidate", type=Path, default=None,
                        help="Local labelled CSV candidate for --stage red-team")
    parser.add_argument("--label-column", default="Label",
                        help="Ground-truth label column for --stage red-team")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Trusted local pipeline artifact for red-team evasion checks")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional output CSV path for --stage predict")
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="Override raw data directory for legacy eda/preprocess stages")
    parser.add_argument("--run-name", default="latest",
                        help="Training output folder under results/ (default: latest)")
    parser.add_argument("--split-manifest", type=Path, default=None,
                        help="Versioned source-held-out split manifest for training")
    parser.add_argument("--profile", choices=("dev", "overnight"), default="dev",
                        help="Training resource profile")
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="cpu",
                        help="Use GPU-capable XGBoost/CatBoost backends when installed")
    parser.add_argument("--gpu-devices", default="0",
                        help="CatBoost GPU device selector (default: 0)")
    parser.add_argument("--preset", choices=RAM_PRESETS, default=None,
                        help="Training RAM preset for train.py")
    parser.add_argument("--force", action="store_true",
                        help="Retrain selected model artefacts even if they already exist")
    parser.add_argument("--refresh-cache", action="store_true",
                        help="Rebuild data/processed/cicids_clean.parquet from raw CSVs")
    parser.add_argument("--refresh-plots", action="store_true",
                        help="Re-evaluate saved models and refresh metrics/plots")
    parser.add_argument("--skip-tuning", "--skip-hp", dest="skip_tuning",
                        action="store_true",
                        help="Skip hyperparameter search during training")
    parser.add_argument("--reuse-best-params", action="store_true",
                        help="Reuse saved best_params while refitting")
    parser.add_argument("--skip-cv", action="store_true",
                        help="Skip full-train cross-validation during training")
    parser.add_argument("--skip-label-shuffle", action="store_true",
                        help="Skip shuffled-label sanity check during training")
    parser.add_argument("--primary-metric", default=None,
                        help="Hyperparameter-search metric, e.g. accuracy or f1_macro")
    parser.add_argument("--rf-class-weight", choices=RF_CLASS_WEIGHTS, default=None,
                        help="RandomForest class_weight override")
    parser.add_argument("--imbalance-strategy", choices=IMBALANCE_STRATEGIES,
                        default="class_weight",
                        help="Train-only class imbalance treatment")
    parser.add_argument("--target-class", default="Infiltration",
                        help="Minority class targeted by sampling strategies")
    parser.add_argument("--target-ratio", type=float, default=1.00,
                        help="Target-class / majority ratio after train sampling")
    parser.add_argument("--target-max-fpr", type=float, default=0.02,
                        help="Maximum validation false-positive rate for target threshold")
    parser.add_argument("--threshold-validation-size", type=float, default=0.0,
                        help="Train-only calibration fraction (0 keeps the required 70/30 split)")
    parser.add_argument("--port", type=int, default=8501,
                        help="Dashboard port for --stage dashboard")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(level=args.log_level, log_file=default_log_file())

    if args.stage == "train":
        return _run_latest_train(args)

    if args.stage == "evaluate":
        _print_latest_evaluation(args.run_name)
        return 0

    if args.stage == "dashboard":
        return _run_dashboard(args.port)

    if args.stage == "all":
        return _run_latest_train(args)

    if args.stage == "audit":
        return _run_audit(args)

    if args.stage == "promote":
        return _run_promote(args)

    if args.stage == "red-team":
        return _run_red_team(args)

    if args.stage == "eda":
        from src.pipelines.eda import run as run_eda
        run_eda(args.config, raw_dir_override=args.raw_dir)
        return 0

    if args.stage == "preprocess":
        from src.pipelines.preprocess import run as run_preprocess
        run_preprocess(args.config, raw_dir_override=args.raw_dir)
        return 0

    if args.stage == "explain":
        from src.pipelines.explain import run as run_explain
        run_explain(args.config, model=args.model)
        return 0

    if args.stage == "predict":
        if args.input is None:
            raise SystemExit("--stage predict requires --input PATH.csv")
        from src.pipelines.predict import run as run_predict
        out = run_predict(
            input_csv=args.input,
            model_name=_canonical_model_name(args.model),
            output_csv=args.output,
        )
        print(f"Predictions written to: {out}")
        return 0

    raise SystemExit(f"Unhandled stage: {args.stage}")


def _run_latest_train(args: argparse.Namespace) -> int:
    from train import main as train_main

    train_args: list[str] = ["--run-name", args.run_name]
    if args.split_manifest:
        train_args.extend(["--split-manifest", str(args.split_manifest)])
    if args.model != "all":
        train_args.extend(["--models", _canonical_model_name(args.model)])
    if args.preset:
        train_args.extend(["--preset", args.preset])
    elif args.profile == "overnight":
        train_args.extend(["--preset", "16gb"])
    if args.force:
        train_args.append("--force")
    if args.refresh_cache:
        train_args.append("--refresh-cache")
    if args.refresh_plots:
        train_args.append("--refresh-plots")
    if args.skip_tuning:
        train_args.append("--skip-hp")
    if args.reuse_best_params:
        train_args.append("--reuse-best-params")
    if args.skip_cv:
        train_args.append("--skip-cv")
    if args.skip_label_shuffle:
        train_args.append("--skip-label-shuffle")
    if args.primary_metric:
        train_args.extend(["--primary-metric", args.primary_metric])
    if args.rf_class_weight:
        train_args.extend(["--rf-class-weight", args.rf_class_weight])
    train_args.extend(["--imbalance-strategy", args.imbalance_strategy])
    train_args.extend(["--target-class", args.target_class])
    train_args.extend(["--target-ratio", str(args.target_ratio)])
    train_args.extend(["--target-max-fpr", str(args.target_max_fpr)])
    train_args.extend(["--accelerator", args.accelerator])
    train_args.extend(["--gpu-devices", args.gpu_devices])
    train_args.extend([
        "--threshold-validation-size", str(args.threshold_validation_size),
    ])

    return int(train_main(train_args))


def _print_latest_evaluation(run_name: str) -> None:
    metrics_path = PROJECT_ROOT / "results" / run_name / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(
            f"{metrics_path} not found. Run `python main.py --stage train` first."
        )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    models = payload.get("models", [])
    if not models:
        raise SystemExit(f"No models found in {metrics_path}.")
    best = min(
        models,
        key=lambda m: m.get("target_false_negatives", float("inf")),
    )
    print(f"Run: {payload.get('run_name', run_name)}")
    print(f"Rows: train={payload.get('n_train')} test={payload.get('n_test')}")
    print(
        f"Fewest target false negatives: {best['model']} = "
        f"{best.get('target_false_negatives', 0)}"
    )
    print("\nModel metrics:")
    for item in models:
        print(
            f"- {item['model']}: "
            f"acc={item.get('accuracy', 0.0):.4f}, "
            f"f1_weighted={item.get('f1_weighted', 0.0):.4f}, "
            f"f1_macro={item.get('f1_macro', 0.0):.4f}, "
            f"target_recall={item.get('target_recall', 0.0):.4f}, "
            f"target_f2={item.get('target_f2', 0.0):.4f}, "
            f"target_fn={item.get('target_false_negatives', 0)}, "
            f"target_fp={item.get('target_false_positives', 0)}"
        )
    print(f"\nDashboard report: {metrics_path.parent / 'report.md'}")


def _run_dashboard(port: int) -> int:
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_ROOT / "dashboard" / "app.py"),
        "--server.port",
        str(port),
    ]
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


def _run_audit(args: argparse.Namespace) -> int:
    from src.data.deterministic_split import load_split_manifest

    path = args.split_manifest or PROJECT_ROOT / "configs" / "splits" / "source_holdout_v2_70_30.json"
    try:
        payload = load_split_manifest(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid split manifest: {exc}") from exc
    print(json.dumps({"manifest": str(path), "version": payload["version"], "valid": True}, indent=2))
    return 0


def _run_promote(args: argparse.Namespace) -> int:
    from src.artifacts.paths import result_run_dir
    from src.artifacts.publish import promote_run

    run_id = args.run_name
    try:
        run_dir = result_run_dir(run_id, results_root=PROJECT_ROOT / "results")
    except ValueError as exc:
        raise SystemExit(f"Invalid run name: {exc}") from exc
    champion = PROJECT_ROOT / "results" / "champion.json"
    selected_model = None if args.model == "all" else _canonical_model_name(args.model)
    promote_run(
        run_dir,
        champion,
        champion_model=selected_model,
        target_max_fpr=args.target_max_fpr,
    )
    print(f"Promoted run {run_id} -> {champion}")
    return 0


def _run_red_team(args: argparse.Namespace) -> int:
    if args.reference is None or args.candidate is None:
        raise SystemExit(
            "--stage red-team requires --reference CLEAN.csv and --candidate LABELLED.csv"
        )
    import pandas as pd

    from src.security.red_team import run_red_team_report, write_red_team_report

    predictor = None
    if args.model_path is not None:
        from src.utils.io import load_joblib

        trusted_model = load_joblib(args.model_path)
        if not hasattr(trusted_model, "predict"):
            raise SystemExit("--model-path must be a trusted local object with predict(DataFrame)")
        predictor = trusted_model.predict
    report = run_red_team_report(
        pd.read_csv(args.reference, low_memory=False),
        pd.read_csv(args.candidate, low_memory=False),
        label_column=args.label_column,
        predictor=predictor,
    )
    output = PROJECT_ROOT / "results" / args.run_name / "red_team.json"
    write_red_team_report(report, output)
    print(f"Offline red-team report written to: {output}")
    return 0


def _canonical_model_name(name: str) -> str:
    from src.models.registry import resolve_name
    if name == "all":
        return "random_forest"
    return resolve_name(name)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
