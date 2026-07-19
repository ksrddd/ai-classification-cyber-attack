"""Validate DGX delivery configuration without pretending local hardware is a DGX."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.deterministic_split import load_split_manifest  # noqa: E402
from src.utils.io import json_dumps_strict  # noqa: E402

REQUIRED_SBATCH_DIRECTIVES = (
    "#SBATCH --nodes=1",
    "#SBATCH --ntasks=1",
    "#SBATCH --cpus-per-task=32",
    "#SBATCH --gres=gpu:1",
    "#SBATCH --mem=128G",
    "#SBATCH --time=12:00:00",
)
REQUIRED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "catboost",
    "pyarrow",
)


def _source_held_ratio(cache_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Count the selected source-held rows without loading the full cache."""
    import pyarrow.parquet as pq

    role_by_source = {
        Path(source).name: role
        for role, sources in manifest["roles"].items()
        for source in sources
    }
    available_train: Counter[str] = Counter()
    test_by_class: Counter[str] = Counter()
    unknown_sources: set[str] = set()

    parquet = pq.ParquetFile(cache_path)
    for batch in parquet.iter_batches(
        columns=["Label", "source_file"],
        batch_size=500_000,
    ):
        frame = batch.to_pandas()
        source_names = frame["source_file"].astype(str).map(
            lambda value: Path(value).name
        )
        roles = source_names.map(role_by_source)
        unknown_sources.update(source_names[roles.isna()].unique().tolist())
        for (role, label), count in (
            frame.assign(_role=roles)
            .dropna(subset=["_role"])
            .groupby(["_role", "Label"], observed=True)
            .size()
            .items()
        ):
            destination = available_train if role == "train" else test_by_class
            destination[str(label)] += int(count)

    quotas = manifest["train_quotas"]
    selected_train = {
        label: min(count, int(quotas.get(label, count)))
        for label, count in sorted(available_train.items())
    }
    train_rows = sum(selected_train.values())
    test_rows = sum(test_by_class.values())
    total_rows = train_rows + test_rows
    train_fraction = train_rows / total_rows if total_rows else 0.0
    test_fraction = test_rows / total_rows if total_rows else 0.0
    return {
        "train_rows": train_rows,
        "test_rows": test_rows,
        "total_rows": total_rows,
        "train_fraction": train_fraction,
        "test_fraction": test_fraction,
        "selected_train_by_class": selected_train,
        "test_by_class": dict(sorted(test_by_class.items())),
        "unknown_sources": sorted(unknown_sources),
    }


def _check(ok: bool, detail: Any) -> dict[str, Any]:
    return {"passed": bool(ok), "detail": detail}


def run_static_preflight(project_root: Path, manifest_path: Path) -> dict[str, Any]:
    """Run hardware-independent checks on the exact files submitted to Slurm."""
    project_root = project_root.resolve()
    manifest_path = manifest_path.resolve()
    manifest = load_split_manifest(manifest_path)
    sbatch_path = project_root / "scripts" / "dgx_train.sbatch"
    sbatch = sbatch_path.read_text(encoding="utf-8")

    source_names = sorted(
        name for names in manifest["roles"].values() for name in names
    )
    raw_dir = project_root / "data" / "raw"
    missing_sources = [name for name in source_names if not (raw_dir / name).is_file()]
    empty_sources = [
        name for name in source_names
        if (raw_dir / name).is_file() and (raw_dir / name).stat().st_size == 0
    ]

    versions: dict[str, str] = {}
    missing_packages: list[str] = []
    for package in REQUIRED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing_packages.append(package)

    from train import build_pipeline

    xgb = build_pipeline("xgboost", 9, 42, accelerator="gpu").named_steps["clf"]
    cat = build_pipeline("catboost", 9, 42, accelerator="gpu").named_steps["clf"]
    stack = build_pipeline("stacking", 9, 42, accelerator="gpu").named_steps["clf"]
    stack_xgb = dict(stack.estimators)["xgb"]
    gpu_config = {
        "xgboost_device": xgb.get_params().get("device"),
        "catboost_task_type": cat.get_params().get("task_type"),
        "catboost_devices": cat.get_params().get("devices"),
        "stacking_xgboost_device": stack_xgb.get_params().get("device"),
    }

    cache_path = project_root / "data" / "processed" / "cicids_clean.parquet"
    cache_columns: list[str] = []
    cache_error: str | None = None
    ratio_detail: dict[str, Any] | None = None
    ratio_error: str | None = None
    try:
        import pyarrow.parquet as pq

        cache_columns = pq.ParquetFile(cache_path).schema.names
    except Exception as exc:  # noqa: BLE001
        cache_error = str(exc)
    if cache_error is None:
        try:
            ratio_detail = _source_held_ratio(cache_path, manifest)
        except Exception as exc:  # noqa: BLE001
            ratio_error = str(exc)

    disk = shutil.disk_usage(project_root)
    checks = {
        "python_version": _check(sys.version_info >= (3, 10), platform.python_version()),
        "manifest": _check(True, manifest["version"]),
        "raw_sources": _check(
            not missing_sources and not empty_sources,
            {"expected": len(source_names), "missing": missing_sources, "empty": empty_sources},
        ),
        "clean_cache_provenance": _check(
            cache_error is None and {"source_file", "_row_hash"}.issubset(cache_columns),
            cache_error or {"path": str(cache_path), "columns": len(cache_columns)},
        ),
        "source_held_ratio": _check(
            ratio_error is None
            and ratio_detail is not None
            and not ratio_detail["unknown_sources"]
            and ratio_detail["train_rows"] == manifest.get("expected_train_rows")
            and ratio_detail["test_rows"] == manifest.get("expected_test_rows")
            and abs(ratio_detail["train_fraction"] - 0.70) <= 1e-6,
            ratio_error or ratio_detail,
        ),
        "python_packages": _check(not missing_packages, {"versions": versions, "missing": missing_packages}),
        "slurm_directives": _check(
            all(token in sbatch for token in REQUIRED_SBATCH_DIRECTIVES),
            list(REQUIRED_SBATCH_DIRECTIVES),
        ),
        "manifest_forwarding": _check(
            'main.py --stage audit --split-manifest "${MANIFEST}"' in sbatch
            and '--split-manifest "${MANIFEST}"' in sbatch,
            "audit and training receive the same manifest",
        ),
        "gpu_pipeline_configuration": _check(
            gpu_config == {
                "xgboost_device": "cuda",
                "catboost_task_type": "GPU",
                "catboost_devices": "0",
                "stacking_xgboost_device": "cuda",
            },
            gpu_config,
        ),
    }
    return {
        "preflight_version": "dgx-preflight-v1",
        "mode": "static",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "project_root": str(project_root),
        "manifest": str(manifest_path),
        "environment": {
            "platform": platform.platform(),
            "disk_free_gib": round(disk.free / 1024**3, 2),
        },
        "checks": checks,
        "passed": all(item["passed"] for item in checks.values()),
        "claim": "Configuration preflight only; this does not prove DGX hardware execution.",
    }


def run_gpu_acceptance(gpu_devices: str) -> dict[str, Any]:
    """Exercise the installed XGBoost and CatBoost CUDA backends on a tiny dataset."""
    checks: dict[str, dict[str, Any]] = {}
    try:
        probe = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, check=True, timeout=30
        )
        checks["nvidia_smi"] = _check(True, probe.stdout.strip())
    except (OSError, subprocess.SubprocessError) as exc:
        checks["nvidia_smi"] = _check(False, str(exc))

    import numpy as np
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier

    rng = np.random.default_rng(42)
    X = rng.normal(size=(96, 8))
    y = np.repeat(np.arange(3), 32)
    models = {
        "xgboost_cuda_fit": XGBClassifier(
            n_estimators=2, max_depth=2, tree_method="hist", device="cuda",
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        ),
        "catboost_gpu_fit": CatBoostClassifier(
            iterations=2, depth=2, task_type="GPU", devices=gpu_devices,
            loss_function="MultiClass", verbose=False, allow_writing_files=False,
        ),
    }
    for name, model in models.items():
        try:
            model.fit(X, y)
            predictions = model.predict(X)
            checks[name] = _check(len(predictions) == len(y), f"predictions={len(predictions)}")
        except Exception as exc:  # noqa: BLE001
            checks[name] = _check(False, f"{type(exc).__name__}: {exc}")
    return {"checks": checks, "passed": all(item["passed"] for item in checks.values())}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DGX delivery preflight and GPU acceptance")
    parser.add_argument("--mode", choices=("static", "gpu"), default="static")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "configs" / "splits" / "source_holdout_v3_full_70_30.json",
    )
    parser.add_argument("--gpu-devices", default="0")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_static_preflight(PROJECT_ROOT, args.manifest)
    if args.mode == "gpu":
        report["mode"] = "gpu"
        report["gpu_acceptance"] = run_gpu_acceptance(args.gpu_devices)
        report["passed"] = report["passed"] and report["gpu_acceptance"]["passed"]
        report["claim"] = "GPU backend acceptance executed on this host; retain this report with scheduler logs."
    rendered = json_dumps_strict(report, indent=2) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"DGX preflight report: {output}")
    print(json.dumps({
        "mode": report["mode"],
        "passed": report["passed"],
        "checks": {name: item["passed"] for name, item in report["checks"].items()},
    }, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
