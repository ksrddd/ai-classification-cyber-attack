"""Atomic promotion of an integrity-checked model run."""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.artifacts.bundle import ArtifactIntegrityError, verify_bundle_manifest


def select_champion_model(
    run_dir: Path,
    *,
    target_max_fpr: float = 0.02,
) -> dict[str, object]:
    """Select transparently: policy-compliant macro-F1 winner, or lowest-FPR fallback."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        raise ArtifactIntegrityError(f"Missing aggregate metrics: {metrics_path}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        raise ArtifactIntegrityError("Aggregate metrics contain no model results")
    eligible = [
        item for item in models
        if item.get("target_fpr") is not None and float(item["target_fpr"]) <= target_max_fpr
    ]
    if eligible:
        chosen = max(eligible, key=lambda item: float(item.get("f1_macro") or 0.0))
        status = "policy_pass"
        rule = "highest f1_macro among models with target_fpr <= target_max_fpr"
    else:
        candidates = [item for item in models if item.get("target_fpr") is not None]
        if not candidates:
            raise ArtifactIntegrityError("No model has a finite target_fpr")
        chosen = min(
            candidates,
            key=lambda item: (
                float(item["target_fpr"]),
                -float(item.get("f1_macro") or 0.0),
            ),
        )
        status = "conditional_no_model_meets_fpr"
        rule = "lowest target_fpr, then highest f1_macro"
    return {
        "model": str(chosen["model"]),
        "status": status,
        "rule": rule,
        "target_max_fpr": target_max_fpr,
        "target_fpr": float(chosen["target_fpr"]),
        "f1_macro": float(chosen.get("f1_macro") or 0.0),
    }


def promote_run(
    run_dir: Path,
    champion_path: Path,
    *,
    champion_model: str | None = None,
    target_max_fpr: float = 0.02,
) -> Path:
    """Verify a run and atomically update the champion pointer."""
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "bundle_manifest.json"
    if not manifest_path.exists():
        raise ArtifactIntegrityError(f"Missing bundle manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_bundle_manifest(run_dir, manifest)
    if manifest.get("run_id") != run_dir.name:
        raise ArtifactIntegrityError(
            f"Manifest run_id {manifest.get('run_id')!r} does not match directory {run_dir.name!r}"
        )
    selection = select_champion_model(run_dir, target_max_fpr=target_max_fpr)
    if champion_model is not None:
        selection["model"] = champion_model
        selection["rule"] = "explicit operator selection"
        selection["status"] = "explicit_selection"
    model_name = str(selection["model"])
    if f"{model_name}.joblib" not in manifest.get("files", {}):
        raise ArtifactIntegrityError(f"Champion model is not in the verified bundle: {model_name}")
    try:
        portable_bundle_path = run_dir.relative_to(champion_path.parent.parent).as_posix()
    except ValueError:
        portable_bundle_path = str(run_dir)
    payload = {
        "run_id": manifest["run_id"],
        "bundle_path": portable_bundle_path,
        "schema_version": manifest.get("schema_version", "1"),
        "champion_model": model_name,
        "selection": selection,
    }
    champion_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = champion_path.with_suffix(champion_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, champion_path)
    return champion_path
