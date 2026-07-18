from __future__ import annotations

import json

import pytest

from src.artifacts.bundle import (
    ArtifactIntegrityError,
    build_bundle_manifest,
    verify_bundle_manifest,
    write_bundle_manifest,
)
from src.artifacts.paths import result_run_dir
from src.artifacts.publish import promote_run


def test_bundle_manifest_verifies_and_rejects_tampering(tmp_path) -> None:
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"safe-model")
    manifest = build_bundle_manifest(tmp_path, [artifact], run_id="run-1")
    verify_bundle_manifest(tmp_path, manifest)
    artifact.write_bytes(b"tampered")
    with pytest.raises(ArtifactIntegrityError, match="mismatch"):
        verify_bundle_manifest(tmp_path, manifest)


def test_manifest_rejects_path_escape(tmp_path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    manifest = {
        "bundle_version": "1",
        "run_id": "run-1",
        "files": {"..\\outside.txt": {"size": 1, "sha256": "x"}},
    }
    with pytest.raises(ArtifactIntegrityError):
        verify_bundle_manifest(tmp_path, manifest)


def test_promote_uses_canonical_results_run_and_writes_champion(tmp_path) -> None:
    run_dir = result_run_dir("run-1", results_root=tmp_path / "results")
    run_dir.mkdir(parents=True)
    artifact = run_dir / "model.joblib"
    artifact.write_bytes(b"safe-model")
    metrics = run_dir / "metrics.json"
    metrics.write_text(json.dumps({"models": [
        {"model": "model", "target_fpr": 0.03, "f1_macro": 0.9},
        {"model": "other", "target_fpr": 0.04, "f1_macro": 0.95},
    ]}), encoding="utf-8")
    manifest = build_bundle_manifest(run_dir, [artifact, metrics], run_id="run-1")
    write_bundle_manifest(run_dir / "bundle_manifest.json", manifest)

    champion = tmp_path / "results" / "champion.json"
    promote_run(run_dir, champion)

    payload = json.loads(champion.read_text(encoding="utf-8"))
    assert payload["run_id"] == "run-1"
    assert payload["champion_model"] == "model"
    assert payload["selection"]["status"] == "conditional_no_model_meets_fpr"


@pytest.mark.parametrize("run_id", ["../escape", "nested/run", "", "has space"])
def test_result_run_dir_rejects_path_components(tmp_path, run_id) -> None:
    with pytest.raises(ValueError):
        result_run_dir(run_id, results_root=tmp_path / "results")
