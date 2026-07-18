"""Integrity-checked model bundle helpers."""

from src.artifacts.bundle import (
    ArtifactIntegrityError,
    build_bundle_manifest,
    sha256_file,
    verify_bundle_manifest,
)
from src.artifacts.paths import result_run_dir
from src.artifacts.publish import promote_run, select_champion_model

__all__ = [
    "ArtifactIntegrityError",
    "build_bundle_manifest",
    "sha256_file",
    "verify_bundle_manifest",
    "promote_run",
    "result_run_dir",
    "select_champion_model",
]
