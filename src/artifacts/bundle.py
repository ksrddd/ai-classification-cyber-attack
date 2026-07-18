"""Self-contained model bundle manifests and integrity checks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class ArtifactIntegrityError(RuntimeError):
    """Raised when a persisted artifact cannot be trusted."""


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file incrementally so large model files do not fill RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_bundle_manifest(
    root: Path,
    files: Iterable[Path],
    *,
    run_id: str,
    schema_version: str = "1",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a manifest containing relative paths, sizes and SHA-256 hashes."""
    root = root.resolve()
    entries: dict[str, dict[str, Any]] = {}
    for candidate in files:
        path = candidate.resolve()
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Artifact is outside bundle root: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        key = relative.as_posix()
        entries[key] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "bundle_version": "1",
        "run_id": run_id,
        "schema_version": schema_version,
        "files": dict(sorted(entries.items())),
        "metadata": metadata or {},
    }


def verify_bundle_manifest(root: Path, manifest: dict[str, Any]) -> None:
    """Verify every manifest entry before any model deserialization occurs."""
    root = root.resolve()
    if not manifest.get("run_id") or manifest.get("bundle_version") != "1":
        raise ArtifactIntegrityError("Unsupported or incomplete artifact manifest")
    for relative, expected in manifest.get("files", {}).items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ArtifactIntegrityError(f"Manifest path escapes bundle root: {relative}") from exc
        if not path.is_file():
            raise ArtifactIntegrityError(f"Missing bundle artifact: {relative}")
        if path.stat().st_size != int(expected.get("size", -1)):
            raise ArtifactIntegrityError(f"Artifact size mismatch: {relative}")
        actual = sha256_file(path)
        if actual != expected.get("sha256"):
            raise ArtifactIntegrityError(f"Artifact hash mismatch: {relative}")


def write_bundle_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
