"""Atomic, metadata-bound checkpoints for long-running local model training."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

CHECKPOINT_VERSION = 1


def load_checkpoint(path: Path) -> dict[str, Any] | None:
    """Return a checkpoint or ``None`` when it does not exist or is invalid."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        return None
    return payload


def checkpoint_matches(
    checkpoint: dict[str, Any] | None,
    *,
    model_name: str,
    run_signature: str,
) -> bool:
    """Ensure a resumable state belongs to this exact model/data/config run."""
    return bool(
        checkpoint
        and checkpoint.get("model") == model_name
        and checkpoint.get("run_signature") == run_signature
    )


def write_checkpoint(path: Path, payload: dict[str, Any]) -> Path:
    """Atomically persist JSON checkpoint state after each recoverable phase."""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"checkpoint_version": CHECKPOINT_VERSION, **payload}
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)
    return path
