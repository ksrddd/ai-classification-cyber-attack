"""Small I/O helpers shared by the pipelines.

Keep this module narrow. Heavy persistence (parquet, model files) goes here
so callers don't import joblib/pandas just for file plumbing.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if missing; return it for chaining."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_joblib(obj: Any, path: Path) -> Path:
    """Persist ``obj`` via joblib. Parent directory is created if missing."""
    ensure_dir(path.parent)
    joblib.dump(obj, path)
    logger.info("Saved joblib artifact -> %s", path)
    return path


def load_joblib(path: Path, *, expected_sha256: str | None = None) -> Any:
    """Load a joblib artifact. Raises ``FileNotFoundError`` with a clear msg."""
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}. "
            "Run the training pipeline first (`python main.py --stage train`)."
        )
    if expected_sha256 is not None:
        import hashlib

        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise ValueError(
                f"Artifact integrity check failed for {path}: "
                f"expected {expected_sha256}, got {actual}"
            )
    logger.debug("Loading joblib artifact <- %s", path)
    _install_training_pickle_aliases()
    return joblib.load(path)


def json_dumps_strict(value: Any, **kwargs: Any) -> str:
    """Serialize JSON after replacing non-finite floats with ``null``."""
    return json.dumps(_json_safe(value), allow_nan=False, **kwargs)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    return value


def _install_training_pickle_aliases() -> None:
    """Expose train.py estimator classes for artefacts pickled as __main__."""
    try:
        from train import BalancedXGBClassifier, _LGBMNoFeatureNamesCheck
    except Exception as exc:  # pragma: no cover - best-effort compatibility shim
        logger.debug("Could not install training pickle aliases: %s", exc)
        return

    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        return
    main_mod.BalancedXGBClassifier = BalancedXGBClassifier  # type: ignore[attr-defined]
    main_mod._LGBMNoFeatureNamesCheck = _LGBMNoFeatureNamesCheck  # type: ignore[attr-defined]
