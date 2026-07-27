"""GPU acceptance check for the accelerated training path.

Only three models consult ``--accelerator``: XGBoost (``device="cuda"``),
CatBoost (``task_type="GPU"``), and the stacking ensemble, which embeds an
XGBoost base learner. Everything else is CPU-only because scikit-learn has no
CUDA backend for it.

This check exists because the project forbids a silent CPU fallback. A CUDA
build that is present but unusable -- wrong driver, no device visible, a
mismatched toolkit -- would otherwise degrade quietly into a CPU run whose
reported ``accelerator`` says ``gpu``, and every timing in the report would be
a lie. Failing loudly before training starts is the point; do not "fix" it into
a warning.
"""

from __future__ import annotations

import subprocess
from typing import Any


def _check(ok: bool, detail: Any) -> dict[str, Any]:
    return {"passed": bool(ok), "detail": detail}


def run_gpu_acceptance(gpu_devices: str = "0") -> dict[str, Any]:
    """Exercise the installed XGBoost and CatBoost CUDA backends.

    Fits a deliberately tiny 3-class problem on the device, so the check costs
    a second or two but still proves the whole path works end to end rather
    than merely that a library imports.

    Returns ``{"checks": {...}, "passed": bool}``. Callers are expected to
    abort on ``passed is False``.
    """
    checks: dict[str, dict[str, Any]] = {}
    try:
        probe = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True, text=True, check=True, timeout=30,
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
            checks[name] = _check(True, "CUDA fit completed")
        except Exception as exc:  # noqa: BLE001
            checks[name] = _check(False, f"{type(exc).__name__}: {exc}")

    return {
        "checks": checks,
        "passed": all(item["passed"] for item in checks.values()),
    }
