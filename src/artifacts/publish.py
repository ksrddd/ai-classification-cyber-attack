"""Atomic promotion of an integrity-checked model run."""

from __future__ import annotations

import json
import operator as _operator
import os
from pathlib import Path

from src.artifacts.bundle import ArtifactIntegrityError, verify_bundle_manifest

_COMPARATORS = {
    "<=": _operator.le,
    ">=": _operator.ge,
    "<": _operator.lt,
    ">": _operator.gt,
}

DEFAULT_RANKING_POLICY_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "ranking_policy.json"
)


def load_ranking_policy(path: Path | str | None = None) -> dict:
    """Load the declared ranking policy.

    The thresholds live in JSON rather than in code so they are fixed before a
    run and visible to a reader, instead of being adjustable after the results
    are known.
    """
    return json.loads(
        Path(path or DEFAULT_RANKING_POLICY_PATH).read_text(encoding="utf-8")
    )


def _as_float(item: dict, key: str) -> float | None:
    value = item.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _min_per_class_recall(item: dict) -> float | None:
    """Worst per-class recall, or ``None`` when the model did not report it."""
    per_class = item.get("per_class_recall")
    if not isinstance(per_class, dict) or not per_class:
        return None
    try:
        return min(float(value) for value in per_class.values())
    except (TypeError, ValueError):
        return None


def _metric(item: dict, name: str) -> float | None:
    if name == "min_per_class_recall":
        return _min_per_class_recall(item)
    return _as_float(item, name)


def select_rankings(
    run_dir: Path,
    *,
    policy: dict | None = None,
) -> dict[str, dict]:
    """Rank a completed run's models on three independent axes."""
    return rank_models(_load_models(run_dir), policy or load_ranking_policy())


def rank_models(models: list[dict], policy: dict) -> dict[str, dict]:
    """Rank model metric dicts on every axis the policy declares.

    Pure function over already-loaded payloads, so the trainer can rank
    in-memory results while writing its report and the promotion step can rank
    the same models straight from a persisted ``metrics.json``.

    Returns one entry per ranking. Each names the winner, the rule that
    produced it, and -- when a constraint excluded every model -- says so
    explicitly and falls back to the unconstrained winner rather than
    silently reporting nothing.
    """
    tolerance = float(policy.get("quality_tolerance", 0.02))
    if not models:
        raise ArtifactIntegrityError("No model results to rank")
    best_f1 = max((_as_float(item, "f1_macro") or 0.0) for item in models)

    results: dict[str, dict] = {}
    for name, spec in policy["rankings"].items():
        objective = spec.get("maximize") or spec.get("minimize")
        maximise = "maximize" in spec

        eligible = []
        unmet: list[str] = []
        for item in models:
            if _metric(item, objective) is None:
                continue
            satisfied = True
            for constraint in spec.get("constraints", []):
                metric_name = constraint["metric"]
                observed = _metric(item, metric_name)
                if observed is None:
                    satisfied = False
                    unmet.append(f"{item.get('model')}: {metric_name} not reported")
                    break
                limit = constraint["value"]
                # The deployment constraint is expressed relative to the best
                # observed f1_macro, which is only knowable after the run.
                if isinstance(limit, str):
                    limit = best_f1 - tolerance
                limit = float(limit)
                operator = constraint["operator"]
                if operator not in _COMPARATORS:
                    raise ValueError(
                        f"Unknown constraint operator {operator!r} in ranking "
                        f"{name!r}. Choices: {', '.join(sorted(_COMPARATORS))}"
                    )
                satisfied = _COMPARATORS[operator](observed, limit)
                if not satisfied:
                    unmet.append(
                        f"{item.get('model')}: {metric_name}={observed:.6g} "
                        f"fails {operator} {limit:.6g}"
                    )
                    break
            if satisfied:
                eligible.append(item)

        pool = eligible
        status = "policy_pass"
        if not pool:
            pool = [item for item in models if _metric(item, objective) is not None]
            status = "conditional_no_model_meets_constraints"
        if not pool:
            results[name] = {
                "label": spec.get("label", name),
                "model": None,
                "status": "unavailable_metric_missing",
                "rule": spec.get("rule", ""),
                "objective": objective,
            }
            continue

        tie_breaker = spec.get("tie_breaker")
        sign = -1.0 if maximise else 1.0

        def sort_key(item: dict, *, sign: float = sign,
                     objective: str = objective,
                     tie_breaker: str | None = tie_breaker) -> tuple[float, float]:
            primary = sign * (_metric(item, objective) or 0.0)
            secondary = -(_as_float(item, tie_breaker) or 0.0) if tie_breaker else 0.0
            return (primary, secondary)

        chosen = min(pool, key=sort_key)
        results[name] = {
            "label": spec.get("label", name),
            "question": spec.get("question", ""),
            "model": str(chosen["model"]),
            "status": status,
            "rule": spec.get("rule", ""),
            "objective": objective,
            "objective_value": _metric(chosen, objective),
            "tie_breaker": tie_breaker,
            "eligible_models": [str(item["model"]) for item in eligible],
            "excluded": unmet,
        }
    return results


def _load_models(run_dir: Path) -> list[dict]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        raise ArtifactIntegrityError(f"Missing aggregate metrics: {metrics_path}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        raise ArtifactIntegrityError("Aggregate metrics contain no model results")
    return models


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
        # The auto-selected model's numbers would be misleading next to a
        # different, operator-chosen model.
        selection.pop("target_fpr", None)
        selection.pop("f1_macro", None)
    model_name = str(selection["model"])
    if f"{model_name}.joblib" not in manifest.get("files", {}):
        raise ArtifactIntegrityError(f"Champion model is not in the verified bundle: {model_name}")
    try:
        portable_bundle_path = run_dir.relative_to(champion_path.parent.parent).as_posix()
    except ValueError:
        portable_bundle_path = str(run_dir)
    # Best-effort: a run whose metrics predate the three-axis policy still
    # promotes, it just carries no rankings block.
    try:
        rankings = select_rankings(run_dir)
    except (ArtifactIntegrityError, OSError, ValueError, KeyError):
        rankings = {}
    payload = {
        "run_id": manifest["run_id"],
        "bundle_path": portable_bundle_path,
        "schema_version": manifest.get("schema_version", "1"),
        "champion_model": model_name,
        "selection": selection,
        "rankings": rankings,
    }
    champion_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = champion_path.with_suffix(champion_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, champion_path)
    return champion_path
