"""Tests for the three-axis model ranking policy.

The point of publishing three rankings is that they can disagree. These tests
build a field where they *do* disagree and pin each winner to its declared rule.
"""

from __future__ import annotations

import json

import pytest

import train
from src.artifacts.publish import (
    load_ranking_policy,
    rank_models,
    select_rankings,
)


def _model(name: str, **overrides) -> dict:
    base = {
        "model": name,
        "f1_macro": 0.90,
        "f1_weighted": 0.92,
        "recall_macro": 0.90,
        "binary_recall": 0.95,
        "binary_fpr": 0.001,
        "predict_latency_p95_ms": 50.0,
        "model_size_mb": 10.0,
        "per_class_recall": {"BENIGN": 0.99, "DoS": 0.9, "Bot": 0.8},
    }
    base.update(overrides)
    return base


# A field where every axis has a different winner:
#   rf   -- best macro-F1, but slow and huge
#   xgb  -- best macro recall, but 5% binary FPR (alert fatigue)
#   cat  -- competitive and quiet, but never detects Bot at all
#   lgbm -- slightly behind rf, far faster
#   lr   -- fastest, but well below the quality floor
FIELD = [
    _model("rf", f1_macro=0.93, f1_weighted=0.95, recall_macro=0.90,
           predict_latency_p95_ms=120.0, model_size_mb=800.0),
    _model("xgb", f1_macro=0.92, recall_macro=0.97, binary_fpr=0.05,
           predict_latency_p95_ms=20.0, model_size_mb=30.0,
           per_class_recall={"BENIGN": 0.95, "DoS": 0.99, "Bot": 0.95}),
    _model("cat", f1_macro=0.925, recall_macro=0.94, binary_fpr=0.002,
           predict_latency_p95_ms=45.0, model_size_mb=12.0,
           per_class_recall={"BENIGN": 0.99, "DoS": 0.99, "Bot": 0.0}),
    _model("lgbm", f1_macro=0.915, recall_macro=0.89, binary_fpr=0.003,
           predict_latency_p95_ms=25.0, model_size_mb=20.0),
    _model("lr", f1_macro=0.80, recall_macro=0.78, binary_fpr=0.004,
           predict_latency_p95_ms=3.0, model_size_mb=0.1),
]


@pytest.fixture
def policy() -> dict:
    return load_ranking_policy()


def test_shipped_policy_declares_three_rankings(policy) -> None:
    assert set(policy["rankings"]) == {"overall", "security", "deployment"}
    for spec in policy["rankings"].values():
        assert spec["label"]
        assert spec["rule"]
        # Every constraint states why it exists, for the write-up.
        for constraint in spec.get("constraints", []):
            assert constraint["reason"]


def test_overall_best_maximises_macro_f1(policy) -> None:
    result = rank_models(FIELD, policy)["overall"]

    assert result["model"] == "rf"
    assert result["status"] == "policy_pass"


def test_security_best_excludes_alert_fatigue_and_blind_spots(policy) -> None:
    """xgb has the best recall but 5% FPR; cat is quiet but never sees Bot."""
    result = rank_models(FIELD, policy)["security"]

    assert result["model"] == "rf"
    assert result["status"] == "policy_pass"
    assert "xgb" not in result["eligible_models"]
    assert "cat" not in result["eligible_models"]
    assert any("binary_fpr" in reason for reason in result["excluded"])
    assert any("min_per_class_recall" in reason for reason in result["excluded"])


def test_deployment_best_is_fastest_above_the_quality_floor(policy) -> None:
    """lr is fastest overall but 0.13 macro-F1 below the best, so it is out."""
    result = rank_models(FIELD, policy)["deployment"]

    assert result["model"] == "xgb"
    assert "lr" not in result["eligible_models"]
    assert any("f1_macro" in reason for reason in result["excluded"])


def test_the_three_rankings_can_disagree(policy) -> None:
    """If they always agreed, publishing three would be pointless."""
    rankings = rank_models(FIELD, policy)
    winners = {name: entry["model"] for name, entry in rankings.items()}

    assert len(set(winners.values())) > 1, winners


def test_falls_back_transparently_when_no_model_qualifies(policy) -> None:
    noisy = [
        _model("a", recall_macro=0.9, binary_fpr=0.20),
        _model("b", recall_macro=0.8, binary_fpr=0.30),
    ]

    result = rank_models(noisy, policy)["security"]

    assert result["status"] == "conditional_no_model_meets_constraints"
    assert result["model"] == "a"          # best unconstrained recall
    assert result["excluded"]              # and it says why


def test_missing_metric_is_reported_not_silently_zero(policy) -> None:
    """Older runs lack the efficiency block; that must not fake a winner."""
    legacy = [
        {"model": "old", "f1_macro": 0.9, "f1_weighted": 0.9},
    ]

    rankings = rank_models(legacy, policy)

    assert rankings["overall"]["model"] == "old"
    assert rankings["deployment"]["model"] is None
    assert rankings["deployment"]["status"] == "unavailable_metric_missing"


def test_empty_field_is_an_error(policy) -> None:
    from src.artifacts.bundle import ArtifactIntegrityError

    with pytest.raises(ArtifactIntegrityError):
        rank_models([], policy)


def test_select_rankings_reads_a_run_directory(tmp_path) -> None:
    (tmp_path / "metrics.json").write_text(
        json.dumps({"models": FIELD}), encoding="utf-8"
    )

    rankings = select_rankings(tmp_path)

    assert rankings["overall"]["model"] == "rf"
    assert rankings["deployment"]["model"] == "xgb"


# ---------------------------------------------------------------------------
# Contract between what train.py writes and what consumers read
# ---------------------------------------------------------------------------
def test_serialised_metrics_cover_every_key_downstream_consumers_need() -> None:
    """Guard against a rename silently degrading promotion or the dashboard.

    ``publish.select_champion_model`` reads ``f1_macro`` and ``target_fpr`` via
    ``.get(...)`` with numeric fallbacks, so a rename would not raise -- it
    would quietly promote the wrong model. This asserts the writer still emits
    every key those consumers depend on.
    """
    import numpy as np

    result = train.EvalResult(
        model="rf",
        accuracy=0.99, balanced_accuracy=0.9, f1_macro=0.9, f1_weighted=0.99,
        per_class=None, confusion=np.zeros((2, 2)),
        extended=train.extended_metrics_from_confusion(
            np.array([[90, 10], [5, 95]]), ["BENIGN", "DoS"],
            benign_class_index=0, reportable_min_test=5,
        ),
        efficiency={
            "fit_seconds": 1.0, "model_size_mb": 2.0,
            "predict_latency_p50_ms": 3.0, "predict_latency_p95_ms": 4.0,
            "throughput_flows_per_sec": 5.0,
        },
    )
    payload = train._eval_result_to_dict(result, train.CONFIG)

    # publish.select_champion_model / select_rankings
    required = {
        "model", "f1_macro", "f1_weighted", "target_fpr",
        "recall_macro", "binary_fpr", "binary_recall", "per_class_recall",
        "predict_latency_p95_ms", "model_size_mb",
    }
    # main.py --stage evaluate
    required |= {
        "accuracy", "target_recall", "target_f2",
        "target_false_negatives", "target_false_positives",
    }
    # scripts/promote_best_models.py (direct subscript -> KeyError on rename)
    required |= {"target_threshold", "imbalance_strategy"}

    assert required <= set(payload), sorted(required - set(payload))


def test_eval_result_survives_a_json_round_trip() -> None:
    """_eval_result_to_dict and _eval_result_from_saved must be inverses."""
    import numpy as np

    original = train.EvalResult(
        model="rf",
        accuracy=0.99, balanced_accuracy=0.9, f1_macro=0.9, f1_weighted=0.99,
        per_class=None, confusion=np.zeros((2, 2)),
        cv_scores=[0.1, 0.2], cv_mean=0.15, cv_std=0.05,
        extended=train.extended_metrics_from_confusion(
            np.array([[90, 10], [5, 95]]), ["BENIGN", "DoS"],
            benign_class_index=0, reportable_min_test=5,
        ),
        efficiency={"fit_seconds": 1.0, "model_size_mb": 2.0,
                    "predict_latency_p95_ms": 4.0},
        hp_space_size=144, hp_tuned=True,
    )

    payload = json.loads(json.dumps(train._eval_result_to_dict(original, train.CONFIG)))
    restored = train._eval_result_from_saved(payload, "rf")

    assert restored.f1_macro == original.f1_macro
    assert restored.cv_mean == original.cv_mean
    assert restored.cv_scores == original.cv_scores
    assert restored.extended == original.extended
    assert restored.efficiency == original.efficiency
    assert restored.hp_space_size == 144
    assert restored.hp_tuned is True
