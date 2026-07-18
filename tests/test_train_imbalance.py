"""Regression tests for the canonical train.py imbalance experiments."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from train import (
    CONFIG,
    _apply_target_ratio,
    _imbalance_config_matches,
    _imbalance_metadata,
    budgeted_train_test_split,
    build_pipeline,
    calibrate_target_threshold,
    parse_args,
)

ALL_MODELS = (
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "mlp",
    "logistic_regression",
    "stacking",
)
ALL_STRATEGIES = (
    "class_weight",
    "targeted",
    "random_over",
    "borderline_smote",
    "smoteenn",
)


def _toy_flows() -> pd.DataFrame:
    labels = (
        ["BENIGN"] * 1_000
        + ["Infiltration"] * 100
        + ["Web Attack"] * 100
    )
    return pd.DataFrame({
        "row_id": np.arange(len(labels)),
        "feature": np.linspace(0.0, 1.0, len(labels)),
        "Label": labels,
    })


def test_targeted_real_sampling_keeps_natural_test_untouched() -> None:
    df = _toy_flows()
    common = dict(
        label_col="Label",
        total_budget=300,
        test_size=0.2,
        min_test_per_class=3,
        rare_threshold=5,
        target_class="Infiltration",
        target_ratio=0.20,
        random_state=42,
    )
    natural_train, natural_test = budgeted_train_test_split(
        df, train_sampling="natural", **common,
    )
    targeted_train, targeted_test = budgeted_train_test_split(
        df, train_sampling="targeted", **common,
    )

    assert len(natural_train) == len(targeted_train) == 240
    assert len(natural_test) == len(targeted_test) == 60
    assert natural_test["Label"].value_counts().to_dict() == (
        targeted_test["Label"].value_counts().to_dict()
    )
    assert set(targeted_train["row_id"]).isdisjoint(targeted_test["row_id"])

    train_counts = targeted_train["Label"].value_counts()
    assert train_counts["Infiltration"] / train_counts["BENIGN"] >= 0.20


def test_targeted_ratio_one_keeps_all_available_genuine_target_rows() -> None:
    train, test = budgeted_train_test_split(
        _toy_flows(),
        label_col="Label",
        total_budget=300,
        test_size=0.2,
        min_test_per_class=3,
        rare_threshold=5,
        train_sampling="targeted",
        target_class="Infiltration",
        target_ratio=1.0,
        random_state=42,
    )

    assert (test["Label"] == "Infiltration").sum() == 5
    assert (train["Label"] == "Infiltration").sum() == 95


def test_threshold_calibration_holdout_stays_natural_and_disjoint() -> None:
    common = dict(
        label_col="Label",
        total_budget=300,
        test_size=0.2,
        calibration_size=0.2,
        min_test_per_class=3,
        rare_threshold=5,
        target_class="Infiltration",
        target_ratio=1.0,
        random_state=42,
    )
    natural_train, natural_calibration, natural_test = (
        budgeted_train_test_split(
            _toy_flows(),
            train_sampling="natural",
            **common,
        )
    )
    targeted_train, targeted_calibration, targeted_test = (
        budgeted_train_test_split(
            _toy_flows(),
            train_sampling="targeted",
            **common,
        )
    )

    assert natural_calibration["Label"].value_counts().to_dict() == (
        targeted_calibration["Label"].value_counts().to_dict()
    )
    assert natural_test["Label"].value_counts().to_dict() == (
        targeted_test["Label"].value_counts().to_dict()
    )
    selected = [
        set(targeted_train["row_id"]),
        set(targeted_calibration["row_id"]),
        set(targeted_test["row_id"]),
    ]
    assert selected[0].isdisjoint(selected[1])
    assert selected[0].isdisjoint(selected[2])
    assert selected[1].isdisjoint(selected[2])
    assert sum(map(len, (natural_train, natural_calibration, natural_test))) == 300
    assert sum(map(len, (targeted_train, targeted_calibration, targeted_test))) == 300


def test_target_ratio_uses_current_largest_non_target_class() -> None:
    quotas = {"BENIGN": 100, "DoS": 90, "Infiltration": 10}
    available = {name: 1_000 for name in quotas}

    adjusted = _apply_target_ratio(
        quotas,
        available,
        target_class="Infiltration",
        target_ratio=1.0,
    )

    largest_non_target = max(
        count for name, count in adjusted.items() if name != "Infiltration"
    )
    assert sum(adjusted.values()) == sum(quotas.values())
    assert adjusted["Infiltration"] / largest_non_target >= 1.0


def test_artifact_reuse_requires_matching_dataset_fingerprint() -> None:
    cfg = dict(CONFIG)
    cfg["data_fingerprint"] = "dataset-a"
    saved = _imbalance_metadata(cfg)

    assert _imbalance_config_matches(saved, cfg)
    cfg["data_fingerprint"] = "dataset-b"
    assert not _imbalance_config_matches(saved, cfg)


def test_target_threshold_calibration_balances_fp_and_fn_under_fpr_cap() -> None:
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    target_probability = np.array([0.90, 0.80, 0.40, 0.30, 0.70, 0.20, 0.10, 0.05])

    calibrated = calibrate_target_threshold(
        y_true,
        target_probability,
        target_class_index=1,
        max_false_positive_rate=0.25,
    )

    assert calibrated.threshold == pytest.approx(0.30)
    assert calibrated.recall == 1.0
    assert calibrated.false_positive_rate == 0.25
    assert calibrated.false_negatives == 0
    assert calibrated.false_positives == 1


def test_target_threshold_calibration_does_not_spend_fpr_for_recall_alone() -> None:
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    target_probability = np.array([
        0.90, 0.80, 0.70, 0.10,
        0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30,
    ])

    calibrated = calibrate_target_threshold(
        y_true,
        target_probability,
        target_class_index=1,
        max_false_positive_rate=1.0,
    )

    assert calibrated.threshold == pytest.approx(0.70)
    assert calibrated.false_positives == 0
    assert calibrated.false_negatives == 1


def test_target_threshold_fails_when_fpr_policy_is_unattainable() -> None:
    y_true = np.array([1, 1, 0, 0, 0, 0])
    target_probability = np.array([0.80, 0.70, 0.99, 0.98, 0.20, 0.10])

    with pytest.raises(ValueError, match="satisfies max FPR"):
        calibrate_target_threshold(
            y_true,
            target_probability,
            target_class_index=1,
            max_false_positive_rate=0.0,
        )


def test_target_threshold_pipeline_applies_saved_decision_rule(tmp_path) -> None:
    rng = np.random.default_rng(42)
    y = np.array([0] * 60 + [1] * 30 + [2] * 30)
    X = pd.DataFrame(rng.normal(size=(len(y), 3)), columns=list("abc"))
    pipeline = build_pipeline(
        "random_forest",
        n_classes=3,
        random_state=42,
        imbalance_strategy="targeted",
        target_class_index=2,
        target_ratio=1.0,
    )
    pipeline.set_params(clf__n_estimators=5, clf__n_jobs=1)
    pipeline.fit(X, y)
    pipeline.set_params(target_threshold=0.0)

    assert np.all(pipeline.predict(X.iloc[:10]) == 2)
    path = tmp_path / "threshold_pipeline.joblib"
    joblib.dump(pipeline, path)
    restored = joblib.load(path)
    assert np.all(restored.predict(X.iloc[:10]) == 2)


def test_random_over_sampler_is_inside_pipeline_and_disables_double_weighting() -> None:
    rng = np.random.default_rng(42)
    y = np.array([0] * 120 + [1] * 30 + [2] * 12)
    X = pd.DataFrame(rng.normal(size=(len(y), 4)), columns=list("abcd"))

    pipeline = build_pipeline(
        "random_forest",
        n_classes=3,
        random_state=42,
        imbalance_strategy="random_over",
        target_class_index=2,
        target_ratio=0.20,
    )
    pipeline.set_params(clf__n_estimators=5)
    pipeline.fit(X, y)

    assert list(pipeline.named_steps) == ["imputer", "scaler", "sampler", "clf"]
    assert pipeline.named_steps["clf"].class_weight is None
    assert len(pipeline.predict(X.iloc[:3])) == 3


def test_class_weight_baseline_retains_model_weighting() -> None:
    pipeline = build_pipeline(
        "random_forest",
        n_classes=3,
        random_state=42,
        imbalance_strategy="class_weight",
    )
    assert list(pipeline.named_steps) == ["imputer", "scaler", "clf"]
    assert pipeline.named_steps["clf"].class_weight == "balanced_subsample"


def test_parse_imbalance_cli_options() -> None:
    args = parse_args([
        "--imbalance-strategy", "borderline_smote",
        "--target-class", "Infiltration",
        "--target-ratio", "0.25",
        "--target-max-fpr", "0.03",
        "--threshold-validation-size", "0.15",
    ])
    assert args["imbalance_strategy"] == "borderline_smote"
    assert args["target_class"] == "Infiltration"
    assert args["target_ratio"] == 0.25
    assert args["target_max_fpr"] == 0.03
    assert args["threshold_validation_size"] == 0.15


def test_parse_all_models_and_aliases() -> None:
    assert parse_args(["--models", "all"])["models"] == ALL_MODELS
    assert parse_args(["--models", "rf,cat,nn,lr,stack"])["models"] == (
        "random_forest",
        "catboost",
        "mlp",
        "logistic_regression",
        "stacking",
    )


@pytest.mark.parametrize("model_name", ALL_MODELS)
@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_every_model_builds_with_every_imbalance_strategy(
    model_name: str,
    strategy: str,
) -> None:
    pipeline = build_pipeline(
        model_name,
        n_classes=3,
        random_state=42,
        imbalance_strategy=strategy,
        target_class_index=2,
        target_ratio=0.20,
    )
    expected_sampler = strategy in {"random_over", "borderline_smote", "smoteenn"}
    assert ("sampler" in pipeline.named_steps) is expected_sampler


def _small_model_params(model_name: str) -> dict[str, object]:
    if model_name == "random_forest":
        return {"clf__n_estimators": 5, "clf__n_jobs": 1}
    if model_name == "xgboost":
        return {"clf__n_estimators": 5, "clf__max_depth": 2, "clf__n_jobs": 1}
    if model_name == "lightgbm":
        return {"clf__n_estimators": 5, "clf__num_leaves": 7, "clf__n_jobs": 1}
    if model_name == "catboost":
        return {"clf__iterations": 5, "clf__depth": 2, "clf__thread_count": 1}
    if model_name == "mlp":
        return {
            "clf__hidden_layer_sizes": (8,),
            "clf__max_iter": 20,
            "clf__early_stopping": False,
            "clf__tol": 1e9,
            "clf__n_iter_no_change": 1,
        }
    if model_name == "logistic_regression":
        return {"clf__max_iter": 100}
    return {
        "clf__cv": 2,
        "clf__lgbm__n_estimators": 3,
        "clf__lgbm__n_jobs": 1,
        "clf__xgb__n_estimators": 3,
        "clf__xgb__max_depth": 2,
        "clf__xgb__n_jobs": 1,
        "clf__rf__n_estimators": 3,
        "clf__rf__n_jobs": 1,
        "clf__final_estimator__max_iter": 100,
    }


@pytest.mark.parametrize("model_name", ALL_MODELS)
@pytest.mark.parametrize("strategy", ["class_weight", "random_over"])
def test_every_model_fits_weighted_and_resampled_data(
    model_name: str,
    strategy: str,
) -> None:
    rng = np.random.default_rng(7)
    y = np.array([0] * 90 + [1] * 45 + [2] * 30)
    X = pd.DataFrame(rng.normal(size=(len(y), 6)), columns=list("abcdef"))
    X.loc[X.index[::17], "a"] = np.nan

    pipeline = build_pipeline(
        model_name,
        n_classes=3,
        random_state=42,
        imbalance_strategy=strategy,
        target_class_index=2,
        target_ratio=0.50,
    )
    pipeline.set_params(**_small_model_params(model_name))
    pipeline.fit(X, y)
    predictions = np.asarray(pipeline.predict(X.iloc[:3]))

    assert predictions.shape == (3,)
