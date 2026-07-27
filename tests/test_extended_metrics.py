"""Tests for the metrics the evaluation standard requires beyond the headline.

Values are checked against hand-computed numbers and against sklearn, because
these feed the report tables and the ranking policy directly.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    matthews_corrcoef,
    precision_score,
    recall_score,
)

import train

CLASS_NAMES = ["BENIGN", "A", "B"]
BENIGN = 0

# rows = truth, columns = prediction.
#   BENIGN: 95 correct, 3 -> A, 2 -> B      (support 100)
#   A:      4 -> BENIGN, 6 correct          (support 10)
#   B:      1 -> BENIGN, 3 correct          (support 4)
CM = np.array([
    [95, 3, 2],
    [4, 6, 0],
    [1, 0, 3],
])


@pytest.fixture
def metrics() -> dict:
    return train.extended_metrics_from_confusion(
        CM, CLASS_NAMES, benign_class_index=BENIGN, reportable_min_test=5,
    )


def _expand(cm: np.ndarray) -> tuple[list[int], list[int]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            y_true += [i] * int(cm[i, j])
            y_pred += [j] * int(cm[i, j])
    return y_true, y_pred


def test_macro_and_weighted_averages_match_sklearn(metrics) -> None:
    y_true, y_pred = _expand(CM)

    assert metrics["precision_macro"] == pytest.approx(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    assert metrics["recall_macro"] == pytest.approx(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )
    assert metrics["precision_weighted"] == pytest.approx(
        precision_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    assert metrics["recall_weighted"] == pytest.approx(
        recall_score(y_true, y_pred, average="weighted", zero_division=0)
    )


def test_mcc_matches_sklearn(metrics) -> None:
    y_true, y_pred = _expand(CM)
    assert metrics["mcc"] == pytest.approx(matthews_corrcoef(y_true, y_pred))


def test_per_class_fnr_is_one_minus_recall(metrics) -> None:
    # A: 4 of 10 missed. B: 1 of 4 missed. BENIGN: 5 of 100 misrouted.
    assert metrics["per_class_fnr"]["A"] == pytest.approx(0.4)
    assert metrics["per_class_fnr"]["B"] == pytest.approx(0.25)
    assert metrics["per_class_fnr"]["BENIGN"] == pytest.approx(0.05)

    for name in CLASS_NAMES:
        assert metrics["per_class_fnr"][name] == pytest.approx(
            1.0 - metrics["per_class_recall"][name]
        )


def test_per_class_fpr_is_one_vs_rest(metrics) -> None:
    """Negatives for class c are every row whose TRUE label is not c."""
    # A: predicted 3 times wrongly, against 100 + 4 = 104 non-A rows.
    assert metrics["per_class_fpr"]["A"] == pytest.approx(3 / 104)
    # B: predicted 2 times wrongly, against 100 + 10 = 110 non-B rows.
    assert metrics["per_class_fpr"]["B"] == pytest.approx(2 / 110)
    # BENIGN: 4 + 1 = 5 of the 14 attack rows were called BENIGN.
    assert metrics["per_class_fpr"]["BENIGN"] == pytest.approx(5 / 14)


def test_binary_view_collapses_attacks(metrics) -> None:
    """Attack vs BENIGN: FP is a benign flow alerted, FN is a missed attack."""
    # 3 + 2 = 5 BENIGN rows predicted as some attack, out of 100 BENIGN.
    assert metrics["binary_fpr"] == pytest.approx(5 / 100)
    assert metrics["binary_false_positives"] == 5
    # 4 + 1 = 5 attack rows predicted BENIGN, out of 14 attack rows.
    assert metrics["binary_fnr"] == pytest.approx(5 / 14)
    assert metrics["binary_false_negatives"] == 5
    assert metrics["binary_recall"] == pytest.approx(9 / 14)
    assert metrics["binary_benign_support"] == 100
    assert metrics["binary_attack_support"] == 14


def test_reportable_macro_excludes_undersupported_classes(metrics) -> None:
    """B has 4 test rows, below the threshold of 5, so it is excluded."""
    assert metrics["reportable_classes"] == ["BENIGN", "A"]

    f1 = metrics["per_class_f1"]
    expected = (f1["BENIGN"] + f1["A"]) / 2
    assert metrics["f1_macro_reportable"] == pytest.approx(expected)


def test_reportable_macro_equals_plain_macro_when_all_qualify() -> None:
    metrics = train.extended_metrics_from_confusion(
        CM, CLASS_NAMES, benign_class_index=BENIGN, reportable_min_test=1,
    )

    assert metrics["reportable_classes"] == CLASS_NAMES
    assert metrics["f1_macro_reportable"] == pytest.approx(
        float(np.mean(list(metrics["per_class_f1"].values())))
    )


def test_perfect_classifier_reports_zero_error_rates() -> None:
    metrics = train.extended_metrics_from_confusion(
        np.diag([100, 10, 4]), CLASS_NAMES,
        benign_class_index=BENIGN, reportable_min_test=1,
    )

    assert metrics["binary_fpr"] == 0.0
    assert metrics["binary_fnr"] == 0.0
    assert metrics["mcc"] == pytest.approx(1.0)
    assert all(v == pytest.approx(1.0) for v in metrics["per_class_recall"].values())


def test_benign_only_predictor_exposes_what_accuracy_hides() -> None:
    """The imbalance argument, as a test.

    A model that always answers BENIGN scores 100/114 = 87.7% accuracy while
    detecting nothing. Macro recall and binary FNR must both expose that.
    """
    cm = np.array([[100, 0, 0], [10, 0, 0], [4, 0, 0]])

    metrics = train.extended_metrics_from_confusion(
        cm, CLASS_NAMES, benign_class_index=BENIGN, reportable_min_test=1,
    )

    assert metrics["recall_macro"] == pytest.approx(1 / 3)
    assert metrics["binary_fnr"] == pytest.approx(1.0)
    assert metrics["binary_recall"] == 0.0
    assert metrics["binary_fpr"] == 0.0  # never raises an alert at all
    assert metrics["per_class_recall"]["A"] == 0.0
    assert metrics["per_class_recall"]["B"] == 0.0


def test_declared_key_set_matches_what_is_produced(metrics) -> None:
    """EXTENDED_METRIC_KEYS drives serialisation; drift silently drops metrics."""
    assert set(metrics) == set(train.EXTENDED_METRIC_KEYS)


def test_non_scalar_keys_cover_every_dict_or_list_value(metrics) -> None:
    """Anything nested must be declared, or the dashboard's table breaks."""
    from src.config.constants import NON_SCALAR_METRIC_KEYS

    nested = {k for k, v in metrics.items() if isinstance(v, (dict, list))}
    assert nested <= set(NON_SCALAR_METRIC_KEYS)
