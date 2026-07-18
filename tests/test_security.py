from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.security.evasion import generate_bounded_perturbations, prediction_flip_rate
from src.security.ood import detect_ood, fit_iqr_profile
from src.security.poisoning import find_conflicting_labels, label_distribution_shift
from src.security.red_team import main as red_team_main
from src.security.red_team import run_red_team_report, write_red_team_report


def test_conflicting_labels_and_distribution_shift() -> None:
    frame = pd.DataFrame({"x": [1, 1, 2], "Label": ["A", "B", "A"]})
    report = find_conflicting_labels(frame, label_column="Label", feature_columns=["x"])
    assert report.conflicting_groups == 1
    assert report.conflicting_rows == 2
    shifted = label_distribution_shift(frame, pd.DataFrame({"Label": ["A", "A"]}), label_column="Label")
    assert shifted.total_variation == pytest.approx(1 / 3)


def test_perturbation_is_deterministic_bounded_and_non_mutating() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0], "name": ["a", "b"]})
    perturbed = generate_bounded_perturbations(frame, relative_epsilon=0.1, bounds={"x": (0, 2.05)})
    assert frame.equals(pd.DataFrame({"x": [1.0, 2.0], "name": ["a", "b"]}))
    assert perturbed["x"].tolist() == pytest.approx([1.1, 1.8])
    assert perturbed["name"].tolist() == ["a", "b"]
    assert prediction_flip_rate(["A", "B", "B"], ["A", "A", "B"]).flip_rate == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        generate_bounded_perturbations(frame, relative_epsilon=-1)


def test_ood_profile_flags_extreme_nan_and_schema_mismatch() -> None:
    profile = fit_iqr_profile(pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]}))
    report = detect_ood(pd.DataFrame({"x": [2.0, 100.0], "y": [2.0, np.nan]}), profile)
    assert report.rejected_rows == 1
    with pytest.raises(ValueError, match="missing"):
        detect_ood(pd.DataFrame({"x": [1.0]}), profile)


def test_conflict_count_is_not_limited_by_examples() -> None:
    frame = pd.DataFrame({"x": [1, 1, 2, 2], "Label": ["A", "B", "A", "B"]})
    report = find_conflicting_labels(frame, label_column="Label", feature_columns=["x"], max_examples=1)
    assert report.conflicting_groups == 2
    assert len(report.examples) == 1


def test_red_team_report_is_copy_only_and_records_evasion_policy(tmp_path) -> None:
    reference = pd.DataFrame({"x": [0.0, 0.1, 9.9, 10.0], "Label": ["safe", "safe", "attack", "attack"]})
    candidate = pd.DataFrame({"x": [0.0, 9.9, 100.0, 10.0], "Label": ["safe", "attack", "attack", "attack"]})

    def predictor(features: pd.DataFrame) -> np.ndarray:
        return np.where(features["x"] >= 5.0, "attack", "safe")

    original_reference = reference.copy(deep=True)
    original_candidate = candidate.copy(deep=True)
    report = run_red_team_report(reference, candidate, label_column="Label", predictor=predictor)

    assert reference.equals(original_reference)
    assert candidate.equals(original_candidate)
    assert report["scope"] == "offline-local-data-only"
    assert report["ood"]["rejected_rows"] == 1
    assert report["evasion"]["performed"] is True
    assert len(report["evasion"]["scenarios"]) == 3
    assert report["evasion"]["five_percent_policy_pass"] is True

    output = write_red_team_report(report, tmp_path / "nested" / "red_team.json")
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["report_version"] == "red-team-offline-v1"


def test_red_team_report_without_model_and_bad_predictor() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0], "Label": ["A", "B"]})
    report = run_red_team_report(frame, frame, label_column="Label")
    assert report["evasion"] == {"performed": False, "reason": "No predictor supplied."}
    with pytest.raises(ValueError, match="one prediction"):
        run_red_team_report(frame, frame, label_column="Label", predictor=lambda _: ["A"])


def test_red_team_module_cli_writes_local_report(tmp_path) -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "Label": ["A", "A", "B"]})
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    output = tmp_path / "report.json"
    frame.to_csv(reference, index=False)
    frame.to_csv(candidate, index=False)

    assert red_team_main([
        "--reference", str(reference), "--candidate", str(candidate), "--output", str(output),
    ]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["evasion"]["performed"] is False
