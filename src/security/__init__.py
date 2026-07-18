"""Offline adversarial-ML checks used by the research pipeline."""

from src.security.evasion import generate_bounded_perturbations, prediction_flip_rate
from src.security.ood import detect_ood, fit_iqr_profile
from src.security.poisoning import find_conflicting_labels, label_distribution_shift
from src.security.red_team import run_red_team_report, write_red_team_report

__all__ = [
    "detect_ood",
    "find_conflicting_labels",
    "fit_iqr_profile",
    "generate_bounded_perturbations",
    "label_distribution_shift",
    "prediction_flip_rate",
    "run_red_team_report",
    "write_red_team_report",
]
