"""Bounded, copy-only feature perturbations for offline robustness tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def generate_bounded_perturbations(
    frame: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    relative_epsilon: float = 0.01,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Perturb numeric columns deterministically and leave the input untouched."""
    if relative_epsilon < 0:
        raise ValueError("relative_epsilon must be non-negative")
    out = frame.copy(deep=True)
    selected = columns or [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    for column in selected:
        if column not in out.columns:
            raise ValueError(f"Unknown perturbation column: {column}")
        if not pd.api.types.is_numeric_dtype(out[column]):
            continue
        values = out[column].to_numpy(dtype=float, copy=True)
        direction = np.where(np.arange(len(values)) % 2 == 0, 1.0, -1.0)
        values = values + direction * relative_epsilon * np.maximum(np.abs(values), 1.0)
        if bounds and column in bounds:
            lower, upper = bounds[column]
            if lower > upper:
                raise ValueError(f"Invalid bounds for {column}")
            values = np.clip(values, lower, upper)
        out[column] = values
    return out


@dataclass(frozen=True)
class StabilityReport:
    compared: int
    flips: int
    flip_rate: float


def prediction_flip_rate(clean: object, perturbed: object) -> StabilityReport:
    """Compare two prediction sequences and report class-flip rate."""
    clean_values = list(clean)
    perturbed_values = list(perturbed)
    if len(clean_values) != len(perturbed_values):
        raise ValueError("Prediction sequences must have equal length")
    flips = sum(a != b for a, b in zip(clean_values, perturbed_values, strict=True))
    return StabilityReport(len(clean_values), flips, flips / max(len(clean_values), 1))
