"""Exact null thresholds for SHAP round 2 (descriptive, does not change any verdict).

Round 2 decides with 2,000 random feature sets (seed 2026), as pre-registered.
This script answers a different question: how large would G have had to be to
pass, measured precisely? It redraws the null with more and more random sets
(separate seed 2027) until the thresholds stop moving.

- threshold used by the decision: the 1 - 0.05/3 = 98.33% point of the null
- also the 95% point, and a precise p for the observed G
- stops when no threshold moves by more than 0.002 between steps
- 95% interval for each threshold from order statistics

Needs shap_round2/blocks.joblib (written by scripts/shap_round2.py).

Run::

    python scripts/shap_null_thresholds.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import dataset_vs_attack_importance as dva  # noqa: E402
import shap_importance as r1  # noqa: E402
import shap_round2 as r2  # noqa: E402

OUT = r2.OUT_DIR
STEPS = (2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000)
BATCH = 5_000
SEED = 2027
TOL = 0.002
DIRECTION = "ids2017->ids2018"


def quantile_ci(sorted_vals: np.ndarray, q: float) -> tuple[float, float]:
    """95% interval for the q-quantile from binomial order statistics."""
    n = len(sorted_vals)
    half = 1.96 * np.sqrt(n * q * (1 - q))
    lo = int(max(0, np.floor(n * q - half)))
    hi = int(min(n - 1, np.ceil(n * q + half)))
    return float(sorted_vals[lo]), float(sorted_vals[hi])


def main() -> int:
    blocks = joblib.load(OUT / "blocks.joblib")
    features = dva.features_from_findings()
    sets = r2.feature_sets(features)
    decided = pd.read_csv(OUT / "null_test.csv")
    cells = decided[(decided["direction"] == DIRECTION) & decided["eligible"]]
    n_classes = len(cells["class"].unique())
    q_decision = 1 - r2.ALPHA / n_classes

    rng = np.random.default_rng(SEED)
    null = {(s, c): np.empty(0) for s, c in zip(cells["set"], cells["class"], strict=True)}
    history, previous = [], None
    for step in STEPS:
        for (set_name, cls), have in null.items():
            need = step - len(have)
            k = len(sets[set_name])
            parts = [have]
            while need > 0:
                b = min(BATCH, need)
                draws = np.zeros((b, len(features)))
                for i in range(b):
                    draws[i, rng.choice(len(features), size=k, replace=False)] = 1.0
                g, _ = r2.class_gaps(blocks, DIRECTION, cls, draws, r1.SEEDS)
                parts.append(g)
                need -= b
            null[(set_name, cls)] = np.concatenate(parts)

        current = {}
        for (set_name, cls), vals in null.items():
            v = np.sort(vals[~np.isnan(vals)])
            G = float(cells[(cells["set"] == set_name) & (cells["class"] == cls)]["G"].iloc[0])
            current[(set_name, cls)] = {
                "set": set_name, "class": cls, "draws": step, "G": G,
                "q95": float(np.quantile(v, 0.95)),
                "q_decision": float(np.quantile(v, q_decision)),
                "q_decision_ci": quantile_ci(v, q_decision),
                "p": float((1 + np.sum(v >= G)) / (1 + len(v))),
            }
        moved = (max(abs(current[k]["q_decision"] - previous[k]["q_decision"]) for k in current)
                 if previous else None)
        history.append({"draws": step, "max_change": moved})
        print(f"draws {step:>6}: max change of the {q_decision:.2%} point = "
              f"{'-' if moved is None else f'{moved:.4f}'}")
        previous = current
        if moved is not None and moved < TOL:
            break

    table = pd.DataFrame(previous.values())
    table["q_decision_lo"] = [c[0] for c in table.pop("q_decision_ci")]
    table["q_decision_hi"] = [previous[(s, c)]["q_decision_ci"][1]
                              for s, c in zip(table["set"], table["class"], strict=True)]
    table["G_needed_minus_G"] = table["q_decision"] - table["G"]
    table = table.round(4)
    table.to_csv(OUT / "null_thresholds.csv", index=False)
    (OUT / "null_thresholds.json").write_text(json.dumps({
        "note": "descriptive only; verdicts come from null_test.csv (2,000 draws, seed 2026)",
        "direction": DIRECTION, "decision_quantile": q_decision, "seed": SEED,
        "tolerance": TOL, "history": history,
    }, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
