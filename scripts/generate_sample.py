"""Generate a synthetic CICIDS2017-shaped CSV under ``data/sample/``.

Why this exists
---------------
Real CICIDS2017 is multi-GB and gated behind a UNB CIC email request,
so the test suite + the Streamlit dashboard + the EDA notebook cannot
assume it is on disk. This script produces a tiny **synthetic** CSV with
exactly the same column names, dtypes, and label values as the real
dataset, so everything downstream of the loader can be exercised locally.

It is **not** a substitute for real data:
- distributions are simulated, not from real traffic
- inter-feature correlations are not modeled
- DO NOT report metrics computed on this data in the senior project

It IS useful for:
- making ``pytest`` green
- running the dashboard's Predict-New-CSV page in demo mode
- driving the EDA pipeline before the real CSVs are downloaded

Usage
-----
    python scripts/generate_sample.py            # default: 2,000 rows
    python scripts/generate_sample.py --rows 500
    python scripts/generate_sample.py --out data/sample/my.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.constants import RANDOM_STATE, SAMPLE_DIR  # noqa: E402
from src.data.schema import EXPECTED_FEATURES  # noqa: E402

# Class mix roughly mirrors CICIDS2017 imbalance (BENIGN dominant).
# Labels here are the RAW CICIDS strings; the label-mapping module
# normalizes them downstream. The Web Attack label uses the same 0x96
# control byte the real CSVs ship with.
_WEB_ATTACK_PREFIX = "Web Attack \x96 "

CLASS_WEIGHTS: dict[str, float] = {
    "BENIGN":                              0.55,
    "DoS Hulk":                            0.13,
    "DDoS":                                0.08,
    "PortScan":                            0.08,
    "DoS GoldenEye":                       0.03,
    "FTP-Patator":                         0.03,
    "SSH-Patator":                         0.02,
    "DoS slowloris":                       0.02,
    "DoS Slowhttptest":                    0.02,
    "Bot":                                 0.01,
    _WEB_ATTACK_PREFIX + "Brute Force":    0.01,
    _WEB_ATTACK_PREFIX + "XSS":            0.005,
    _WEB_ATTACK_PREFIX + "Sql Injection":  0.003,
    "Infiltration":                        0.002,
    "Heartbleed":                          0.001,
}


def generate(n_rows: int, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Build a synthetic CICIDS-shaped DataFrame with realistic label mix.

    Features are drawn from per-class log-normal distributions whose
    parameters are mildly class-dependent (so EDA + models have something
    to learn) but kept simple -- this is fixture data, not research data.
    """
    rng = np.random.default_rng(seed)
    # Normalize weights (may not sum to 1.0 exactly).
    labels_array, weights = zip(*CLASS_WEIGHTS.items(), strict=True)
    weights_arr = np.array(weights, dtype=float)
    weights_arr = weights_arr / weights_arr.sum()
    counts = rng.multinomial(n_rows, weights_arr)

    rows: list[pd.DataFrame] = []
    for label, count in zip(labels_array, counts, strict=True):
        if count == 0:
            continue
        rows.append(_synthesize_class(label, int(count), rng))
    df = pd.concat(rows, axis=0, ignore_index=True)

    # Shuffle so labels aren't in blocks.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Inject a few real-data quirks the cleaning step is supposed to fix:
    # - a handful of +/-Inf values in Flow Bytes/s
    # - a handful of NaN rows
    # - a small number of exact duplicate rows
    df = _inject_dirty_values(df, rng)

    # Column-name leading whitespace bug -- present in real CICIDS CSVs.
    df = df.rename(columns={c: f" {c}" if c != "Label" else c for c in df.columns})

    return df


def _synthesize_class(label: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate ``n`` rows for one class."""
    bias = _class_bias_vector(label, len(EXPECTED_FEATURES))
    base = rng.lognormal(mean=2.0, sigma=1.5, size=(n, len(EXPECTED_FEATURES)))
    base = base * bias
    df = pd.DataFrame(base, columns=list(EXPECTED_FEATURES))
    df["Label"] = label
    return df


def _class_bias_vector(label: str, n_features: int) -> np.ndarray:
    """Deterministic per-class bias so each class has a distinct fingerprint."""
    label_seed = abs(hash(label)) % (2**32)
    class_rng = np.random.default_rng(label_seed)
    return class_rng.uniform(0.5, 2.0, size=n_features)


def _inject_dirty_values(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate the real-data quirks the cleaning step is supposed to fix."""
    n = len(df)
    df = df.copy()

    inf_rows = rng.choice(n, size=max(1, n // 200), replace=False)
    df.loc[inf_rows, "Flow Bytes/s"] = np.inf

    nan_rows = rng.choice(n, size=max(1, n // 200), replace=False)
    df.loc[nan_rows, "Flow Packets/s"] = np.nan

    dup_count = max(1, n // 100)
    df = pd.concat([df, df.iloc[:dup_count]], axis=0, ignore_index=True)
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic CICIDS CSV.")
    parser.add_argument("--rows", type=int, default=2000, help="Total rows before dirty-value injection")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--out", type=Path, default=SAMPLE_DIR / "synthetic_cicids.csv")
    args = parser.parse_args(argv)

    df = generate(n_rows=args.rows, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows x {df.shape[1]} cols to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
