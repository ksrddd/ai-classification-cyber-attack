"""gen_cicids_mock.py -- Generate realistic CICIDS2017-style mock test data.

Samples are drawn from the per-class mean/std of the actual training split
so the feature distribution matches what the models were trained on.

Usage:
    python scripts/gen_cicids_mock.py                    # 20 per class -> Downloads
    python scripts/gen_cicids_mock.py --n 50             # 50 per class
    python scripts/gen_cicids_mock.py --out my_test.csv  # custom output path
    python scripts/gen_cicids_mock.py --seed 123         # reproducible
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.encoder import load_label_encoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate mock CICIDS2017-style test CSV")
    p.add_argument("--n",    type=int,  default=20,   help="Samples per class (default 20)")
    p.add_argument("--out",  type=str,  default=None, help="Output CSV path")
    p.add_argument("--seed", type=int,  default=42,   help="Random seed")
    p.add_argument("--classes", nargs="+", default=None,
                   help="Only generate these classes, e.g. --classes DoS DDoS PortScan")
    return p.parse_args()


def load_training_stats(features: list[str]) -> dict[str, pd.DataFrame]:
    """Return per-class {mean, std, min, max} from actual training parquet."""
    df = pd.read_parquet(PROJECT_ROOT / "data/processed/train.parquet")
    le = load_label_encoder()
    stats: dict[str, pd.DataFrame] = {}
    for cls_id, cls_name in enumerate(le.classes_):
        subset = df[df["label_encoded"] == cls_id][features]
        if len(subset) == 0:
            continue
        stats[cls_name] = pd.DataFrame({
            "mean": subset.mean(),
            "std":  subset.std().fillna(0),
            "min":  subset.min(),
            "max":  subset.max(),
            "p05":  subset.quantile(0.05),
            "p95":  subset.quantile(0.95),
        })
    return stats


def generate_class_samples(
    cls_name: str,
    stats: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw n samples per feature using truncated Gaussian clipped to [p05, p95]."""
    rows = {}
    for feat in stats.index:
        mean = stats.loc[feat, "mean"]
        std  = stats.loc[feat, "std"]
        lo   = stats.loc[feat, "p05"]
        hi   = stats.loc[feat, "p95"]

        if std == 0 or np.isnan(std):
            vals = np.full(n, mean)
        else:
            vals = rng.normal(mean, std, size=n)
            vals = np.clip(vals, lo, hi)

        # Preserve integer-like features
        orig_min = stats.loc[feat, "min"]
        orig_max = stats.loc[feat, "max"]
        if orig_min >= 0 and orig_max == int(orig_max) and mean == int(mean):
            vals = np.round(vals).astype(int)
            vals = np.clip(vals, int(orig_min), int(orig_max))

        rows[feat] = vals

    df = pd.DataFrame(rows)
    df["true_label"] = cls_name
    return df


def main() -> None:
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    features: list[str] = json.load(
        open(PROJECT_ROOT / "data/processed/feature_names.json")
    )

    print("Loading training statistics...")
    all_stats = load_training_stats(features)

    target_classes = args.classes or list(all_stats.keys())
    missing = [c for c in target_classes if c not in all_stats]
    if missing:
        print(f"WARNING: unknown classes (skipped): {missing}")
        target_classes = [c for c in target_classes if c in all_stats]

    frames: list[pd.DataFrame] = []
    for cls_name in target_classes:
        df_cls = generate_class_samples(cls_name, all_stats[cls_name], args.n, rng)
        frames.append(df_cls)
        print(f"  {cls_name:20s}: {len(df_cls)} samples generated")

    result = pd.concat(frames, ignore_index=True)
    result = result.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    out_path = Path(args.out) if args.out else Path.home() / "Downloads/mock_cicids_test.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    print(f"\nSaved {len(result)} rows ({len(target_classes)} classes x {args.n}) -> {out_path}")
    print("\nClass counts in output:")
    print(result["true_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
