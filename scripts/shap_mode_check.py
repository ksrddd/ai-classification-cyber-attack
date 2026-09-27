"""Robustness check 1b (exploratory): does the SHAP mode change LightGBM's ranking?

The main run used tree_path_dependent for LightGBM because interventional was
too slow. This fits the Part A attack model once (ids2017, seed 42), explains
the same rows both ways, and compares the feature shares.

Small on purpose (300 rows, 100 background rows) so it finishes in minutes.

Run::

    python scripts/shap_mode_check.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import dataset_vs_attack_importance as dva  # noqa: E402
from shap_importance import sample_rows, shap_array, shares  # noqa: E402

from src.crossdataset.labels import SHARED_CLASSES  # noqa: E402
from src.crossdataset.loaders import load_ids2017  # noqa: E402
from src.ids2018.models import build_model, fit_model  # noqa: E402
from src.ids2018.preprocessing import Ids2018Preprocessor  # noqa: E402

OUT = PROJECT_ROOT / "results" / "crossdataset" / "importance_comparison" / "shap" / "mode_check.json"
SEED = 42
ROWS = 300
BACKGROUND = 100


def main() -> int:
    corpus = load_ids2017()
    features = dva.features_from_findings()
    encoder = LabelEncoder().fit(np.array(SHARED_CLASSES))
    y = encoder.transform(corpus.y.astype(str))
    X_tr, X_te, y_tr, y_te = train_test_split(
        corpus.X[features], y, test_size=dva.TEST_SIZE, random_state=SEED, stratify=y
    )
    pre = Ids2018Preprocessor()
    Xt, Xe = pre.fit_transform(X_tr), pre.transform(X_te)
    names = list(pre.feature_names)
    model = fit_model("lightgbm", dva.seed_model(build_model("lightgbm"), SEED), Xt, y_tr)
    bg, _ = sample_rows(Xt, y_tr, BACKGROUND, SEED)
    rows, _ = sample_rows(Xe, y_te, ROWS, SEED)

    out = {}
    t0 = time.perf_counter()
    path = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    sv_path = shap_array(path.shap_values(rows, check_additivity=False), *rows.shape)
    out["path_seconds"] = round(time.perf_counter() - t0, 1)

    t0 = time.perf_counter()
    masker = shap.maskers.Independent(bg, max_samples=len(bg))
    inter = shap.TreeExplainer(model, data=masker.data, feature_perturbation="interventional")
    sv_int = shap_array(inter.shap_values(rows, check_additivity=False), *rows.shape)
    out["interventional_seconds"] = round(time.perf_counter() - t0, 1)

    a = shares(sv_path, names, features)
    b = shares(sv_int, names, features)
    floor = 1 / len(features)
    out.update({
        "rows": ROWS, "background": BACKGROUND, "corpus": "ids2017", "seed": SEED,
        "spearman": round(float(spearmanr(a, b).statistic), 4),
        "top10_overlap": len(set(a.nlargest(10).index) & set(b.nlargest(10).index)),
        "above_1_60": {"path": int((a >= floor).sum()), "interventional": int((b >= floor).sum())},
        "above_1_60_agree": round(float(((a >= floor) == (b >= floor)).mean()), 3),
        "top10_path": [[f, round(float(v), 4)] for f, v in a.nlargest(10).items()],
        "top10_interventional": [[f, round(float(v), 4)] for f, v in b.nlargest(10).items()],
    })
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("top10_")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
