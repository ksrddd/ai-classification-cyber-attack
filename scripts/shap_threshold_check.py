"""Robustness check 1a (exploratory, not part of the pre-registered decision).

Part A compared SHAP roles (1/60 share rule) with permutation roles (noise-floor
rule). Two things changed at once: the method and the threshold. This puts both
methods on the same top-K rule so the effect of the method can be seen alone.

- dataset side: a feature carries if it is in the top K of any shared class
- attack side: a feature carries if it is in the top K for either corpus

Uses only saved outputs, nothing is retrained.

Run::

    python scripts/shap_threshold_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dataset_vs_attack_importance import quadrant  # noqa: E402

CMP = PROJECT_ROOT / "results" / "crossdataset" / "importance_comparison"
AV = PROJECT_ROOT / "results" / "crossdataset" / "adversarial_validation"
OUT = CMP / "shap" / "threshold_check.json"

MODELS = ("lightgbm", "logistic_regression")
CORPORA = ("ids2017", "ids2018")
KS = (5, 10, 15, 20)


def top_k_any(frame: pd.DataFrame, k: int) -> pd.Series:
    """True if the feature is in the top k of any column of ``frame``."""
    hit = pd.Series(False, index=frame.index)
    for col in frame.columns:
        s = frame[col]
        s = s[s > 0]
        hit[s.nlargest(k).index] = True
    return hit


def main() -> int:
    shap_roles = pd.read_csv(CMP / "shap" / "shap_roles.csv", index_col="feature")
    features = list(shap_roles.index)
    shap_ds = pd.read_csv(CMP / "shap" / "dataset_shap_per_class.csv", index_col="feature")
    perm_ds = (pd.read_csv(AV / "importance.csv")
               .pivot(index="feature", columns="shared_class", values="importance")
               .reindex(features).fillna(0.0))
    perm = pd.read_csv(CMP / "feature_roles.csv", index_col="feature").reindex(features)

    result = {"note": "exploratory; same top-K rule for both methods", "k": {}}
    for k in KS:
        ds_shap = top_k_any(shap_ds.reindex(features).fillna(0.0), k)
        ds_perm = top_k_any(perm_ds, k)
        row = {"dataset_side_agreement": round(float((ds_shap == ds_perm).mean()), 3)}
        for m in MODELS:
            a_shap = top_k_any(shap_roles[[f"attack_{m}_{c}_share" for c in CORPORA]], k)
            a_perm = top_k_any(perm[[f"attack_{m}_{c}" for c in CORPORA]], k)
            r_shap = [quadrant(d, a) for d, a in zip(ds_shap, a_shap, strict=True)]
            r_perm = [quadrant(d, a) for d, a in zip(ds_perm, a_perm, strict=True)]
            row[m] = {
                "kappa": round(float(cohen_kappa_score(r_perm, r_shap)), 3),
                "attack_side_agreement": round(float((a_shap == a_perm).mean()), 3),
            }
        result["k"][str(k)] = row
        print(f"K={k:>2}  dataset agree {row['dataset_side_agreement']:.2f}  "
              + "  ".join(f"{m}: kappa {row[m]['kappa']:+.3f} attack agree "
                          f"{row[m]['attack_side_agreement']:.2f}" for m in MODELS))

    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
