"""Put the SHAP round 2-3 results in the four tables the advisor asked for.

Nothing is refit and no SHAP is recomputed. The per-row positive Benign SHAP is
read from the saved blocks (round 2: LightGBM, round 3: RF, XGBoost, CatBoost)
and the 2,000 random feature sets are drawn again with the same generator, so
G and p must match null_test.csv of rounds 2 and 3. The script stops if they
do not.

Outputs in results/crossdataset/importance_comparison/shap_evidence/:

1. feature_group.csv     the 13 features of the dataset-specific group F
2. per_seed_stats.csv    s(missed), s(caught) and the gap per model, class, seed
3. random_sets.csv       the 2,000 random sets (one row per set)
   random_set_feature_counts.csv  how often each feature was drawn
4. null_position.csv     where G sits in its null distribution
   null_distribution.csv all 2,000 G_null values per model, direction and class
   null_position_2017_to_2018.png, null_position_2018_to_2017.png

Run::

    python scripts/shap_evidence_pack.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import dataset_vs_attack_importance as dva  # noqa: E402
import shap_importance as r1  # noqa: E402
import shap_round2 as r2  # noqa: E402

from src.crossdataset.labels import SHARED_CLASSES  # noqa: E402

CMP = r2.CMP
OUT_DIR = CMP / "shap_evidence"
BLOCKS = {
    "lightgbm": CMP / "shap_round2" / "blocks.joblib",
    "random_forest": CMP / "shap_round3" / "blocks_random_forest.joblib",
    "xgboost": CMP / "shap_round3" / "blocks_xgboost.joblib",
    "catboost": CMP / "shap_round3" / "blocks_catboost.joblib",
}
SAVED = {
    "lightgbm": CMP / "shap_round2" / "null_test.csv",
    "random_forest": CMP / "shap_round3" / "null_test.csv",
    "xgboost": CMP / "shap_round3" / "null_test.csv",
    "catboost": CMP / "shap_round3" / "null_test.csv",
}
DIRECTIONS = ("ids2017->ids2018", "ids2018->ids2017")
CLASSES = [c for c in SHARED_CLASSES if c != "Benign"]


def draw_sets(features: list[str], k: int) -> np.ndarray:
    """The consensus draws of rounds 2 and 3: first set drawn from a fresh generator."""
    rng = np.random.default_rng(r2.DRAW_SEED)
    draws = np.zeros((r2.N_DRAWS, len(features)))
    for i in range(r2.N_DRAWS):
        draws[i, rng.choice(len(features), size=k, replace=False)] = 1.0
    return draws


def share(block, cls, masks):
    """Mean share of positive Benign SHAP inside each mask, for rows of one class."""
    pos = block["pos"][block["labels"] == cls]
    total = pos.sum(axis=1)
    keep = total > 0
    if not keep.any():
        return np.full(len(masks), np.nan), 0, np.nan
    s = (pos[keep] @ masks.T / total[keep, None]).mean(axis=0)
    return s, int(keep.sum()), float(total[keep].mean())


def per_seed(blocks, direction, cls, masks, seeds):
    rows, gaps = [], []
    for seed in seeds:
        g = {b["group"]: b for b in blocks if b["direction"] == direction and b["seed"] == seed}
        s_m, n_m, t_m = share(g["missed"], cls, masks)
        s_c, n_c, t_c = share(g["caught"], cls, masks)
        gaps.append(s_m - s_c)
        rows.append({
            "seed": seed,
            "missed_total": int(g["missed"]["missed_counts"].get(cls, 0)),
            "missed_explained": n_m, "caught_explained": n_c,
            "mean_pos_benign_shap_missed": t_m, "mean_pos_benign_shap_caught": t_c,
            "s_missed": s_m[0], "s_caught": s_c[0], "gap": s_m[0] - s_c[0],
        })
    return rows, np.mean(gaps, axis=0)


def feature_group(features, F) -> pd.DataFrame:
    perm = pd.read_csv(CMP / "feature_roles.csv", index_col="feature")
    shp = pd.read_csv(CMP / "shap" / "shap_roles.csv", index_col="feature")
    t = pd.DataFrame({"feature": features})
    t["in_F"] = t["feature"].isin(F)
    t["dataset_perm"] = t["feature"].map(perm["dataset_perm"])
    t["dataset_perm_carries"] = t["feature"].map(perm["dataset_carries"])
    t["dataset_univariate_auc"] = t["feature"].map(perm["dataset_univariate_auc"])
    t["dataset_shap_max_class_share"] = t["feature"].map(shp["dataset_shap_max_class_share"])
    t["dataset_shap_carries"] = t["feature"].map(shp["dataset_carries"])
    t["role_lightgbm_perm"] = t["feature"].map(perm["role_lightgbm"])
    t["role_lightgbm_shap"] = t["feature"].map(shp["role_lightgbm_shap"])
    t = t.sort_values(["in_F", "dataset_perm"], ascending=[False, False])
    return t


def plot(null_tbl, pos_tbl, direction, path):
    models = list(BLOCKS)
    fig, axes = plt.subplots(len(models), len(CLASSES), figsize=(3.0 * len(CLASSES), 2.4 * len(models)),
                             squeeze=False)
    for i, m in enumerate(models):
        for j, cls in enumerate(CLASSES):
            ax = axes[i, j]
            v = null_tbl[(null_tbl["model"] == m) & (null_tbl["direction"] == direction)
                         & (null_tbl["class"] == cls)]["G_null"].dropna()
            r = pos_tbl[(pos_tbl["model"] == m) & (pos_tbl["direction"] == direction)
                        & (pos_tbl["class"] == cls)].iloc[0]
            if len(v) == 0 or np.isnan(r["G"]):
                ax.text(0.5, 0.5, "no caught rows", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.hist(v, bins=40, color="#9aa7b8")
                ax.axvline(r["null_p95"], color="#555555", ls=":", lw=1)
                ax.axvline(r["G"], color="#c0392b", lw=2)
                tag = "" if r["eligible"] else " (not eligible)"
                ax.set_title(f"pct {r['percentile']:.1f}, p={r['p']:.3f}{tag}", fontsize=8)
            if i == 0:
                ax.annotate(cls, (0.5, 1.25), xycoords="axes fraction", ha="center",
                            fontsize=10, weight="bold")
            if j == 0:
                ax.set_ylabel(m, fontsize=10)
            ax.tick_params(labelsize=7)
    fig.suptitle(f"G of F (red) vs 2,000 random 13-feature sets, {direction}. "
                 "Dotted line = null 95th percentile.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = r1.SEEDS
    features = dva.features_from_findings()
    F = r2.feature_sets(features)["consensus"]
    k = len(F)
    real = np.isin(features, F)[None, :].astype(float)
    draws = draw_sets(features, k)

    # 1. the group
    feature_group(features, F).to_csv(OUT_DIR / "feature_group.csv", index=False)

    # 3. the random sets
    idx = [np.flatnonzero(d) for d in draws]
    pd.DataFrame({"set_id": range(r2.N_DRAWS),
                  "features": [" | ".join(features[j] for j in row) for row in idx],
                  "overlap_with_F": [int(np.isin(np.array(features)[row], F).sum()) for row in idx]}
                 ).to_csv(OUT_DIR / "random_sets.csv", index=False)
    pd.DataFrame({"feature": features, "in_F": np.isin(features, F),
                  "times_drawn": draws.sum(axis=0).astype(int)}
                 ).to_csv(OUT_DIR / "random_set_feature_counts.csv", index=False)

    # 2 and 4
    seed_rows, null_rows, pos_rows = [], [], []
    for model, path in BLOCKS.items():
        blocks = joblib.load(path)
        saved = pd.read_csv(SAVED[model])
        if "model" in saved:
            saved = saved[saved["model"] == model]
        saved = saved[saved["set"] == "consensus"]
        for direction in DIRECTIONS:
            for cls in CLASSES:
                rows, G = per_seed(blocks, direction, cls, real, seeds)
                _, G_null = per_seed(blocks, direction, cls, draws, seeds)
                G = float(G[0])
                for r in rows:
                    seed_rows.append({"model": model, "direction": direction, "class": cls, **r})
                null_rows += [{"model": model, "direction": direction, "class": cls,
                               "set_id": i, "G_null": float(v)} for i, v in enumerate(G_null)]
                ref = saved[(saved["direction"] == direction) & (saved["class"] == cls)].iloc[0]
                eligible = (min(r["missed_total"] for r in rows) >= r1.MIN_MISSED
                            and min(r["caught_explained"] for r in rows) >= r1.MIN_CAUGHT
                            and not np.isnan(G))
                if np.isnan(G):
                    pos_rows.append({"model": model, "direction": direction, "class": cls,
                                     "eligible": False, "G": np.nan})
                    continue
                p = (1 + np.sum(G_null >= G)) / (1 + r2.N_DRAWS)
                if abs(round(G, 4) - ref["G"]) > 1e-4 or abs(round(p, 4) - ref["p"]) > 1e-4:
                    raise SystemExit(f"mismatch {model} {direction} {cls}: "
                                     f"G {G:.4f} vs {ref['G']}, p {p:.4f} vs {ref['p']}")
                q = np.nanpercentile(G_null, [5, 25, 50, 75, 95, 99])
                verdict = ""
                if direction == "ids2017->ids2018" and eligible:
                    n_elig = int((saved[saved["direction"] == direction]["eligible"]).sum())
                    verdict = ("supported" if G > 0 and p < r2.ALPHA / n_elig
                               else "not supported" if p >= r2.ALPHA else "unclear")
                pos_rows.append({
                    "model": model, "direction": direction, "class": cls,
                    "eligible": eligible, "G": G,
                    "null_mean": float(np.nanmean(G_null)), "null_sd": float(np.nanstd(G_null)),
                    "null_p5": q[0], "null_p25": q[1], "null_median": q[2], "null_p75": q[3],
                    "null_p95": q[4], "null_p99": q[5],
                    "z": float((G - np.nanmean(G_null)) / np.nanstd(G_null)),
                    "percentile": float(100 * np.mean(G_null < G)),
                    "n_null_at_or_above_G": int(np.sum(G_null >= G)),
                    "p": float(p), "verdict": verdict,
                })
        print(f"{model}: checked against {SAVED[model].parent.name}/null_test.csv, all match")

    seed_tbl = pd.DataFrame(seed_rows)
    null_tbl = pd.DataFrame(null_rows)
    pos_tbl = pd.DataFrame(pos_rows)
    seed_tbl.round(5).to_csv(OUT_DIR / "per_seed_stats.csv", index=False)
    null_tbl.round(5).to_csv(OUT_DIR / "null_distribution.csv", index=False)
    pos_tbl.round(4).to_csv(OUT_DIR / "null_position.csv", index=False)
    plot(null_tbl, pos_tbl, "ids2017->ids2018", OUT_DIR / "null_position_2017_to_2018.png")
    plot(null_tbl, pos_tbl, "ids2018->ids2017", OUT_DIR / "null_position_2018_to_2017.png")

    (OUT_DIR / "summary.json").write_text(json.dumps({
        "source_blocks": {m: str(p.relative_to(PROJECT_ROOT)) for m, p in BLOCKS.items()},
        "F": F, "set_size": k, "n_features_pool": len(features),
        "n_draws": r2.N_DRAWS, "draw_seed": r2.DRAW_SEED, "seeds": list(seeds),
        "checked_against_saved_null_test": True,
    }, indent=2), encoding="utf-8")
    print("done ->", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
