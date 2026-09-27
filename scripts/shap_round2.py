"""SHAP round 2: Part B again, with a consensus F and a random-feature-set null.

Rules are in results/crossdataset/importance_comparison/SHAP_ROUND2_PREREGISTRATION.md.
Uses the same models, rows and SHAP settings as round 1 Part B
(scripts/shap_importance.py); only the way the result is judged changes.

Run::

    python scripts/shap_round2.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import dataset_vs_attack_importance as dva  # noqa: E402
import shap_importance as r1  # noqa: E402
from run_crossdataset import partition  # noqa: E402

from src.crossdataset.labels import SHARED_CLASSES  # noqa: E402
from src.crossdataset.loaders import load_ids2017, load_ids2018  # noqa: E402
from src.ids2018.models import build_model, fit_model  # noqa: E402
from src.ids2018.preprocessing import Ids2018Preprocessor  # noqa: E402

logger = logging.getLogger("shap_round2")

CMP = PROJECT_ROOT / "results" / "crossdataset" / "importance_comparison"
OUT_DIR = CMP / "shap_round2"
PREREG = CMP / "SHAP_ROUND2_PREREGISTRATION.md"

N_DRAWS = 2_000
DRAW_SEED = 2026
ALPHA = 0.05


def lean_parquet_reads() -> None:
    """Read parquet files with a lower memory peak; the values are the same.

    The default read builds one 77 x 11.4M float32 block (3.3 GB) while the
    Arrow copy is still alive, which failed on this laptop when other programs
    were open. split_blocks keeps one block per column and self_destruct frees
    the Arrow buffers as each column is converted.
    """
    import pyarrow.parquet as pq

    def read(path, columns=None, **_):
        return pq.read_table(path, columns=columns).to_pandas(split_blocks=True,
                                                             self_destruct=True)

    pd.read_parquet = read


def feature_sets(features: list[str]) -> dict[str, list[str]]:
    perm = pd.read_csv(CMP / "feature_roles.csv", index_col="feature")["dataset_carries"]
    shap_t = pd.read_csv(CMP / "shap" / "shap_roles.csv", index_col="feature")["dataset_carries"]
    perm = perm.reindex(features).fillna(False).astype(bool)
    shap_t = shap_t.reindex(features).fillna(False).astype(bool)
    return {
        "consensus": [f for f in features if perm[f] and shap_t[f]],
        "shap": [f for f in features if shap_t[f]],
        "perm": [f for f in features if perm[f]],
    }


def positive_benign(sv_benign, names, mapping, features) -> np.ndarray:
    """Rows x 60 matrix of positive Benign SHAP, duplicates summed into representatives."""
    frame = pd.DataFrame(sv_benign, columns=names).T.groupby(lambda c: mapping.get(c, c)).sum().T
    return frame.reindex(columns=features, fill_value=0.0).clip(lower=0).to_numpy()


def collect(c17, c18, features, seeds) -> list[dict]:
    """Same models and rows as round 1 Part B; keep the positive Benign SHAP per row."""
    encoder = LabelEncoder().fit(np.array(SHARED_CLASSES))
    benign = int(encoder.transform(["Benign"])[0])
    corpora = {"ids2017": c17, "ids2018": c18}
    parts = {n: partition(c, "chronological", 0) for n, c in corpora.items()}
    mapping = r1.representative_map(list(c17.X.columns))

    blocks = []
    for source, target in (("ids2017", "ids2018"), ("ids2018", "ids2017")):
        src, tgt = corpora[source], corpora[target]
        ps, pt = parts[source], parts[target]
        for seed in seeds:
            pre = Ids2018Preprocessor()
            Xt = pre.fit_transform(src.X.iloc[ps.train])
            yt = encoder.transform(src.y.iloc[ps.train])
            names = list(pre.feature_names)
            model = fit_model("lightgbm", dva.seed_model(build_model("lightgbm"), seed), Xt, yt)

            Xe_t = pre.transform(tgt.X.iloc[pt.test])
            y_t = tgt.y.iloc[pt.test].astype(str).reset_index(drop=True)
            missed_mask = (y_t != "Benign").to_numpy() & (model.predict(Xe_t) == benign)
            missed_counts = y_t[missed_mask].value_counts()

            Xe_s = pre.transform(src.X.iloc[ps.test])
            y_s = src.y.iloc[ps.test].astype(str).reset_index(drop=True)
            pred_s = encoder.inverse_transform(model.predict(Xe_s))
            caught_mask = (y_s != "Benign").to_numpy() & (pred_s == y_s.to_numpy())

            for gname, mask, X_all, y_all in (
                ("missed", missed_mask, Xe_t, y_t),
                ("caught", caught_mask, Xe_s, y_s),
            ):
                idx = np.flatnonzero(mask)
                idx = r1.cap_per_class(idx, y_all.iloc[idx].reset_index(drop=True),
                                       r1.MISSED_PER_CLASS, seed)
                if len(idx) == 0:
                    pos = np.zeros((0, len(features)))
                else:
                    sv = r1.explain(model, "lightgbm", None, X_all[idx])
                    pos = positive_benign(sv[:, :, benign], names, mapping, features)
                blocks.append({
                    "direction": f"{source}->{target}", "seed": seed, "group": gname,
                    "labels": y_all.iloc[idx].to_numpy(), "pos": pos,
                    "missed_counts": missed_counts,
                })
            logger.info("  %s->%s seed %d done", source[-4:], target[-4:], seed)
            del model
    return blocks


def class_gaps(blocks, direction, cls, masks: np.ndarray, seeds) -> tuple[np.ndarray, dict]:
    """Mean-over-seeds gap for every feature set in ``masks`` (sets x 60)."""
    per_seed, info = [], {"n_missed_total": [], "n_caught": []}
    for seed in seeds:
        g = {b["group"]: b for b in blocks if b["direction"] == direction and b["seed"] == seed}
        s = {}
        for name in ("missed", "caught"):
            b = g[name]
            pos = b["pos"][b["labels"] == cls]
            total = pos.sum(axis=1)
            keep = total > 0
            s[name] = (pos[keep] @ masks.T / total[keep, None]).mean(axis=0) if keep.any() \
                else np.full(len(masks), np.nan)
        info["n_missed_total"].append(int(g["missed"]["missed_counts"].get(cls, 0)))
        info["n_caught"].append(int((g["caught"]["labels"] == cls).sum()))
        per_seed.append(s["missed"] - s["caught"])
    return np.mean(per_seed, axis=0), info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--cached", action="store_true",
                    help="reuse blocks.joblib from an earlier run instead of refitting")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(args.out_dir / "run.log", encoding="utf-8")],
    )
    prereg_sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    logger.info("pre-registration sha256 %s", prereg_sha)

    seeds = r1.SEEDS
    features = dva.features_from_findings()
    sets = feature_sets(features)
    logger.info("feature sets: %s", {k: len(v) for k, v in sets.items()})

    started = time.perf_counter()
    cache = args.out_dir / "blocks.joblib"
    if args.cached and cache.exists():
        blocks = joblib.load(cache)
        logger.info("loaded %d SHAP blocks from %s", len(blocks), cache)
    else:
        lean_parquet_reads()
        c17, c18 = load_ids2017(), load_ids2018()
        blocks = collect(c17, c18, features, seeds)
        # per-row positive Benign SHAP, so the null can be redrawn without refitting
        joblib.dump(blocks, cache, compress=3)

    rng = np.random.default_rng(DRAW_SEED)
    rows, summary_sets = [], {}
    for set_name, F in sets.items():
        k = len(F)
        real = np.isin(features, F)[None, :].astype(float)
        draws = np.zeros((N_DRAWS, len(features)))
        for i in range(N_DRAWS):
            draws[i, rng.choice(len(features), size=k, replace=False)] = 1.0
        for direction in ("ids2017->ids2018", "ids2018->ids2017"):
            for cls in SHARED_CLASSES:
                if cls == "Benign":
                    continue
                G, info = class_gaps(blocks, direction, cls, real, seeds)
                G_null, _ = class_gaps(blocks, direction, cls, draws, seeds)
                G = float(G[0])
                eligible = (min(info["n_missed_total"]) >= r1.MIN_MISSED
                            and min(info["n_caught"]) >= r1.MIN_CAUGHT and not np.isnan(G))
                p = float((1 + np.sum(G_null >= G)) / (1 + N_DRAWS)) if not np.isnan(G) else np.nan
                rows.append({
                    "set": set_name, "set_size": k, "direction": direction, "class": cls,
                    "eligible": eligible, "G": round(G, 4),
                    "null_mean": round(float(np.nanmean(G_null)), 4),
                    "null_95": round(float(np.nanpercentile(G_null, 95)), 4),
                    "p": round(p, 4),
                    "min_missed_total": min(info["n_missed_total"]),
                    "min_caught": min(info["n_caught"]),
                })

    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "null_test.csv", index=False)

    verdicts = {}
    for set_name in sets:
        main_rows = table[(table["set"] == set_name) & (table["direction"] == "ids2017->ids2018")
                          & table["eligible"]]
        n = len(main_rows)
        per_class = {}
        for _, r in main_rows.iterrows():
            if r["G"] > 0 and r["p"] < ALPHA / n:
                per_class[r["class"]] = "supported"
            elif r["p"] >= ALPHA:
                per_class[r["class"]] = "not supported"
            else:
                per_class[r["class"]] = "unclear"
        sup = sum(v == "supported" for v in per_class.values())
        nos = sum(v == "not supported" for v in per_class.values())
        overall = ("fingerprint causes the misses" if n and sup > n / 2
                   else "not supported" if n and nos > n / 2 else "partial")
        verdicts[set_name] = {"overall": overall, "per_class": per_class}
        logger.info("%-9s (%d features): %s %s", set_name, len(sets[set_name]), overall, per_class)

    summary = {
        "preregistration_sha256": prereg_sha, "seeds": list(seeds), "n_draws": N_DRAWS,
        "draw_seed": DRAW_SEED, "feature_sets": sets, "decision_set": "consensus",
        "verdicts": verdicts, "minutes": round((time.perf_counter() - started) / 60, 1),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("done in %.1f min -> %s", summary["minutes"], args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
