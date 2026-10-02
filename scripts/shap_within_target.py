"""SHAP within the target dataset: missed (predicted Benign) vs caught, same class.

Rules are in results/crossdataset/importance_comparison/SHAP_WITHIN_TARGET_PREREGISTRATION.md.
Rounds 2 and 3 compared missed rows (target) with caught rows (source), so the
gap mixed "missed vs caught" with "2018 vs 2017". Here the caught rows come
from the target test split too. missed and caught_src are read from the round 2
and round 3 caches; only caught_tgt is computed.

Run::

    python scripts/shap_within_target.py
    python scripts/shap_within_target.py --models lightgbm xgboost
    python scripts/shap_within_target.py --cached        # reuse blocks_<model>.joblib
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
import shap_round2 as r2  # noqa: E402
import shap_round3 as r3  # noqa: E402
from run_crossdataset import partition  # noqa: E402

from src.crossdataset.labels import SHARED_CLASSES  # noqa: E402
from src.crossdataset.loaders import load_ids2017, load_ids2018  # noqa: E402
from src.ids2018.models import build_model, fit_model  # noqa: E402
from src.ids2018.preprocessing import Ids2018Preprocessor  # noqa: E402

logger = logging.getLogger("shap_within_target")

CMP = r2.CMP
OUT_DIR = CMP / "shap_within_target"
PREREG = CMP / "SHAP_WITHIN_TARGET_PREREGISTRATION.md"

MODELS = ("lightgbm", "xgboost", "catboost", "random_forest")
DIRECTIONS = ("ids2017->ids2018", "ids2018->ids2017")
ATTACKS = [c for c in SHARED_CLASSES if c != "Benign"]
GROUPS = ("missed", "caught_tgt", "caught_src")


def old_cache(model_name: str) -> Path:
    if model_name == "lightgbm":
        return CMP / "shap_round2" / "blocks.joblib"
    return CMP / "shap_round3" / f"blocks_{model_name}.joblib"


def collect(model_name, c17, c18, features, seeds, old, checkpoint: Path) -> list[dict]:
    """Refit each (direction, seed) model and explain the target rows it caught."""
    encoder = LabelEncoder().fit(np.array(SHARED_CLASSES))
    benign = int(encoder.transform(["Benign"])[0])
    corpora = {"ids2017": c17, "ids2018": c18}
    parts = {n: partition(c, "chronological", 0) for n, c in corpora.items()}
    mapping = r1.representative_map(list(c17.X.columns))

    blocks = joblib.load(checkpoint) if checkpoint.exists() else []
    done = {(b["direction"], b["seed"]) for b in blocks}
    for direction in DIRECTIONS:
        source, target = direction.split("->")
        src, tgt = corpora[source], corpora[target]
        ps, pt = parts[source], parts[target]
        for seed in seeds:
            if (direction, seed) in done:
                continue
            t0 = time.perf_counter()
            pre = Ids2018Preprocessor()
            Xt = pre.fit_transform(src.X.iloc[ps.train])
            yt = encoder.transform(src.y.iloc[ps.train])
            names = list(pre.feature_names)
            model = fit_model(model_name, dva.seed_model(build_model(model_name), seed), Xt, yt)

            Xe_t = pre.transform(tgt.X.iloc[pt.test])
            y_t = tgt.y.iloc[pt.test].astype(str).reset_index(drop=True)
            pred = encoder.inverse_transform(np.asarray(model.predict(Xe_t)).ravel().astype(int))
            attack = (y_t != "Benign").to_numpy()
            missed_mask = attack & (pred == "Benign")
            caught_mask = attack & (pred == y_t.to_numpy())

            # same model as the cache, or the groups are not comparable
            cached = next(b for b in old if b["direction"] == direction
                          and b["seed"] == seed and b["group"] == "missed")
            now = y_t[missed_mask].value_counts()
            if not now.sort_index().equals(cached["missed_counts"].sort_index()):
                raise RuntimeError(f"{model_name} {direction} seed {seed}: missed counts "
                                   f"{now.to_dict()} differ from cache "
                                   f"{cached['missed_counts'].to_dict()}")

            counts = pd.DataFrame({
                "n_attack": y_t[attack].value_counts(),
                "n_missed": now,
                "n_caught_tgt": y_t[caught_mask].value_counts(),
            }).reindex(ATTACKS).fillna(0).astype(int)
            counts["n_other_attack"] = counts["n_attack"] - counts["n_missed"] - counts["n_caught_tgt"]

            idx = np.flatnonzero(caught_mask)
            idx = r1.cap_per_class(idx, y_t.iloc[idx].reset_index(drop=True),
                                   r1.MISSED_PER_CLASS, seed)
            if len(idx) == 0:
                pos = np.zeros((0, len(features)))
            else:
                sv = r3.explain(model, model_name, None, Xe_t[idx])
                pos = r2.positive_benign(sv[:, :, benign], names, mapping, features)
            blocks.append({"direction": direction, "seed": seed, "group": "caught_tgt",
                           "labels": y_t.iloc[idx].to_numpy(), "pos": pos, "counts": counts})
            logger.info("  %s %s seed %d done (%.0fs, caught in target %s)", model_name,
                        direction, seed, time.perf_counter() - t0,
                        counts["n_caught_tgt"].to_dict())
            del model
            joblib.dump(blocks, checkpoint, compress=3)
    return blocks


def merged(old, new) -> list[dict]:
    out = []
    for b in old:
        out.append({**b, "group": "missed" if b["group"] == "missed" else "caught_src"})
    return out + list(new)


def share(blocks, direction, seed, group, cls, masks) -> tuple[np.ndarray, int]:
    b = next(x for x in blocks if x["direction"] == direction and x["seed"] == seed
             and x["group"] == group)
    pos = b["pos"][b["labels"] == cls]
    total = pos.sum(axis=1)
    keep = total > 0
    if not keep.any():
        return np.full(len(masks), np.nan), len(pos)
    return (pos[keep] @ masks.T / total[keep, None]).mean(axis=0), len(pos)


def gaps(blocks, direction, cls, masks, seeds) -> dict[str, np.ndarray]:
    per = {g: [] for g in GROUPS}
    for seed in seeds:
        for g in GROUPS:
            per[g].append(share(blocks, direction, seed, g, cls, masks)[0])
    s = {g: np.mean(v, axis=0) for g, v in per.items()}
    return {"within": s["missed"] - s["caught_tgt"],
            "old": s["missed"] - s["caught_src"],
            "dataset": s["caught_tgt"] - s["caught_src"]}


def counts_table(new, model_name) -> pd.DataFrame:
    rows = []
    for b in new:
        c = b["counts"].reset_index(names="class")
        c.insert(0, "seed", b["seed"])
        c.insert(0, "direction", b["direction"])
        rows.append(c)
    t = pd.concat(rows, ignore_index=True)
    t.insert(0, "model", model_name)
    return t


def null_test(model_name, blocks, counts, sets, features, seeds) -> pd.DataFrame:
    rng = np.random.default_rng(r2.DRAW_SEED)
    rows = []
    for set_name, F in sets.items():
        k = len(F)
        real = np.isin(features, F)[None, :].astype(float)
        draws = np.zeros((r2.N_DRAWS, len(features)))
        for i in range(r2.N_DRAWS):
            draws[i, rng.choice(len(features), size=k, replace=False)] = 1.0
        for direction in DIRECTIONS:
            for cls in ATTACKS:
                c = counts[(counts["direction"] == direction) & (counts["class"] == cls)]
                min_missed, min_caught = int(c["n_missed"].min()), int(c["n_caught_tgt"].min())
                eligible = min_missed >= r1.MIN_MISSED and min_caught >= r1.MIN_CAUGHT
                g = gaps(blocks, direction, cls, real, seeds)
                G = float(g["within"][0])
                row = {"model": model_name, "set": set_name, "set_size": k,
                       "direction": direction, "class": cls,
                       "min_missed": min_missed, "min_caught_tgt": min_caught,
                       "eligible": eligible and not np.isnan(G),
                       "G_within": round(G, 4), "G_old": round(float(g["old"][0]), 4),
                       "G_dataset": round(float(g["dataset"][0]), 4)}
                if not np.isnan(G):
                    G_null = gaps(blocks, direction, cls, draws, seeds)["within"]
                    row.update({"null_mean": round(float(np.nanmean(G_null)), 4),
                                "null_95": round(float(np.nanpercentile(G_null, 95)), 4),
                                "p": round(float((1 + np.sum(G_null >= G)) / (1 + r2.N_DRAWS)), 4)})
                rows.append(row)
    return pd.DataFrame(rows)


def judge(table: pd.DataFrame, set_name: str) -> dict:
    main = table[(table["set"] == set_name) & table["eligible"]]
    n = len(main)
    per_cell = {}
    for _, r in main.iterrows():
        key = f"{r['direction']} {r['class']}"
        if r["G_within"] > 0 and r["p"] < r2.ALPHA / n:
            per_cell[key] = "supported"
        elif r["p"] >= r2.ALPHA:
            per_cell[key] = "not supported"
        else:
            per_cell[key] = "unclear"
    sup = sum(v == "supported" for v in per_cell.values())
    nos = sum(v == "not supported" for v in per_cell.values())
    overall = ("miss-specific" if n and sup > n / 2
               else "not supported" if n and nos > n / 2 else "partial" if n else "no eligible cell")
    return {"overall": overall, "n_eligible": n, "per_cell": per_cell}


def feature_diff(model_name, blocks, features, seeds) -> pd.DataFrame:
    """Per feature: mean share of positive Benign SHAP, missed minus caught_tgt."""
    eye = np.eye(len(features))
    rows = []
    for direction in DIRECTIONS:
        for cls in ATTACKS:
            g = gaps(blocks, direction, cls, eye, seeds)
            if np.all(np.isnan(g["within"])):
                continue
            for f, w, d in zip(features, g["within"], g["dataset"], strict=True):
                rows.append({"model": model_name, "direction": direction, "class": cls,
                             "feature": f, "within": round(float(w), 4),
                             "dataset": round(float(d), 4)})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    ap.add_argument("--cached", action="store_true")
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
    sets = r2.feature_sets(features)

    need = [m for m in args.models
            if not (args.cached and (args.out_dir / f"blocks_{m}.joblib").exists())]
    if need:
        r2.lean_parquet_reads()
        c17, c18 = load_ids2017(), load_ids2018()
        logger.info("rows: 2017 %d, 2018 %d", len(c17.X), len(c18.X))

    summary_path = args.out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary.update({"preregistration_sha256": prereg_sha, "seeds": list(seeds),
                    "n_draws": r2.N_DRAWS, "draw_seed": r2.DRAW_SEED,
                    "feature_sets": sets, "decision_set": "consensus"})
    summary.setdefault("models", {})

    def upsert(path: Path, table: pd.DataFrame, model_name: str) -> None:
        if path.exists():
            old_t = pd.read_csv(path)
            table = pd.concat([old_t[old_t["model"] != model_name], table], ignore_index=True)
        table.to_csv(path, index=False)

    for model_name in args.models:
        t0 = time.perf_counter()
        old = joblib.load(old_cache(model_name))
        cache = args.out_dir / f"blocks_{model_name}.joblib"
        if args.cached and cache.exists():
            new = joblib.load(cache)
        else:
            partial = args.out_dir / f"blocks_{model_name}.partial.joblib"
            new = collect(model_name, c17, c18, features, seeds, old, partial)
            joblib.dump(new, cache, compress=3)
            partial.unlink(missing_ok=True)

        blocks = merged(old, new)
        counts = counts_table(new, model_name)
        table = null_test(model_name, blocks, counts, sets, features, seeds)
        upsert(args.out_dir / "counts.csv", counts, model_name)
        upsert(args.out_dir / "null_test.csv", table, model_name)
        upsert(args.out_dir / "feature_diff.csv", feature_diff(model_name, blocks, features, seeds),
               model_name)
        summary["models"][model_name] = {
            **{s: judge(table, s) for s in sets},
            "minutes": round((time.perf_counter() - t0) / 60, 1),
        }
        logger.info("%s: %s", model_name,
                    {s: v["overall"] for s, v in summary["models"][model_name].items()
                     if isinstance(v, dict)})
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    logger.info("done -> %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
