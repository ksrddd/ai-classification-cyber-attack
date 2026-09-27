"""SHAP round 3: rounds 1-2 on random forest, XGBoost and CatBoost.

Rules are in results/crossdataset/importance_comparison/SHAP_ROUND3_PREREGISTRATION.md.
Part B is round 2 (consensus F, random-set null) with a different model.
Part A is the round 1 attack side, compared with the LightGBM SHAP roles.

Run::

    python scripts/shap_round3.py
    python scripts/shap_round3.py --models xgboost          # one model
    python scripts/shap_round3.py --cached                  # reuse blocks_<model>.joblib
    python scripts/shap_round3.py --smoke --out-dir <dir>   # 1 seed, small sample, code test only
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
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import dataset_vs_attack_importance as dva  # noqa: E402
import shap_importance as r1  # noqa: E402
import shap_round2 as r2  # noqa: E402
from run_crossdataset import partition  # noqa: E402

from src.crossdataset.labels import SHARED_CLASSES  # noqa: E402
from src.crossdataset.loaders import load_ids2017, load_ids2018  # noqa: E402
from src.ids2018.models import build_model, fit_model  # noqa: E402
from src.ids2018.preprocessing import Ids2018Preprocessor  # noqa: E402

logger = logging.getLogger("shap_round3")

CMP = r2.CMP
OUT_DIR = CMP / "shap_round3"
PREREG = CMP / "SHAP_ROUND3_PREREGISTRATION.md"

MODELS = ("random_forest", "xgboost", "catboost")
_round1_explain = r1.explain


def explain(model, model_name: str, background, rows: np.ndarray) -> np.ndarray:
    """Round 1 explainer, except CatBoost uses its own SHAP.

    shap.TreeExplainer crashes with an access violation while reading a
    CatBoost model (shap 0.52.0, catboost 1.2.10). CatBoost's ShapValues are
    exact tree SHAP too; the last column is the expected value and is dropped.
    """
    if model_name != "catboost":
        return _round1_explain(model, model_name, background, rows)
    from catboost import Pool

    raw = np.asarray(model.get_feature_importance(Pool(rows), type="ShapValues"))
    if raw.ndim == 2:  # binary: (rows, features + 1)
        raw = raw[:, None, :]
    return r1.shap_array(np.transpose(raw[:, :, :-1], (0, 2, 1)), rows.shape[0], rows.shape[1])


# round 1 Part A code calls r1.explain, so point it here
r1.explain = explain


# ----------------------------------------------------------------------
# Part A (description only)
# ----------------------------------------------------------------------
def part_a(model_name, c17, c18, features, seeds, lgbm: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = pd.DataFrame(index=features)
    dataset = lgbm["dataset_carries"].reindex(features).fillna(False).astype(bool)
    attack_any = pd.Series(False, index=features)
    for corpus_name, corpus in (("ids2017", c17), ("ids2018", c18)):
        per_seed = r1.attack_side(corpus, features, model_name, seeds, corpus_name)
        out[f"attack_{model_name}_{corpus_name}_share"] = per_seed.mean(axis=1)
        c = r1.carries(per_seed, len(features))
        out[f"attack_{model_name}_{corpus_name}_carries"] = c
        attack_any |= c

    out[f"role_{model_name}_shap"] = [dva.quadrant(d, a)
                                     for d, a in zip(dataset, attack_any, strict=True)]
    return out, part_a_info(model_name, out, features, lgbm)


def part_a_info(model_name, out: pd.DataFrame, features, lgbm: pd.DataFrame) -> dict:
    """Summary of one model's Part A table; also rebuilds it from part_a_roles.csv."""
    info = {"spearman_vs_lightgbm": {}}
    attack_any = pd.Series(False, index=features)
    for corpus_name in ("ids2017", "ids2018"):
        share = out[f"attack_{model_name}_{corpus_name}_share"].reindex(features)
        rho = spearmanr(share, lgbm[f"attack_lightgbm_{corpus_name}_share"].reindex(features))
        info["spearman_vs_lightgbm"][corpus_name] = round(float(rho.statistic), 4)
        carries = out[f"attack_{model_name}_{corpus_name}_carries"].reindex(features)
        attack_any |= carries.astype(bool)
    roles = out[f"role_{model_name}_shap"].reindex(features)
    lgbm_roles = lgbm["role_lightgbm_shap"].reindex(features)
    info["kappa_vs_lightgbm"] = round(float(cohen_kappa_score(lgbm_roles, roles)), 4)
    info["roles"] = roles.value_counts().to_dict()
    info["n_attack_carriers"] = int(attack_any.sum())
    info["differs_from_lightgbm"] = [
        {"feature": f, "lightgbm": lgbm_roles[f], model_name: roles[f]}
        for f in features if lgbm_roles[f] != roles[f]
    ]
    logger.info("Part A %s: kappa vs LightGBM %.3f, spearman %s", model_name,
                info["kappa_vs_lightgbm"], info["spearman_vs_lightgbm"])
    return info


# ----------------------------------------------------------------------
# Part B (the decision)
# ----------------------------------------------------------------------
def collect(model_name, c17, c18, features, seeds, checkpoint: Path | None = None) -> list[dict]:
    """Round 2 collect() with the model as a parameter.

    With ``checkpoint``, the blocks are saved after every (direction, seed), and
    pairs already in the file are skipped. Each pair fits its own model from a
    fixed split and seed, so a resumed run gives the same blocks.
    """
    encoder = LabelEncoder().fit(np.array(SHARED_CLASSES))
    benign = int(encoder.transform(["Benign"])[0])
    corpora = {"ids2017": c17, "ids2018": c18}
    parts = {n: partition(c, "chronological", 0) for n, c in corpora.items()}
    mapping = r1.representative_map(list(c17.X.columns))

    blocks = joblib.load(checkpoint) if checkpoint and checkpoint.exists() else []
    done = {(b["direction"], b["seed"]) for b in blocks}
    for source, target in (("ids2017", "ids2018"), ("ids2018", "ids2017")):
        src, tgt = corpora[source], corpora[target]
        ps, pt = parts[source], parts[target]
        for seed in seeds:
            if (f"{source}->{target}", seed) in done:
                logger.info("  %s %s->%s seed %d already in checkpoint", model_name,
                            source[-4:], target[-4:], seed)
                continue
            t0 = time.perf_counter()
            pre = Ids2018Preprocessor()
            Xt = pre.fit_transform(src.X.iloc[ps.train])
            yt = encoder.transform(src.y.iloc[ps.train])
            names = list(pre.feature_names)
            model = fit_model(model_name, dva.seed_model(build_model(model_name), seed), Xt, yt)

            Xe_t = pre.transform(tgt.X.iloc[pt.test])
            y_t = tgt.y.iloc[pt.test].astype(str).reset_index(drop=True)
            pred_t = np.asarray(model.predict(Xe_t)).ravel()
            missed_mask = (y_t != "Benign").to_numpy() & (pred_t == benign)
            missed_counts = y_t[missed_mask].value_counts()

            Xe_s = pre.transform(src.X.iloc[ps.test])
            y_s = src.y.iloc[ps.test].astype(str).reset_index(drop=True)
            pred_s = encoder.inverse_transform(np.asarray(model.predict(Xe_s)).ravel().astype(int))
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
                    sv = explain(model, model_name, None, X_all[idx])
                    pos = r2.positive_benign(sv[:, :, benign], names, mapping, features)
                blocks.append({
                    "direction": f"{source}->{target}", "seed": seed, "group": gname,
                    "labels": y_all.iloc[idx].to_numpy(), "pos": pos,
                    "missed_counts": missed_counts,
                })
            logger.info("  %s %s->%s seed %d done (%.0fs, %d missed attacks)", model_name,
                        source[-4:], target[-4:], seed, time.perf_counter() - t0,
                        int(missed_mask.sum()))
            del model
            if checkpoint:
                joblib.dump(blocks, checkpoint, compress=3)
    return blocks


def null_test(blocks, sets, features, seeds) -> pd.DataFrame:
    """Round 2 null test for one model. The generator restarts at DRAW_SEED."""
    rng = np.random.default_rng(r2.DRAW_SEED)
    rows = []
    for set_name, F in sets.items():
        k = len(F)
        real = np.isin(features, F)[None, :].astype(float)
        draws = np.zeros((r2.N_DRAWS, len(features)))
        for i in range(r2.N_DRAWS):
            draws[i, rng.choice(len(features), size=k, replace=False)] = 1.0
        for direction in ("ids2017->ids2018", "ids2018->ids2017"):
            for cls in SHARED_CLASSES:
                if cls == "Benign":
                    continue
                G, info = r2.class_gaps(blocks, direction, cls, real, seeds)
                G_null, _ = r2.class_gaps(blocks, direction, cls, draws, seeds)
                G = float(G[0])
                eligible = (min(info["n_missed_total"]) >= r1.MIN_MISSED
                            and min(info["n_caught"]) >= r1.MIN_CAUGHT and not np.isnan(G))
                p = (float((1 + np.sum(G_null >= G)) / (1 + r2.N_DRAWS))
                     if not np.isnan(G) else np.nan)
                rows.append({
                    "set": set_name, "set_size": k, "direction": direction, "class": cls,
                    "eligible": eligible, "G": round(G, 4),
                    "null_mean": round(float(np.nanmean(G_null)), 4),
                    "null_95": round(float(np.nanpercentile(G_null, 95)), 4),
                    "p": round(p, 4),
                    "min_missed_total": min(info["n_missed_total"]),
                    "min_caught": min(info["n_caught"]),
                })
    return pd.DataFrame(rows)


def judge(table: pd.DataFrame, set_name: str) -> dict:
    main = table[(table["set"] == set_name) & (table["direction"] == "ids2017->ids2018")
                 & table["eligible"]]
    n = len(main)
    per_class = {}
    for _, r in main.iterrows():
        if r["G"] > 0 and r["p"] < r2.ALPHA / n:
            per_class[r["class"]] = "supported"
        elif r["p"] >= r2.ALPHA:
            per_class[r["class"]] = "not supported"
        else:
            per_class[r["class"]] = "unclear"
    sup = sum(v == "supported" for v in per_class.values())
    nos = sum(v == "not supported" for v in per_class.values())
    overall = ("fingerprint causes the misses" if n and sup > n / 2
               else "not supported" if n and nos > n / 2 else "partial")
    return {"overall": overall, "per_class": per_class}


def overall(per_model: dict[str, str]) -> str:
    votes = list(per_model.values())
    if sum(v == "fingerprint causes the misses" for v in votes) >= 2:
        return "fingerprint causes the misses"
    if sum(v == "not supported" for v in votes) >= 2:
        return "not supported"
    return "partial"


# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    ap.add_argument("--part", choices=["a", "b", "both"], default="both")
    ap.add_argument("--cached", action="store_true",
                    help="reuse blocks_<model>.joblib instead of refitting Part B")
    ap.add_argument("--smoke", action="store_true",
                    help="1 seed and 20,000 rows per corpus, only to test the code")
    args = ap.parse_args(argv)
    if args.smoke and args.out_dir == OUT_DIR:
        ap.error("--smoke must write to another --out-dir")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(args.out_dir / "run.log", encoding="utf-8")],
    )
    prereg_sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    logger.info("pre-registration sha256 %s", prereg_sha)

    seeds = r1.SEEDS[:1] if args.smoke else r1.SEEDS
    features = dva.features_from_findings()
    sets = r2.feature_sets(features)
    lgbm = pd.read_csv(CMP / "shap" / "shap_roles.csv", index_col="feature")
    logger.info("models %s | seeds %s | feature sets %s", args.models, list(seeds),
                {k: len(v) for k, v in sets.items()})

    started = time.perf_counter()
    summary_path = args.out_dir / "summary.json"
    summary = (json.loads(summary_path.read_text(encoding="utf-8"))
               if summary_path.exists() else {})
    summary.update({"preregistration_sha256": prereg_sha, "smoke": args.smoke,
                    "seeds": list(seeds), "n_draws": r2.N_DRAWS, "draw_seed": r2.DRAW_SEED,
                    "feature_sets": sets, "decision_set": "consensus"})
    summary.setdefault("part_a", {})
    summary.setdefault("part_b", {})

    needs_data = args.part in ("a", "both") or not args.cached or any(
        not (args.out_dir / f"blocks_{m}.joblib").exists() for m in args.models)
    if needs_data:
        r2.lean_parquet_reads()
        target = 20_000 if args.smoke else 300_000
        c17, c18 = load_ids2017(target=target), load_ids2018(target=target)
        logger.info("rows: 2017 %d, 2018 %d", len(c17.X), len(c18.X))

    roles_csv = args.out_dir / "part_a_roles.csv"
    roles_table = (pd.read_csv(roles_csv, index_col="feature") if roles_csv.exists()
                   else pd.DataFrame(index=features))
    null_csv = args.out_dir / "null_test.csv"
    null_all = pd.read_csv(null_csv) if null_csv.exists() else pd.DataFrame()

    for model_name in args.models:
        t0 = time.perf_counter()
        if args.part in ("a", "both"):
            logger.info("Part A: %s", model_name)
            table, info = part_a(model_name, c17, c18, features, seeds, lgbm)
            roles_table = roles_table.drop(columns=table.columns, errors="ignore").join(table)
            roles_table.to_csv(roles_csv, index_label="feature")
            summary["part_a"][model_name] = info
        elif model_name not in summary["part_a"] and f"role_{model_name}_shap" in roles_table:
            # Part A finished in an earlier run that stopped before writing the summary
            summary["part_a"][model_name] = part_a_info(model_name, roles_table, features, lgbm)
            logger.info("Part A %s rebuilt from %s", model_name, roles_csv.name)

        if args.part in ("b", "both"):
            cache = args.out_dir / f"blocks_{model_name}.joblib"
            if args.cached and cache.exists():
                blocks = joblib.load(cache)
                logger.info("loaded %d SHAP blocks from %s", len(blocks), cache)
            else:
                logger.info("Part B: %s", model_name)
                partial = args.out_dir / f"blocks_{model_name}.partial.joblib"
                blocks = collect(model_name, c17, c18, features, seeds, partial)
                joblib.dump(blocks, cache, compress=3)
                partial.unlink(missing_ok=True)
            table = null_test(blocks, sets, features, seeds)
            table.insert(0, "model", model_name)
            if len(null_all):
                null_all = null_all[null_all["model"] != model_name]
            null_all = pd.concat([null_all, table], ignore_index=True)
            null_all.to_csv(null_csv, index=False)
            summary["part_b"][model_name] = {s: judge(table, s) for s in sets}
            logger.info("Part B %s: %s", model_name,
                        {s: v["overall"] for s, v in summary["part_b"][model_name].items()})

        summary.setdefault("minutes_per_model", {})[model_name] = round(
            (time.perf_counter() - t0) / 60, 1)
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    judged = {m: v["consensus"]["overall"] for m, v in summary["part_b"].items()}
    if set(judged) == set(MODELS):
        summary["overall"] = overall(judged)
        logger.info("round 3 overall (consensus): %s %s", summary["overall"], judged)
    summary["minutes_this_run"] = round((time.perf_counter() - started) / 60, 1)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("done in %.1f min -> %s", summary["minutes_this_run"], args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
