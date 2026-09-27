"""SHAP check of the feature roles, and SHAP on missed attacks after transfer.

Rules are fixed in results/crossdataset/importance_comparison/SHAP_PREREGISTRATION.md.

Part A repeats dataset_vs_attack_importance.py with mean |SHAP| instead of
permutation importance and compares the roles (Cohen's kappa).
Part B trains the protocol LightGBM on one corpus, takes the attacks in the
other corpus that it predicts as Benign, and measures how much of the push
toward Benign comes from fingerprint / entangled features.

Run::

    python scripts/shap_importance.py
    python scripts/shap_importance.py --smoke     # 1 seed, small sample, for testing only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import adversarial_validation as av  # noqa: E402
import dataset_vs_attack_importance as dva  # noqa: E402
from run_crossdataset import partition  # noqa: E402

from src.crossdataset.labels import SHARED_CLASSES  # noqa: E402
from src.crossdataset.loaders import load_ids2017, load_ids2018  # noqa: E402
from src.ids2018.models import build_model, fit_model  # noqa: E402
from src.ids2018.preprocessing import Ids2018Preprocessor  # noqa: E402

logger = logging.getLogger("shap_importance")

OUT_DIR = PROJECT_ROOT / "results" / "crossdataset" / "importance_comparison" / "shap"
ROLES_CSV = PROJECT_ROOT / "results" / "crossdataset" / "importance_comparison" / "feature_roles.csv"

SEEDS = (42, 43, 44, 45, 46)
MODELS = ("lightgbm", "logistic_regression")
BACKGROUND_ROWS = 200
EXPLAIN_ROWS = 2_000
MISSED_PER_CLASS = 1_000
MIN_MISSED = 100
MIN_CAUGHT = 30  # same as MEASURABLE_MIN in run_crossdataset.py
SEEDS_NEEDED = 4

# Decision thresholds, copied from the pre-registration.
KAPPA_CONFIRMED = 0.60
KAPPA_OVERTURNED = 0.40
GAP_SUPPORTED = 0.10
GAP_NOT_SUPPORTED = 0.02


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def sample_rows(X, y, n: int, seed: int):
    """Up to n rows, stratified by y when possible."""
    if len(X) <= n:
        return X, y
    try:
        Xs, _, ys, _ = train_test_split(X, y, train_size=n, random_state=seed, stratify=y)
    except ValueError:
        # a class too small to stratify; fall back to a plain random draw
        idx = np.random.default_rng(seed).choice(len(X), size=n, replace=False)
        Xs, ys = X[idx], y[idx]
    return Xs, ys


def shap_array(values, n_rows: int, n_features: int) -> np.ndarray:
    """Return SHAP values as (rows, features, outputs) whatever shap gives back."""
    if isinstance(values, list):
        arr = np.stack(values, axis=-1)
    else:
        arr = np.asarray(values)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[0] != n_rows or arr.shape[1] != n_features:
        raise ValueError(f"unexpected SHAP shape {arr.shape}")
    return arr


def explain(model, model_name: str, background: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Path-dependent SHAP for LightGBM, interventional (linear) for logistic regression.

    Interventional tree SHAP took 9 minutes for one 2,000-row explanation in the
    smoke test, which puts the full run near 20 hours, so LightGBM uses the
    path-dependent mode (see the pre-registration change log).
    """
    if model_name == "logistic_regression":
        # max_samples set explicitly: shap otherwise cuts the background to 100 rows
        masker = shap.maskers.Independent(background, max_samples=len(background))
        explainer = shap.LinearExplainer(model, masker)
        values = explainer.shap_values(rows)
    else:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        values = explainer.shap_values(rows, check_additivity=False)
    return shap_array(values, rows.shape[0], rows.shape[1])


def shares(sv: np.ndarray, names: list[str], features: list[str]) -> pd.Series:
    """Mean |SHAP| summed over outputs, as a share of the total."""
    imp = pd.Series(np.abs(sv).sum(axis=2).mean(axis=0), index=names)
    imp = imp.reindex(features).fillna(0.0)
    total = imp.sum()
    return imp / total if total > 0 else imp


def carries(per_seed: pd.DataFrame, n_features: int) -> pd.Series:
    """Share >= 1/n on the seed mean and in at least 4 of 5 seeds."""
    floor = 1.0 / n_features
    need = min(SEEDS_NEEDED, per_seed.shape[1])
    return (per_seed.mean(axis=1) >= floor) & ((per_seed >= floor).sum(axis=1) >= need)


# ----------------------------------------------------------------------
# Part A
# ----------------------------------------------------------------------
def dataset_side(c17, c18, features, seeds) -> tuple[pd.Series, pd.DataFrame]:
    """SHAP shares of the adversarial LightGBM, per shared class."""
    per_class = {}
    for name in SHARED_CLASSES:
        a = c17.X[(c17.y == name).to_numpy()]
        b = c18.X[(c18.y == name).to_numpy()]
        if len(a) < av.MIN_PER_SIDE or len(b) < av.MIN_PER_SIDE:
            logger.info("  dataset side: skip %s (%d / %d rows)", name, len(a), len(b))
            continue
        runs = {}
        for seed in seeds:
            X, y, n = av.balanced_frame(a, b, features, seed)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=av.TEST_SIZE, random_state=seed, stratify=y
            )
            model = av.build_classifier(seed).fit(X_tr, y_tr)
            bg, _ = sample_rows(X_tr.to_numpy(), y_tr, BACKGROUND_ROWS, seed)
            rows, _ = sample_rows(X_te.to_numpy(), y_te, EXPLAIN_ROWS, seed)
            sv = explain(model, "lightgbm", bg, rows)
            runs[seed] = shares(sv, features, features)
            logger.info("  dataset side %-12s seed %d done (n=%d per side)", name, seed, n)
        per_class[name] = pd.DataFrame(runs)

    table = pd.DataFrame({c: df.mean(axis=1) for c, df in per_class.items()})
    carried = pd.DataFrame({c: carries(df, len(features)) for c, df in per_class.items()})
    return carried.any(axis=1), table


def attack_side(corpus, features, model_name, seeds, corpus_name) -> pd.DataFrame:
    """SHAP shares of one protocol model's attack classifier, per seed."""
    encoder = LabelEncoder().fit(np.array(SHARED_CLASSES))
    y_codes = encoder.transform(corpus.y.astype(str))
    runs = {}
    for seed in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(
            corpus.X[features], y_codes, test_size=dva.TEST_SIZE, random_state=seed,
            stratify=y_codes,
        )
        pre = Ids2018Preprocessor()
        Xt = pre.fit_transform(X_tr)
        Xe = pre.transform(X_te)
        kept = list(pre.feature_names)

        model = fit_model(model_name, dva.seed_model(build_model(model_name), seed), Xt, y_tr)
        bg, _ = sample_rows(Xt, y_tr, BACKGROUND_ROWS, seed)
        rows, _ = sample_rows(Xe, y_te, EXPLAIN_ROWS, seed)
        t0 = time.perf_counter()
        sv = explain(model, model_name, bg, rows)
        runs[seed] = shares(sv, kept, features)
        logger.info(
            "  attack side %-20s %-8s seed %d done (SHAP %.0fs)",
            model_name, corpus_name, seed, time.perf_counter() - t0,
        )
    return pd.DataFrame(runs)


def part_a(c17, c18, features, seeds) -> dict:
    logger.info("Part A: dataset side (adversarial LightGBM)")
    ds_carries, ds_table = dataset_side(c17, c18, features, seeds)

    old = pd.read_csv(ROLES_CSV, index_col="feature")
    out = pd.DataFrame(index=features)
    out["dataset_shap_max_class_share"] = ds_table.max(axis=1)
    out["dataset_carries"] = ds_carries

    result = {"roles": {}, "kappa": {}, "spearman": {}}
    for model_name in MODELS:
        logger.info("Part A: attack side, %s", model_name)
        attack_any = pd.Series(False, index=features)
        for corpus_name, corpus in (("ids2017", c17), ("ids2018", c18)):
            per_seed = attack_side(corpus, features, model_name, seeds, corpus_name)
            out[f"attack_{model_name}_{corpus_name}_share"] = per_seed.mean(axis=1)
            c = carries(per_seed, len(features))
            out[f"attack_{model_name}_{corpus_name}_carries"] = c
            attack_any |= c
            perm = old[f"attack_{model_name}_{corpus_name}"].reindex(features)
            rho = spearmanr(per_seed.mean(axis=1), perm).statistic
            result["spearman"][f"{model_name}_{corpus_name}"] = round(float(rho), 4)

        roles = [dva.quadrant(d, a) for d, a in zip(ds_carries, attack_any, strict=True)]
        out[f"role_{model_name}_shap"] = roles
        out[f"role_{model_name}_perm"] = old[f"role_{model_name}"].reindex(features)
        kappa = cohen_kappa_score(out[f"role_{model_name}_perm"], out[f"role_{model_name}_shap"])
        result["kappa"][model_name] = round(float(kappa), 4)
        result["roles"][model_name] = out[f"role_{model_name}_shap"].value_counts().to_dict()

        moved = out[out[f"role_{model_name}_perm"] != out[f"role_{model_name}_shap"]]
        result.setdefault("moved", {})[model_name] = [
            {"feature": f, "perm": r[f"role_{model_name}_perm"], "shap": r[f"role_{model_name}_shap"]}
            for f, r in moved.iterrows()
        ]

    k = result["kappa"].values()
    if all(v >= KAPPA_CONFIRMED for v in k):
        verdict = "confirmed"
    elif any(v < KAPPA_OVERTURNED for v in k):
        verdict = "not confirmed"
    else:
        verdict = "partial"
    result["verdict"] = verdict
    result["table"] = out
    result["dataset_per_class"] = ds_table
    logger.info("Part A kappa %s -> %s", result["kappa"], verdict)
    return result


# ----------------------------------------------------------------------
# Part B
# ----------------------------------------------------------------------
def representative_map(columns: list[str]) -> dict[str, str]:
    """Map each of the 77 columns to its representative among the 60."""
    reps = json.loads((av.OUT_DIR / "findings.json").read_text(encoding="utf-8"))
    mapping = {c: c for c in columns}
    for rep, members in reps["duplicate_representatives"].items():
        for m in members:
            mapping[m] = rep
    return mapping


def benign_push_share(sv_benign: np.ndarray, names: list[str], mapping, F: set[str]) -> np.ndarray:
    """Per row: positive Benign SHAP from features in F / positive Benign SHAP from all.

    Duplicate columns are summed into their representative first. A row with
    no positive Benign SHAP at all gets NaN.
    """
    frame = pd.DataFrame(sv_benign, columns=names).T.groupby(lambda c: mapping.get(c, c)).sum().T
    pos = frame.clip(lower=0)
    total = pos.sum(axis=1)
    in_f = pos[[c for c in pos.columns if c in F]].sum(axis=1)
    return (in_f / total.where(total > 0)).to_numpy()


def cap_per_class(idx: np.ndarray, labels: pd.Series, cap: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for c in pd.unique(labels):
        pos = idx[(labels == c).to_numpy()]
        if len(pos) > cap:
            pos = rng.choice(pos, size=cap, replace=False)
        out.append(pos)
    return np.concatenate(out) if out else idx[:0]


def part_b(c17, c18, role_sets: dict[str, set[str]], seeds) -> pd.DataFrame:
    """One row per (direction, seed, class, role table)."""
    encoder = LabelEncoder().fit(np.array(SHARED_CLASSES))
    benign = int(encoder.transform(["Benign"])[0])
    corpora = {"ids2017": c17, "ids2018": c18}
    parts = {n: partition(c, "chronological", 0) for n, c in corpora.items()}
    mapping = representative_map(list(c17.X.columns))

    rows = []
    for source, target in (("ids2017", "ids2018"), ("ids2018", "ids2017")):
        src, tgt = corpora[source], corpora[target]
        ps, pt = parts[source], parts[target]
        for seed in seeds:
            pre = Ids2018Preprocessor()
            Xt = pre.fit_transform(src.X.iloc[ps.train])
            yt = encoder.transform(src.y.iloc[ps.train])
            names = list(pre.feature_names)
            model = fit_model("lightgbm", dva.seed_model(build_model("lightgbm"), seed), Xt, yt)
            bg, _ = sample_rows(Xt, yt, BACKGROUND_ROWS, seed)

            # missed: attacks in the target test split predicted Benign
            Xe_t = pre.transform(tgt.X.iloc[pt.test])
            y_t = tgt.y.iloc[pt.test].astype(str).reset_index(drop=True)
            pred_t = model.predict(Xe_t)
            missed_mask = (y_t != "Benign").to_numpy() & (pred_t == benign)
            missed_counts = y_t[missed_mask].value_counts()

            # caught: same classes in the source test split, predicted correctly
            Xe_s = pre.transform(src.X.iloc[ps.test])
            y_s = src.y.iloc[ps.test].astype(str).reset_index(drop=True)
            pred_s = encoder.inverse_transform(model.predict(Xe_s))
            caught_mask = (y_s != "Benign").to_numpy() & (pred_s == y_s.to_numpy())

            # SHAP once per group; the share s is then cheap for each role table
            groups = {}
            for gname, mask, X_all, y_all in (
                ("missed", missed_mask, Xe_t, y_t),
                ("caught", caught_mask, Xe_s, y_s),
            ):
                idx = np.flatnonzero(mask)
                idx = cap_per_class(idx, y_all.iloc[idx].reset_index(drop=True), MISSED_PER_CLASS, seed)
                labels = y_all.iloc[idx].to_numpy()
                if len(idx) == 0:
                    groups[gname] = (None, labels)
                    continue
                t0 = time.perf_counter()
                sv = explain(model, "lightgbm", bg, X_all[idx])
                groups[gname] = (sv[:, :, benign], labels)
                logger.info(
                    "  %s->%s seed %d %s: %d rows (SHAP %.0fs)",
                    source[-4:], target[-4:], seed, gname, len(idx), time.perf_counter() - t0,
                )

            for table_name, F in role_sets.items():
                s = {g: (benign_push_share(v, names, mapping, F) if v is not None else np.empty(0), lab)
                     for g, (v, lab) in groups.items()}
                for cls in SHARED_CLASSES:
                    if cls == "Benign":
                        continue
                    m = s["missed"][0][s["missed"][1] == cls]
                    c = s["caught"][0][s["caught"][1] == cls]
                    rows.append({
                        "direction": f"{source}->{target}",
                        "roles_from": table_name,
                        "seed": seed,
                        "class": cls,
                        "n_missed_total": int(missed_counts.get(cls, 0)),
                        "n_missed_explained": len(m),
                        "n_caught_explained": len(c),
                        "s_missed": float(np.nanmean(m)) if len(m) else np.nan,
                        "s_caught": float(np.nanmean(c)) if len(c) else np.nan,
                    })
            del model

    table = pd.DataFrame(rows)
    table["gap"] = table["s_missed"] - table["s_caught"]
    return table.round(4)


def judge_b(table: pd.DataFrame, n_seeds: int) -> dict:
    main = table[table["direction"] == "ids2017->ids2018"]
    judged, supported, not_supported = [], 0, 0
    for cls, g in main.groupby("class"):
        if (len(g) < n_seeds or (g["n_missed_total"] < MIN_MISSED).any()
                or (g["n_caught_explained"] < MIN_CAUGHT).any() or g["gap"].isna().any()):
            continue
        judged.append(cls)
        if (g["gap"] >= GAP_SUPPORTED).all():
            supported += 1
        if (g["gap"].mean() <= GAP_NOT_SUPPORTED):
            not_supported += 1
    n = len(judged)
    if n and supported > n / 2:
        verdict = "fingerprint causes the misses"
    elif n and not_supported > n / 2:
        verdict = "not supported"
    else:
        verdict = "partial"
    return {"verdict": verdict, "judged_classes": judged,
            "n_supported": supported, "n_not_supported": not_supported}


# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--smoke", action="store_true",
                    help="1 seed and 20,000 rows per corpus, only to test the code")
    ap.add_argument("--part", choices=["a", "b", "both"], default="both")
    ap.add_argument("--ids2018-path", type=Path, default=None,
                    help="other 2018 parquet, for --smoke testing only")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(args.out_dir / "run.log", encoding="utf-8")],
    )

    seeds = SEEDS[:1] if args.smoke else SEEDS
    target = 20_000 if args.smoke else 300_000
    if args.ids2018_path and not args.smoke:
        ap.error("--ids2018-path is only for --smoke; real runs use the full corpus")
    if args.smoke and args.ids2018_path:
        # the loader refuses pre-sampled files on purpose; allowed only for a code test
        import src.crossdataset.loaders as loaders
        loaders.MIN_FULL_CORPUS_ROWS = 0
    c17 = load_ids2017(target=target)
    c18 = load_ids2018(target=target, **({"path": args.ids2018_path} if args.ids2018_path else {}))
    features = dva.features_from_findings()
    logger.info("rows: 2017 %d, 2018 %d | features %d | seeds %s", len(c17.X), len(c18.X),
                len(features), list(seeds))

    prereg = OUT_DIR.parent / "SHAP_PREREGISTRATION.md"
    prereg_sha = hashlib.sha256(prereg.read_bytes()).hexdigest()
    logger.info("pre-registration sha256 %s", prereg_sha)
    summary = {"smoke": args.smoke, "seeds": list(seeds), "n_features": len(features),
               "background_rows": BACKGROUND_ROWS, "explain_rows": EXPLAIN_ROWS,
               "preregistration_sha256": prereg_sha}
    started = time.perf_counter()

    if args.part in ("a", "both"):
        a = part_a(c17, c18, features, seeds)
        a["table"].to_csv(args.out_dir / "shap_roles.csv", index_label="feature")
        a["dataset_per_class"].to_csv(args.out_dir / "dataset_shap_per_class.csv",
                                      index_label="feature")
        summary["part_a"] = {k: v for k, v in a.items()
                             if k not in ("table", "dataset_per_class")}

    if args.part in ("b", "both"):
        tab = pd.read_csv(args.out_dir / "shap_roles.csv", index_col="feature")
        verdict_a = json.loads((args.out_dir / "summary.json").read_text(encoding="utf-8"))[
            "part_a"]["verdict"] if args.part == "b" else summary["part_a"]["verdict"]
        wanted = ["fingerprint only", "entangled"]
        role_sets = {
            "shap": set(tab.index[tab["role_lightgbm_shap"].isin(wanted)]),
            "perm": set(tab.index[tab["role_lightgbm_perm"].isin(wanted)]),
        }
        # the pre-registration uses the SHAP roles unless Part A was not confirmed,
        # in which case both are reported; both are always computed here
        summary["part_b_F"] = {k: sorted(v) for k, v in role_sets.items()}
        table = part_b(c17, c18, role_sets, seeds)
        table.to_csv(args.out_dir / "missed_attacks.csv", index=False)
        summary["part_b"] = {
            name: judge_b(table[table["roles_from"] == name], len(seeds)) for name in role_sets
        }
        summary["part_b"]["main_roles"] = "both" if verdict_a == "not confirmed" else "shap"
        logger.info("Part B -> %s", summary["part_b"])

    summary["minutes"] = round((time.perf_counter() - started) / 60, 1)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                               encoding="utf-8")
    logger.info("done in %.1f min -> %s", summary["minutes"], args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
