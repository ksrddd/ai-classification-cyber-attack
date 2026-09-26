"""Which features name the corpus, and which of those does the attack classifier need?

Adversarial validation ranked the features by how much each helps a classifier
tell CICIDS2017 from CSE-CIC-IDS2018 -- dataset importance. The ablation then
dropped the top of that ranking and transfer got *worse*, with the tree
ensembles losing up to 20 points and the linear and neural models not moving.
Both results point at the same unanswered question: are the columns that name
the corpus the same columns the attack classifier depends on?

If they are separate, the fingerprint is a nuisance that could in principle be
removed. If they overlap, removing it removes attack signal with it, and the
ablation's result stops being surprising. This script measures the second
ranking -- attack importance -- on the same 60 columns and puts the two side by
side.

Two views, because each has a blind spot
----------------------------------------
**Permutation importance** answers "what does this model actually lean on".
Its blind spot is redundancy: shuffle one of two columns that carry the same
information and the model reads the other, so both score near zero. The
adversarial run already showed that problem at full strength -- removing the
top 40 of 60 features left the corpus AUC at 0.985-1.000 -- so a low dataset
score here does not certify a column as clean.

**Univariate separability** has no model and therefore no redundancy to hide
behind. For the dataset side it is the single-column AUC between corpora,
folded so 0.5 is no separation and 1.0 is complete. For the attack side it is
the mutual information between the column and the attack class inside one
corpus. It answers "does this column carry the signal on its own", which is
the question permutation importance cannot.

Attack importance is measured with the protocol's own models, built, reseeded,
preprocessed and fitted exactly as ``run_crossdataset.py`` does it, so the
number describes the classifiers whose transfer the study reports. Two of the
seven: LightGBM, the tree family the ablation hurt most, and logistic
regression, which the ablation left untouched and which transfers best
2018->2017. If the two lean on different columns, that is the ablation result
explained rather than merely observed.

Run::

    python scripts/dataset_vs_attack_importance.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.crossdataset.labels import SHARED_CLASSES  # noqa: E402
from src.crossdataset.loaders import load_ids2017, load_ids2018  # noqa: E402
from src.ids2018.models import build_model, fit_model  # noqa: E402
from src.ids2018.preprocessing import Ids2018Preprocessor  # noqa: E402

logger = logging.getLogger("dataset_vs_attack_importance")

AV_DIR = PROJECT_ROOT / "results" / "crossdataset" / "adversarial_validation"
OUT_DIR = PROJECT_ROOT / "results" / "crossdataset" / "importance_comparison"

SEEDS = (42, 43, 44, 45, 46)
TEST_SIZE = 0.3
#: The two protocol models measured. See the module docstring for why these.
MODELS = ("lightgbm", "logistic_regression")
#: Permutation passes per feature per seed. The spread across five seeds is
#: what the noise floor is built from, so repeats within a seed only need to
#: steady each seed's own estimate.
N_REPEATS = 3
#: Held-out rows scored per permutation pass, drawn stratified from the test
#: split. Sixty features times three repeats times five seeds times two
#: corpora is 1,800 passes per model; scoring all 90,000 test rows each time
#: would cost hours to sharpen a ranking that is already stable at this size.
SCORE_ROWS = 30_000
#: Rows used for mutual information, per corpus. The kNN estimator is
#: quadratic in the sample, and the ranking it produces settles well before
#: this.
MI_ROWS = 30_000
#: A feature "carries" a task when its mean permutation importance is positive
#: and exceeds this many seed-to-seed standard deviations: distinguishable from
#: zero on the spread the measurement itself produced, rather than above a
#: threshold picked by eye.
NOISE_FLOOR_SD = 2.0
#: The ablation's own K values, so the overlap can be read against the
#: experiment it explains.
ABLATION_K = (5, 10, 20)


def features_from_findings() -> list[str]:
    """The exact 60 columns the adversarial run measured over."""
    findings = AV_DIR / "findings.json"
    if not findings.is_file():
        raise FileNotFoundError(
            f"{findings} not found -- run scripts/adversarial_validation.py first."
        )
    return list(json.loads(findings.read_text(encoding="utf-8"))["features"])


def seed_model(model, seed: int):
    """Point every ``random_state`` inside ``model`` at ``seed``.

    As in ``run_crossdataset.py``: ``build_model`` bakes in the project's
    RANDOM_STATE, so without this the five seeds would vary the split but not
    the model, and the spread the noise floor is built from would be too small.
    """
    keys = [
        k
        for k in model.get_params(deep=True)
        if k.split("__")[-1] in ("random_state", "random_seed")
    ]
    if keys:
        model.set_params(**dict.fromkeys(keys, seed))
    return model


def dataset_importance(features: list[str]) -> pd.DataFrame:
    """The adversarial run's permutation importance, read rather than recomputed.

    One number per feature: the mean over the six testable classes. Alongside
    it, whether the feature clears the noise floor in any single class -- a
    column that names the corpus inside Web Attack alone is still a fingerprint,
    and averaging it against five classes where it does nothing would hide that.
    """
    per_class = pd.read_csv(AV_DIR / "importance.csv")
    mean = per_class.groupby("feature")["importance"].mean()
    clears = (per_class["importance"] > 0) & (
        per_class["importance"] > NOISE_FLOOR_SD * per_class["sd"]
    )
    carried = per_class[clears].groupby("feature")["shared_class"].apply(list)

    out = pd.DataFrame(index=features)
    out["dataset_perm"] = mean.reindex(features).fillna(0.0)
    out["dataset_carries"] = [f in carried.index for f in features]
    out["dataset_carries_in"] = [carried.get(f, []) for f in features]
    return out


def dataset_univariate(
    c17: pd.DataFrame, y17: pd.Series, c18: pd.DataFrame, y18: pd.Series, features: list[str]
) -> pd.Series:
    """Single-column AUC between corpora, folded to [0.5, 1], mean over classes.

    Within one shared class at a time, as the adversarial run does it, so a
    column cannot score by tracking class composition.
    """
    scores: dict[str, list[float]] = {f: [] for f in features}
    for name in SHARED_CLASSES:
        a = c17[(y17 == name).to_numpy()]
        b = c18[(y18 == name).to_numpy()]
        if len(a) < 500 or len(b) < 500:
            continue
        y = np.r_[np.zeros(len(a), int), np.ones(len(b), int)]
        for f in features:
            s = roc_auc_score(y, np.r_[a[f].to_numpy(), b[f].to_numpy()])
            scores[f].append(max(s, 1 - s))
    return pd.Series({f: float(np.mean(v)) for f, v in scores.items()})


def attack_mutual_information(
    X: pd.DataFrame, y: pd.Series, features: list[str], seed: int = 42
) -> pd.Series:
    """Mutual information between each column and the attack class, in nats."""
    if len(X) > MI_ROWS:
        X, _, y, _ = train_test_split(X, y, train_size=MI_ROWS, random_state=seed, stratify=y)
    mi = mutual_info_classif(
        X[features].to_numpy(), y.to_numpy(), discrete_features=False, random_state=seed
    )
    return pd.Series(mi, index=features)


def attack_importance(
    X: pd.DataFrame, y: pd.Series, features: list[str], model_name: str, corpus: str
) -> pd.DataFrame:
    """Permutation importance of one protocol model's attack classifier.

    Scored on macro-F1, the study's headline metric, so a column is important
    here in the sense the transfer results are reported in. Permuting a column
    after preprocessing is equivalent to permuting it before: the scaler is a
    per-column affine map and the cleaned corpora hold no missing values to
    impute. A column the preprocessor drops as constant on the training split
    cannot be used by the model at all and scores exactly zero.
    """
    encoder = LabelEncoder().fit(np.array(SHARED_CLASSES))
    y_codes = encoder.transform(y.astype(str))

    runs = []
    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X[features], y_codes, test_size=TEST_SIZE, random_state=seed, stratify=y_codes
        )
        pre = Ids2018Preprocessor()
        Xt = pre.fit_transform(X_tr)
        Xe = pre.transform(X_te)
        kept = list(pre.feature_names)

        if len(Xe) > SCORE_ROWS:
            Xe, _, y_te, _ = train_test_split(
                Xe, y_te, train_size=SCORE_ROWS, random_state=seed, stratify=y_te
            )

        model = fit_model(model_name, seed_model(build_model(model_name), seed), Xt, y_tr)
        imp = permutation_importance(
            model,
            Xe,
            y_te,
            scoring="f1_macro",
            n_repeats=N_REPEATS,
            random_state=seed,
            n_jobs=1,
        )
        runs.append(pd.Series(imp.importances_mean, index=kept).reindex(features).fillna(0.0))
        logger.info("    %-20s %-8s seed %d done", model_name, corpus, seed)

    stacked = pd.concat(runs, axis=1)
    return pd.DataFrame({"mean": stacked.mean(axis=1), "sd": stacked.std(axis=1, ddof=0)})


def carries(frame: pd.DataFrame) -> pd.Series:
    """Positive and clear of the noise floor built from the seed spread."""
    return (frame["mean"] > 0) & (frame["mean"] > NOISE_FLOOR_SD * frame["sd"])


def share(values: pd.Series) -> pd.Series:
    """Each feature's share of the total positive importance.

    Permutation importance is a drop in AUC on one side and a drop in macro-F1
    on the other, so the raw numbers are on different scales. Shares are not:
    both answer "what fraction of everything this model leans on is this one
    column".
    """
    positive = values.clip(lower=0)
    total = positive.sum()
    return positive / total if total > 0 else positive


def quadrant(dataset: bool, attack: bool) -> str:
    if dataset and attack:
        return "entangled"
    if dataset:
        return "fingerprint only"
    if attack:
        return "attack only"
    return "neither"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    features = features_from_findings()
    c17, c18 = load_ids2017(), load_ids2018()
    corpora = {"ids2017": c17, "ids2018": c18}

    table = dataset_importance(features)
    table["dataset_univariate_auc"] = dataset_univariate(c17.X, c17.y, c18.X, c18.y, features)

    logger.info("")
    logger.info("mutual information with the attack class, per corpus")
    for name, corpus in corpora.items():
        table[f"attack_mi_{name}"] = attack_mutual_information(corpus.X, corpus.y, features)

    logger.info("")
    logger.info("permutation importance of the attack classifier, macro-F1")
    for model_name in MODELS:
        for name, corpus in corpora.items():
            imp = attack_importance(corpus.X, corpus.y, features, model_name, name)
            table[f"attack_{model_name}_{name}"] = imp["mean"]
            table[f"attack_{model_name}_{name}_sd"] = imp["sd"]
            table[f"attack_{model_name}_{name}_carries"] = carries(imp)

    # One verdict per model: does the feature carry attack signal in EITHER
    # corpus. Either, because a transfer model is trained on one corpus and a
    # column it needs on that side is lost to it if ablated, whichever side
    # that happens to be.
    for model_name in MODELS:
        attack = (
            table[f"attack_{model_name}_ids2017_carries"]
            | table[f"attack_{model_name}_ids2018_carries"]
        )
        table[f"role_{model_name}"] = [
            quadrant(d, a) for d, a in zip(table["dataset_carries"], attack, strict=True)
        ]

    ranked = table.sort_values("dataset_perm", ascending=False)

    # How much of each attack classifier's importance the ablation removed.
    overlap = {}
    for model_name in MODELS:
        for name in corpora:
            col = f"attack_{model_name}_{name}"
            s = share(ranked[col])
            overlap[f"{model_name}_{name}"] = {
                str(k): round(float(s.iloc[:k].sum()), 4) for k in ABLATION_K
            }

    roles = {m: table[f"role_{m}"].value_counts().to_dict() for m in MODELS}

    logger.info("")
    logger.info("=" * 96)
    logger.info("share of attack importance carried by the top-K dataset features")
    for key, by_k in overlap.items():
        logger.info("  %-34s %s", key, "  ".join(f"K={k}: {v:.1%}" for k, v in by_k.items()))
    logger.info("")
    for model_name, counts in roles.items():
        logger.info("  roles (%s): %s", model_name, counts)
    logger.info("")
    logger.info("top 12 by dataset importance")
    head = ranked.head(12)
    for f, r in head.iterrows():
        logger.info(
            "  %-28s ds_perm %.4f  ds_uni %.3f | lgbm17 %+.4f lgbm18 %+.4f | "
            "lr17 %+.4f lr18 %+.4f | %s / %s",
            f,
            r["dataset_perm"],
            r["dataset_univariate_auc"],
            r["attack_lightgbm_ids2017"],
            r["attack_lightgbm_ids2018"],
            r["attack_logistic_regression_ids2017"],
            r["attack_logistic_regression_ids2018"],
            r["role_lightgbm"],
            r["role_logistic_regression"],
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ranked.drop(columns=["dataset_carries_in"]).to_csv(
        args.out_dir / "feature_roles.csv", index_label="feature"
    )
    (args.out_dir / "comparison.json").write_text(
        json.dumps(
            {
                "question": (
                    "Are the columns that name the corpus the same columns the "
                    "attack classifier depends on?"
                ),
                "seeds": list(SEEDS),
                "models": list(MODELS),
                "n_features": len(features),
                "noise_floor_sd": NOISE_FLOOR_SD,
                "attack_scoring": "f1_macro",
                "score_rows": SCORE_ROWS,
                "roles": roles,
                "ablation_overlap": overlap,
                "features": [
                    {
                        "feature": f,
                        "dataset_perm": round(float(r["dataset_perm"]), 6),
                        "dataset_univariate_auc": round(float(r["dataset_univariate_auc"]), 4),
                        "dataset_carries_in": list(r["dataset_carries_in"]),
                        **{
                            f"attack_{m}_{c}": round(float(r[f"attack_{m}_{c}"]), 6)
                            for m in MODELS
                            for c in corpora
                        },
                        **{f"attack_mi_{c}": round(float(r[f"attack_mi_{c}"]), 5) for c in corpora},
                        **{f"role_{m}": r[f"role_{m}"] for m in MODELS},
                    }
                    for f, r in ranked.iterrows()
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("wrote %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
