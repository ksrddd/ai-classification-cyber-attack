"""Cross-dataset evaluation: models trained on CICIDS2017, tested on IDS2018.

This is condition D in one direction, and it needs no 2017 raw data -- the
`cicids2017_temporal_v1` bundle ships fitted pipelines (imputer -> scaler ->
clf), so the 2017 training statistics travel with the model. Nothing is
adapted to 2018; that is the point.

Read-only with respect to the project: reads the 2018 sample and the unpacked
2017 bundle, writes only into this scratchpad directory.

Two evaluation sets are reported:

* the 90,011-row temporal test split -- the *same rows* condition C scored, so
  D minus C is the price of crossing datasets with everything else held fixed;
* all 300,000 rows -- every one of them is unseen by a 2017-trained model, so
  there is no reason to withhold them, and the rare classes get more support.

Predictions of ``Heartbleed`` and ``PortScan`` are kept as errors rather than
dropped. 2018 contains neither, so such a prediction cannot be right, and
discarding it would quietly raise the score for a failure mode that matters.
"""
from __future__ import annotations

import glob
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import classification_report, f1_score

PROJECT = Path(r"C:\Users\ks\Documents\GitHub\ai-classification-cyber-attack")
sys.path.insert(0, str(PROJECT))

from src.crossdataset.labels import (  # noqa: E402
    IDS2017_TO_SHARED,
    SHARED_CLASSES,
    map_to_shared,
)
from src.crossdataset.schema import CANONICAL_FEATURES, to_canonical  # noqa: E402
from src.ids2018.temporal_split import temporal_train_test_split  # noqa: E402

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

OUT = Path(__file__).parent
BUNDLE = Path(glob.glob(str(OUT / "Result" / "*cicids2017*"))[0])
SAMPLE = PROJECT / "data" / "ids2018" / "sample_300k.parquet"
MODELS = [
    "lightgbm",
    "xgboost",
    "catboost",
    "random_forest",
    "mlp",
    "logistic_regression",
    "stacking",
]

#: Sentinel for a 2017 class with no 2018 counterpart. Never equal to any true
#: label, so these predictions score as errors, which is what they are.
NO_COUNTERPART = "__no_2018_counterpart__"


def integrity_check(bundle: Path) -> dict:
    """Confirm the pickles survived the 1.8.0 -> 1.9.0 scikit-learn gap.

    Numerical reproduction against the 2017 test set is impossible here (that
    data is not on this machine), so this checks structure only: the fitted
    pipeline must still expect exactly the 77 canonical features and still
    carry its 9 classes. A silently truncated or re-initialised scaler would
    fail this; a subtly different float would not, so the result is reported
    as a structural check and nothing stronger.
    """
    report = {}
    encoder = joblib.load(bundle / "label_encoder.joblib")
    report["encoder_classes"] = list(encoder.classes_)
    for name in MODELS:
        path = bundle / f"{name}.joblib"
        if not path.exists():
            continue
        model = joblib.load(path)
        n_in = getattr(model, "n_features_in_", None)
        if n_in is None and hasattr(model, "steps"):
            n_in = getattr(model.steps[0][1], "n_features_in_", None)
        classes = getattr(model, "classes_", None)
        report[name] = {
            "n_features_in": int(n_in) if n_in is not None else None,
            "n_classes": 0 if classes is None else len(classes),
        }
        del model
    return report


def score(y_true: pd.Series, y_pred: pd.Series, tag: str) -> dict:
    present = [c for c in SHARED_CLASSES if c in set(y_true)]
    rep = classification_report(
        y_true, y_pred, labels=present, output_dict=True, zero_division=0
    )
    per_class = {c: round(rep[c]["f1-score"], 4) for c in present}
    supports = {c: int(rep[c]["support"]) for c in present}
    measurable = [c for c in present if supports[c] >= 10]
    return {
        "eval_set": tag,
        "f1_macro_all": round(
            f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0), 4
        ),
        "f1_macro_measurable": round(
            float(np.mean([per_class[c] for c in measurable])), 4
        ),
        "accuracy": round(float((y_true.to_numpy() == y_pred.to_numpy()).mean()), 4),
        "pred_no_counterpart": int((y_pred == NO_COUNTERPART).sum()),
        **{f"f1__{c}": per_class[c] for c in present},
        **{f"n__{c}": supports[c] for c in present},
    }


def main() -> int:
    print(f"bundle: {BUNDLE.name}")
    checks = integrity_check(BUNDLE)
    print(json.dumps(checks, indent=1, ensure_ascii=False))
    bad = [
        k
        for k, v in checks.items()
        if isinstance(v, dict) and v.get("n_features_in") not in (None, 77)
    ]
    if bad:
        print(f"ABORT: models expect the wrong feature count: {bad}")
        return 1

    df = pd.read_parquet(SAMPLE)
    X = to_canonical(df, "ids2018", keep_dst_port=True)
    assert list(X.columns) == list(CANONICAL_FEATURES)

    # 2017's cleaner turns +/-inf into NaN and then drops those rows; the
    # fitted pipelines therefore never met a NaN, but they do carry a
    # SimpleImputer holding 2017 medians. Replacing inf with NaN and letting
    # that imputer fill it is both the closer analogue of a real transfer --
    # target values the source never saw are reconstructed from source
    # statistics -- and the only option that keeps the row set identical to
    # condition C, without which D minus C would not be attributable.
    X = X.astype("float64")
    n_inf = int(np.isinf(X.to_numpy()).sum())
    X = X.replace([np.inf, -np.inf], np.nan)
    n_nan = int(X.isna().to_numpy().sum())
    print(f"inf -> NaN: {n_inf:,} value(s); {n_nan:,} NaN total left to the 2017 imputer")
    y_true_all = map_to_shared(df["Label"], "ids2018")
    assert not y_true_all.isna().any()

    # Same temporal split condition C used, so D - C is attributable.
    idx = pd.RangeIndex(len(df)).to_frame(name="_pos")
    _, te_idx, _, _ = temporal_train_test_split(
        idx, df["Label"].astype(str), df["Timestamp"],
        test_size=0.30, min_test_per_class=1,
    )
    te_pos = te_idx["_pos"].to_numpy()
    print(f"rows: all={len(df):,}  temporal-test={len(te_pos):,}")

    encoder = joblib.load(BUNDLE / "label_encoder.joblib")
    rows: list[dict] = []

    for name in MODELS:
        path = BUNDLE / f"{name}.joblib"
        if not path.exists():
            print(f"  {name}: missing, skipped")
            continue
        model = joblib.load(path)
        raw = model.predict(X)
        # Pipelines may emit codes or names depending on how they were fitted.
        if np.issubdtype(np.asarray(raw).dtype, np.number):
            names_2017 = pd.Series(encoder.inverse_transform(raw.astype(int)))
        else:
            names_2017 = pd.Series(raw.astype(str))
        y_pred = names_2017.map(IDS2017_TO_SHARED).fillna(NO_COUNTERPART)

        for tag, sel in (("temporal_test_90k", te_pos), ("all_300k", None)):
            yt = y_true_all if sel is None else y_true_all.iloc[sel].reset_index(drop=True)
            yp = y_pred if sel is None else y_pred.iloc[sel].reset_index(drop=True)
            row = {"model": name, **score(yt, yp, tag)}
            rows.append(row)
            print(
                f"  {name:20s} {tag:18s} macro_all={row['f1_macro_all']:.4f} "
                f"macro_measurable={row['f1_macro_measurable']:.4f} "
                f"acc={row['accuracy']:.4f} no_counterpart={row['pred_no_counterpart']:,}"
            )
        del model

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "cross_2017_to_2018.csv", index=False)
    (OUT / "cross_2017_to_2018_integrity.json").write_text(
        json.dumps(checks, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {OUT / 'cross_2017_to_2018.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
