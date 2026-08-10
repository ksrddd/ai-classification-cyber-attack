"""How much does the 9-class / 7-class mismatch cost the transfer result?

The 2017 models know nine classes; 2018 contains seven of them. Heartbleed and
PortScan cannot ever be correct on 2018 data, yet the models still spend
predictions on them -- up to 3,632 rows out of 90,011.

Retraining 2017 on exactly the seven shared classes would remove that, but
needs the 2017 data. This measures the size of the effect without it: take
each model's class probabilities, zero out the two columns that cannot be
right, renormalise, and take the argmax over what remains. That is what a
model forced to answer within the shared vocabulary would do with the same
evidence.

It is not equivalent to retraining -- a model trained on seven classes would
have drawn different boundaries, so its Benign/attack trade-off could differ.
What this isolates is one specific component: predictions lost to classes that
do not exist in the target. Read it as the size of that component, not as a
forecast of the retrained result.

Read-only; writes only into this scratchpad.
"""
from __future__ import annotations

import glob
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
from src.crossdataset.schema import to_canonical  # noqa: E402
from src.ids2018.temporal_split import temporal_train_test_split  # noqa: E402

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

OUT = Path(__file__).parent
BUNDLE = Path(glob.glob(str(OUT / "Result" / "*cicids2017*"))[0])
SAMPLE = PROJECT / "data" / "ids2018" / "sample_300k.parquet"
MODELS = ["lightgbm", "xgboost", "catboost", "random_forest", "mlp",
          "logistic_regression", "stacking"]
#: 2017 classes with no 2018 counterpart -- structurally impossible answers.
IMPOSSIBLE = ("Heartbleed", "PortScan")


def score(y_true: pd.Series, y_pred: pd.Series) -> dict:
    present = [c for c in SHARED_CLASSES if c in set(y_true)]
    rep = classification_report(
        y_true, y_pred, labels=present, output_dict=True, zero_division=0
    )
    per_class = {c: rep[c]["f1-score"] for c in present}
    supports = {c: int(rep[c]["support"]) for c in present}
    measurable = [c for c in present if supports[c] >= 10]
    return {
        "f1_macro_all": round(
            f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0), 4
        ),
        "f1_macro_measurable": round(
            float(np.mean([per_class[c] for c in measurable])), 4
        ),
        "accuracy": round(float((y_true.to_numpy() == y_pred.to_numpy()).mean()), 4),
        **{f"f1__{c}": round(per_class[c], 4) for c in present},
    }


def main() -> int:
    df = pd.read_parquet(SAMPLE)
    X = to_canonical(df, "ids2018", keep_dst_port=True).astype("float64")
    X = X.replace([np.inf, -np.inf], np.nan)
    y_all = map_to_shared(df["Label"], "ids2018")

    idx = pd.RangeIndex(len(df)).to_frame(name="_pos")
    _, te_idx, _, _ = temporal_train_test_split(
        idx, df["Label"].astype(str), df["Timestamp"],
        test_size=0.30, min_test_per_class=1,
    )
    te = te_idx["_pos"].to_numpy()
    y_te = y_all.iloc[te].reset_index(drop=True)
    print(f"test rows: {len(te):,}")

    encoder = joblib.load(BUNDLE / "label_encoder.joblib")
    rows: list[dict] = []
    for name in MODELS:
        path = BUNDLE / f"{name}.joblib"
        if not path.exists():
            continue
        model = joblib.load(path)
        if not hasattr(model, "predict_proba"):
            print(f"  {name}: no predict_proba, skipped")
            del model
            continue

        proba = model.predict_proba(X)
        # ``classes_`` holds the encoder's integer codes, not names; decoding is
        # what makes "which column is Heartbleed" answerable at all.
        raw = np.asarray(model.classes_)
        if np.issubdtype(raw.dtype, np.number):
            classes = encoder.inverse_transform(raw.astype(int)).astype(str)
        else:
            classes = raw.astype(str)
        keep = np.array([c not in IMPOSSIBLE for c in classes])
        if keep.sum() == len(classes):
            print(f"  {name}: no impossible classes present, skipped")
            del model
            continue

        # Unrestricted: argmax over all nine.
        free = pd.Series(classes[proba.argmax(axis=1)]).map(
            IDS2017_TO_SHARED
        ).fillna("__no_2018_counterpart__")
        # Restricted: the two impossible columns removed before the argmax.
        restricted = pd.Series(
            classes[keep][proba[:, keep].argmax(axis=1)]
        ).map(IDS2017_TO_SHARED)
        assert not restricted.isna().any()

        for tag, pred in (("free_9class", free), ("restricted_7class", restricted)):
            row = {"model": name, "variant": tag,
                   **score(y_te, pred.iloc[te].reset_index(drop=True))}
            rows.append(row)
        a, b = rows[-2], rows[-1]
        print(
            f"  {name:20s} macro {a['f1_macro_all']:.4f} -> {b['f1_macro_all']:.4f} "
            f"({b['f1_macro_all'] - a['f1_macro_all']:+.4f})   "
            f"measurable {a['f1_macro_measurable']:.4f} -> "
            f"{b['f1_macro_measurable']:.4f}"
        )
        del model, proba

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "restrict_to_seven.csv", index=False)
    print(f"\nwrote {OUT / 'restrict_to_seven.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
