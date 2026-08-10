"""Within-2018 baselines matched to the 2017 -> 2018 transfer run.

The first cross-dataset comparison was not like-for-like. The transfer used
the 2017 feature set (77 columns, ``Destination Port`` included) and models
trained on 9 classes whose predictions were collapsed to 7 afterwards; the
within-2018 condition it was compared against used 76 columns without
``Destination Port`` and models trained directly on 7 classes. Two differences
were folded into one number.

This run rebuilds the within-2018 side to match the transfer on both axes:

    C7    77 canonical features, trained directly on the 7 shared classes
    C15   77 canonical features, trained on 15 classes, predictions collapsed
          to 7 -- the same two-step the 2017 models go through

``C15`` is the honest comparator: D minus C15 differs only in which dataset
the model was trained on. ``C7`` is kept because the advisor's protocol
specifies training on 7 classes directly, so it says what that choice is worth.

Everything else is held to the transfer run: the same 300k sample, the same
temporal split computed once from the 15-class labels, the same 90,011 test
rows, and infinities replaced with NaN before the preprocessor sees them.

Read-only with respect to the project; writes only into this scratchpad.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

PROJECT = Path(r"C:\Users\ks\Documents\GitHub\ai-classification-cyber-attack")
sys.path.insert(0, str(PROJECT))

from src.crossdataset.labels import (  # noqa: E402
    IDS2018_TO_SHARED,
    SHARED_CLASSES,
    map_to_shared,
)
from src.crossdataset.schema import CANONICAL_FEATURES, to_canonical  # noqa: E402
from src.ids2018.models import build_model, fit_model  # noqa: E402
from src.ids2018.preprocessing import Ids2018Preprocessor, encode_labels  # noqa: E402
from src.ids2018.temporal_split import temporal_train_test_split  # noqa: E402

OUT = Path(__file__).parent
SAMPLE = PROJECT / "data" / "ids2018" / "sample_300k.parquet"
TUNING = PROJECT / "results" / "ids2018" / "300k_temporal_tuned_fair" / "tuning.json"
MODELS = [
    "lightgbm",
    "xgboost",
    "catboost",
    "random_forest",
    "mlp",
    "logistic_regression",
    "stacking",
]


def tuned_params() -> dict[str, dict]:
    entries = json.loads(TUNING.read_text(encoding="utf-8"))
    return {e["model"]: e.get("best_params") or {} for e in entries if e.get("tuned")}


def score(y_true: pd.Series, y_pred: pd.Series) -> dict:
    present = [c for c in SHARED_CLASSES if c in set(y_true)]
    rep = classification_report(
        y_true, y_pred, labels=present, output_dict=True, zero_division=0
    )
    per_class = {c: round(rep[c]["f1-score"], 4) for c in present}
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
        **{f"f1__{c}": per_class[c] for c in present},
    }


def main() -> int:
    df = pd.read_parquet(SAMPLE)
    X = to_canonical(df, "ids2018", keep_dst_port=True).astype("float64")
    assert list(X.columns) == list(CANONICAL_FEATURES)
    X = X.replace([np.inf, -np.inf], np.nan)

    y15 = df["Label"].astype(str)
    y7 = map_to_shared(y15, "ids2018")
    assert not y7.isna().any()

    idx = pd.RangeIndex(len(df)).to_frame(name="_pos")
    tr_idx, te_idx, _, _ = temporal_train_test_split(
        idx, y15, df["Timestamp"], test_size=0.30, min_test_per_class=1
    )
    tr, te = tr_idx["_pos"].to_numpy(), te_idx["_pos"].to_numpy()
    print(f"train={len(tr):,}  test={len(te):,}  features={X.shape[1]}")

    X_tr, X_te = X.iloc[tr], X.iloc[te]
    pre = Ids2018Preprocessor()
    Xt_tr, Xt_te = pre.fit_transform(X_tr), pre.transform(X_te)
    print(f"after constant-drop: {Xt_tr.shape[1]} features")

    y7_te = y7.iloc[te].reset_index(drop=True)
    params = tuned_params()
    rows: list[dict] = []

    for cond, labels in (("C7_train_on_7", y7), ("C15_train_on_15", y15)):
        y_tr = labels.iloc[tr].reset_index(drop=True)
        y_te = labels.iloc[te].reset_index(drop=True)
        yt_tr, _, enc = encode_labels(y_tr, y_te)
        print(f"\n=== {cond}: {len(enc.classes_)} training classes")

        for name in MODELS:
            model = build_model(name, "cpu")
            if params.get(name):
                try:
                    model.set_params(**params[name])
                except ValueError as exc:  # pragma: no cover - defensive
                    print(f"  {name}: tuned params rejected ({exc})")
            t0 = time.perf_counter()
            model = fit_model(name, model, Xt_tr, yt_tr)
            pred_names = pd.Series(enc.inverse_transform(model.predict(Xt_te)))
            secs = time.perf_counter() - t0

            # C15 goes through the same collapse the 2017 models go through.
            y_pred = (
                pred_names
                if cond == "C7_train_on_7"
                else pred_names.map(IDS2018_TO_SHARED)
            )
            row = {"condition": cond, "model": name, "seconds": round(secs, 1),
                   **score(y7_te, y_pred)}
            rows.append(row)
            print(
                f"  {name:20s} macro_all={row['f1_macro_all']:.4f} "
                f"macro_measurable={row['f1_macro_measurable']:.4f} "
                f"acc={row['accuracy']:.4f} ({secs:.0f}s)"
            )
            del model

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "within_2018_matched.csv", index=False)
    print(f"\nwrote {OUT / 'within_2018_matched.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
