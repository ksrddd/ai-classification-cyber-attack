"""Schema-ablation for the 2017/2018 cross-dataset work.

Answers: what does moving the 2018 run onto the shared cross-dataset schema
cost, and how much of any change comes from the feature swap vs the class
collapse?

Four conditions, all on the same 300k sample and the *same* temporal split
(the split is computed once from the 15-class labels so every condition sees
byte-identical train/test rows):

    A   baseline           77 feats (has Protocol, no Dst Port), 15 classes
    B   shared schema      76 feats (no Protocol, no Dst Port),  15 classes
    Bp  shared + Dst Port   77 feats (no Protocol, has Dst Port), 15 classes
    C   shared schema      76 feats,                              7 classes

A reproduces results/ids2018/300k_temporal_tuned_fair and acts as the control.
Hyper-parameters are reused from that bundle's tuning.json -- no re-tuning, so
any movement is attributable to the schema change alone.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

PROJECT = Path(r"C:\Users\ks\Documents\GitHub\ai-classification-cyber-attack")
sys.path.insert(0, str(PROJECT))

from src.ids2018.config import (  # noqa: E402
    DST_PORT_COL,
    IDENTITY_COLS,
    LABEL_COL,
    TIMESTAMP_COL,
)
from src.ids2018.models import build_model, fit_model  # noqa: E402
from src.ids2018.preprocessing import Ids2018Preprocessor, encode_labels  # noqa: E402
from src.ids2018.temporal_split import temporal_train_test_split  # noqa: E402

OUT = Path(__file__).parent
SAMPLE = PROJECT / "data" / "ids2018" / "sample_300k.parquet"
BASELINE = PROJECT / "results" / "ids2018" / "300k_temporal_tuned_fair"
MODELS = ["lightgbm", "xgboost", "random_forest", "logistic_regression"]

# 15 -> 7 collapse. Names on the left are the raw 2018 labels; names on the
# right are the shared vocabulary that also exists in CICIDS2017.
SEVEN_CLASS = {
    "Benign": "Benign",
    "Bot": "Bot",
    "Infilteration": "Infiltration",
    "DDOS attack-HOIC": "DDoS",
    "DDOS attack-LOIC-UDP": "DDoS",
    "DDoS attacks-LOIC-HTTP": "DDoS",
    "DoS attacks-GoldenEye": "DoS",
    "DoS attacks-Hulk": "DoS",
    "DoS attacks-SlowHTTPTest": "DoS",
    "DoS attacks-Slowloris": "DoS",
    "FTP-BruteForce": "Brute Force",
    "SSH-Bruteforce": "Brute Force",
    "Brute Force -Web": "Web Attack",
    "Brute Force -XSS": "Web Attack",
    "SQL Injection": "Web Attack",
}


def tuned_params() -> dict[str, dict]:
    entries = json.loads((BASELINE / "tuning.json").read_text(encoding="utf-8"))
    return {e["model"]: e.get("best_params") or {} for e in entries if e.get("tuned")}


def main() -> int:
    df = pd.read_parquet(SAMPLE)
    print(f"loaded {df.shape[0]:,} rows x {df.shape[1]} cols")

    y_raw = df[LABEL_COL].astype(str)
    timestamps = df[TIMESTAMP_COL]

    # Everything that is never a feature under any condition.
    never = [c for c in [*IDENTITY_COLS, TIMESTAMP_COL, LABEL_COL] if c in df.columns]
    feature_frame = df.drop(columns=never)

    conditions = {
        "A_baseline_77": {"drop": [DST_PORT_COL], "classes": 15},
        "B_shared_76": {"drop": [DST_PORT_COL, "Protocol"], "classes": 15},
        "Bp_shared_77_dstport": {"drop": ["Protocol"], "classes": 15},
        "C_shared_76_7class": {"drop": [DST_PORT_COL, "Protocol"], "classes": 7},
    }

    # One split for all conditions -- computed on the full frame and the raw
    # 15-class labels, so the row partition is identical everywhere.
    print("computing temporal split (shared by every condition) ...")
    idx = pd.RangeIndex(len(df)).to_frame(name="_pos")
    tr_idx, te_idx, y_tr_raw, y_te_raw = temporal_train_test_split(
        idx, y_raw, timestamps, test_size=0.30, min_test_per_class=1
    )
    tr_pos = tr_idx["_pos"].to_numpy()
    te_pos = te_idx["_pos"].to_numpy()
    print(f"  train={len(tr_pos):,}  test={len(te_pos):,}")

    params = tuned_params()
    rows: list[dict] = []
    per_class: dict[str, pd.DataFrame] = {}

    for cond, spec in conditions.items():
        drop = [c for c in spec["drop"] if c in feature_frame.columns]
        X = feature_frame.drop(columns=drop)
        if spec["classes"] == 7:
            y_tr = y_tr_raw.map(SEVEN_CLASS)
            y_te = y_te_raw.map(SEVEN_CLASS)
            assert not y_tr.isna().any() and not y_te.isna().any(), "unmapped label"
        else:
            y_tr, y_te = y_tr_raw, y_te_raw

        X_tr, X_te = X.iloc[tr_pos], X.iloc[te_pos]
        pre = Ids2018Preprocessor()
        Xt_tr = pre.fit_transform(X_tr)
        Xt_te = pre.transform(X_te)
        yt_tr, yt_te, enc = encode_labels(y_tr, y_te)
        print(
            f"\n=== {cond}: {X.shape[1]} cols in, {Xt_tr.shape[1]} after "
            f"constant-drop, {len(enc.classes_)} classes"
        )

        for name in MODELS:
            model = build_model(name, "cpu")
            if params.get(name):
                try:
                    model.set_params(**params[name])
                except ValueError as exc:  # pragma: no cover - defensive
                    print(f"  {name}: could not apply tuned params ({exc})")
            t0 = time.perf_counter()
            model = fit_model(name, model, Xt_tr, yt_tr)
            pred = model.predict(Xt_te)
            secs = time.perf_counter() - t0

            macro = f1_score(yt_te, pred, average="macro", zero_division=0)
            weighted = f1_score(yt_te, pred, average="weighted", zero_division=0)
            rows.append(
                {
                    "condition": cond,
                    "model": name,
                    "n_features": int(Xt_tr.shape[1]),
                    "n_classes": len(enc.classes_),
                    "f1_macro": round(float(macro), 4),
                    "f1_weighted": round(float(weighted), 4),
                    "seconds": round(secs, 1),
                }
            )
            print(f"  {name:22s} macro={macro:.4f} weighted={weighted:.4f} ({secs:.0f}s)")

            rep = classification_report(
                yt_te, pred, target_names=list(enc.classes_),
                output_dict=True, zero_division=0,
            )
            frame = pd.DataFrame(rep).T
            frame = frame.loc[[c for c in enc.classes_]][["f1-score", "support"]]
            frame.columns = [f"{cond}|{name}", "support"]
            key = f"{cond}|{name}"
            per_class[key] = frame[[f"{cond}|{name}"]]
            per_class[f"__support__{cond}"] = frame[["support"]]

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "ablation_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))

    # Per-class F1 for the 15-class conditions, side by side.
    fifteen = [k for k in per_class if not k.startswith("__support__")
               and not k.startswith("C_")]
    pc15 = pd.concat([per_class[k] for k in fifteen], axis=1)
    pc15.insert(0, "support", per_class["__support__A_baseline_77"]["support"])
    pc15.to_csv(OUT / "ablation_per_class_15.csv")

    seven = [k for k in per_class if k.startswith("C_")]
    if seven:
        pc7 = pd.concat([per_class[k] for k in seven], axis=1)
        pc7.insert(0, "support", per_class["__support__C_shared_76_7class"]["support"])
        pc7.to_csv(OUT / "ablation_per_class_7.csv")

    print(f"\nwrote {OUT / 'ablation_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
