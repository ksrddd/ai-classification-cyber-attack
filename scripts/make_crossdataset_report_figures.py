"""Figures (and LaTeX table rows) for the report's cross-dataset results, from protocol_v3 data.

Draws the four chapter-4 figures of the final report: random vs capture-order
transfer, per-class best F1, the 60% vs 70% split sensitivity, and the
adversarial-validation AUC ladder. Numbers come from
``docs/experiments/crossdataset/round6_split_70_30/data``; the AUC ladder uses
ranges checked against ``round5_reliability_audit/data`` (diagnostics.json,
within_dataset_control.json, tool_matched.json).

The report sets its text in TH SarabunPSK, which is not shipped in this repo,
so the font folder is passed in (the report's own ``fonts/`` folder).

usage:
    python scripts/make_crossdataset_report_figures.py <out_images_dir> --font-dir <report>/fonts
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

REPO = Path(__file__).resolve().parents[1]
D = REPO / "docs/experiments/crossdataset/round6_split_70_30/data"

_parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_parser.add_argument("out", type=Path, help="folder the PNG figures are written to")
_parser.add_argument("--font-dir", type=Path, required=True,
                     help="folder holding THSarabun*.ttf (the report's fonts/ folder)")
_args = _parser.parse_args()
out = _args.out
out.mkdir(parents=True, exist_ok=True)

# Same typeface as the report body
for _f in _args.font_dir.glob("THSarabun*.ttf"):
    font_manager.fontManager.addfont(str(_f))
plt.rcParams["font.family"] = font_manager.FontProperties(
    fname=str(_args.font_dir / "THSarabun.ttf")).get_name()
plt.rcParams["font.size"] = 15
plt.rcParams["axes.unicode_minus"] = False

MODELS = ["lightgbm", "xgboost", "catboost", "random_forest", "stacking", "mlp", "logistic_regression"]
NAME = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "catboost": "CatBoost",
        "random_forest": "Random Forest", "stacking": "Stacking", "mlp": "MLP",
        "logistic_regression": "Logistic Regression"}
DS = {"ids2017": "2017", "ids2018": "2018"}

m3 = pd.read_csv(D / "summary_macro_f1.csv")
m2 = pd.read_csv(D / "summary_macro_f1_v2.csv")
pc = pd.read_csv(D / "summary_per_class.csv")


def cell(df, mode, tr, te, model):
    r = df[(df["mode"] == mode) & (df.train == tr) & (df.test == te) & (df.model == model)]
    assert len(r) == 1, (mode, tr, te, model)
    return r.iloc[0]


def recovery(df, mode, tr, te, model):
    t = cell(df, mode, tr, te, model)
    c = cell(df, mode, te, te, model)          # ceiling = same model, in-domain on the target
    floor = t.baseline_f1_macro
    return 100 * (t.f1_macro - floor) / (c.f1_macro - floor), c.f1_macro, floor


# ---- main table rows -------------------------------------------------------
print("% main table rows: model & 17->18 F1(SD) & Rec & ceil & 18->17 F1(SD) & Rec & ceil")
for mode in ["random", "chronological"]:
    print(f"% mode = {mode}")
    best = {}
    for tr, te in [("ids2017", "ids2018"), ("ids2018", "ids2017")]:
        best[tr] = max(MODELS, key=lambda m: cell(m3, mode, tr, te, m).f1_macro)
    for m in MODELS:
        parts = [NAME[m]]
        for tr, te in [("ids2017", "ids2018"), ("ids2018", "ids2017")]:
            t = cell(m3, mode, tr, te, m)
            rec, ceil, floor = recovery(m3, mode, tr, te, m)
            f1 = f"{t.f1_macro:.4f} $\\pm$ {t.f1_macro_sd:.4f}"
            if best[tr] == m:
                f1 = f"\\textbf{{{t.f1_macro:.4f}}} $\\pm$ {t.f1_macro_sd:.4f}"
            parts += [f1, f"{rec:.1f}", f"{ceil:.4f}"]
        print("        " + " & ".join(parts) + r" \\")
    # asymmetry summary
    recs = {m: (recovery(m3, mode, "ids2017", "ids2018", m)[0], recovery(m3, mode, "ids2018", "ids2017", m)[0]) for m in MODELS}
    a = np.mean([b - f for f, b in recs.values()])
    n = sum(b > f for f, b in recs.values())
    print(f"% {mode}: mean rec 17->18 {np.mean([f for f,_ in recs.values()]):.1f}, 18->17 {np.mean([b for _,b in recs.values()]):.1f}, asym {a:.1f}, models favour 18->17 {n}/7")
    for tr, te in [("ids2017", "ids2018"), ("ids2018", "ids2017")]:
        print(f"%   best {DS[tr]}->{DS[te]}: {best[tr]} {cell(m3, mode, tr, te, best[tr]).f1_macro:.4f}")

# ---- figure 1: random vs capture-order transfer macro-F1 ---------------------
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5), sharey=True)
x = np.arange(len(MODELS))
for ax, (tr, te) in zip(axes, [("ids2017", "ids2018"), ("ids2018", "ids2017")]):
    for k, (mode, lab, col) in enumerate([("random", "Random split", "#9ab8d6"),
                                          ("chronological", "Capture-order split", "#1f4e79")]):
        v = [cell(m3, mode, tr, te, m).f1_macro for m in MODELS]
        e = [cell(m3, mode, tr, te, m).f1_macro_sd for m in MODELS]
        ax.bar(x + (k - 0.5) * 0.38, v, 0.38, yerr=e, capsize=2, label=lab, color=col)
    base = cell(m3, "chronological", tr, te, MODELS[0]).baseline_f1_macro
    ax.axhline(base, ls="--", c="grey", lw=1, label=f"Majority baseline ({base:.4f})")
    ax.set_title(f"CICIDS{DS[tr]} $\\rightarrow$ CICIDS{DS[te]}" if tr == "ids2017" else
                 f"CSE-CIC-IDS{DS[tr]} $\\rightarrow$ CICIDS{DS[te]}", fontsize=16)
    ax.set_xticks(x, [NAME[m].replace("Logistic Regression", "LR").replace("Random Forest", "RF") for m in MODELS],
                  rotation=30, ha="right", fontsize=13)
    ax.set_ylim(0, 0.55)
    ax.grid(axis="y", alpha=.3)
axes[0].set_title("CICIDS2017 $\\rightarrow$ CSE-CIC-IDS2018", fontsize=16)
axes[0].set_ylabel("Transfer macro-F1 (mean $\\pm$ SD, 5 seeds)")
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, ["Majority baseline" if s.startswith("Majority") else s for s in l],
           loc="lower center", ncol=3, fontsize=13, frameon=False)
fig.tight_layout(rect=(0, 0.09, 1, 1))
fig.savefig(out / "fig_cd_split_modes.png", dpi=200)
plt.close(fig)

# ---- figure 2: per-class heatmap, capture-order ----------------------------
CLASSES = ["Brute Force", "DoS", "DDoS", "Bot", "Web Attack", "Infiltration", "Benign"]
cols = [("ids2017", "ids2017", "Ceiling\n2017"), ("ids2017", "ids2018", "Transfer\n2017 to 2018"),
        ("ids2018", "ids2018", "Ceiling\n2018"), ("ids2018", "ids2017", "Transfer\n2018 to 2017")]
c = pc[pc["mode"] == "chronological"]
M = np.zeros((len(CLASSES), len(cols)))
for j, (tr, te, _) in enumerate(cols):
    g = c[(c.train == tr) & (c.test == te)]
    for i, cl in enumerate(CLASSES):
        M[i, j] = g[f"f1__{cl}"].max()
fig, ax = plt.subplots(figsize=(6.4, 4.6))
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=14,
                color="white" if M[i, j] > .6 else "black")
ax.set_xticks(range(len(cols)), [t for *_, t in cols], fontsize=13)
ax.set_yticks(range(len(CLASSES)), [c + (" *" if c == "Infiltration" else "") for c in CLASSES], fontsize=14)
fig.colorbar(im, ax=ax, fraction=.04, label="Best F1 over 7 models")
fig.tight_layout()
fig.savefig(out / "fig_cd_per_class.png", dpi=200)
plt.close(fig)

# ---- figure 3: split sensitivity 60% vs 70% --------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
v2 = [cell(m2, "chronological", "ids2017", "ids2018", m).f1_macro for m in MODELS]
v3 = [cell(m3, "chronological", "ids2017", "ids2018", m).f1_macro for m in MODELS]
ax = axes[0]
ax.bar(x - .19, v2, .38, label="Train 60% (protocol_v2)", color="#c9c9c9")
ax.bar(x + .19, v3, .38, label="Train 70% (protocol_v3)", color="#1f4e79")
ax.axhline(0.1337, ls="--", c="grey", lw=1)
ax.set_ylim(0, 0.45)
ax.set_title("Transfer macro-F1", fontsize=16)
ax.set_xticks(x, [NAME[m].replace("Logistic Regression", "LR").replace("Random Forest", "RF") for m in MODELS],
              rotation=30, ha="right", fontsize=13)
ax.grid(axis="y", alpha=.3)
ax = axes[1]
p = pd.read_csv(D / "per_class_v3.csv")
p = p[(p["mode"] == "chronological") & (p.cell == "2017>2018") & (p["class"] == "Brute Force")].set_index("model")
ax.bar(x - .19, [p.loc[m, "f1_v2_mean"] for m in MODELS], .38, label="Train 60%", color="#c9c9c9")
ax.bar(x + .19, [p.loc[m, "f1_v3_mean"] for m in MODELS], .38, label="Train 70%", color="#1f4e79")
ax.set_title("Brute Force F1", fontsize=16)
ax.set_xticks(x, [NAME[m].replace("Logistic Regression", "LR").replace("Random Forest", "RF") for m in MODELS],
              rotation=30, ha="right", fontsize=13)
for i, m in enumerate(MODELS):
    v = p.loc[m, "f1_v3_mean"]
    if v > 0:
        ax.text(i + .19, v + .02, f"{v:.2f}", ha="center", fontsize=12)
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=.3)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=13, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(out / "fig_cd_split_sensitivity.png", dpi=200)
plt.close(fig)

# ---- figure 4: adversarial validation ladder (ranges checked against round 5 JSON) --
rows = [("Label-shuffle (real rows,\nshuffled dataset label)", 0.4928, 0.5055),
        ("Null control (one dataset,\nrandom halves)", 0.4915, 0.5273),
        ("Same dataset, different\nday/file (all 7 models)", 0.6580, 0.9365),
        ("Same tool, different\ndataset (tool-matched)", 0.9999, 1.0000),
        ("Cross-dataset\n(LightGBM, 6 classes)", 0.9990, 1.0000)]
fig, ax = plt.subplots(figsize=(6.6, 3.3))
for i, (lab, lo, hi) in enumerate(rows):
    ax.plot([lo, hi], [i, i], lw=8, solid_capstyle="butt", color="#1f4e79" if i >= 3 else "#9ab8d6")
    ax.plot([lo, hi], [i, i], "|", ms=14, color="black")
    ax.text(hi + .006 if hi < .97 else lo - .006, i + .22, f"{lo:.4f}–{hi:.4f}", fontsize=12,
            ha="left" if hi < .97 else "right")
ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=13)
ax.set_xlim(0.45, 1.03)
ax.axvline(0.5, ls=":", c="grey")
ax.set_xlabel("ROC AUC (range over classes, models and seeds)")
ax.grid(axis="x", alpha=.3)
fig.tight_layout()
fig.savefig(out / "fig_cd_av.png", dpi=200)
plt.close(fig)
print("% figures written to", out)
