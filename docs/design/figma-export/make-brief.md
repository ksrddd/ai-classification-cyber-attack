# CyberML dashboard — Figma Make brief

Paste this whole file into the Figma Make prompt box, attach the ten screenshots
in `screens/`, and Make has everything it needs to rebuild — or redesign — the
dashboard without inventing a single colour, label, or number.

Nothing here is aspirational. Every token, label, and figure below is read out of
the running app at `web/` and the API at `api/main.py`, captured 2026-08-23
against bundle `cicids2017_temporal_v1`.

---

## 1. What this product is

A network intrusion detection (NIDS) results dashboard. It reads **results
bundles** produced by two different training pipelines — CICIDS2017 and
CSE-CIC-IDS2018 — and lets a security analyst answer three questions in order:

1. **Look at the data** — what was trained on, how the classes are balanced.
2. **Look at the results** — how each of 7 models scored, and where they fail.
3. **Use the model** — score a new CSV of traffic.

That order is the information architecture. The sidebar is grouped by it
literally: `DATA` / `RESULTS` / `USE`.

### The one rule that governs the whole UI

A metric the bundle never recorded arrives as `null` and **must render as an
em-dash (—), never as 0**. A false-positive rate of `0.0000` is the best score a
detector can post; a false-positive rate that was never measured tells you
nothing. Rendering the second as the first makes the least-evaluated model look
like the best one. Any redesign must keep "absent" and "zero" visually distinct.

Second rule: **support sits next to every per-class score.** Some classes have 4
test flows. A per-class F1 computed on 4 flows is not a measurement, and the UI
flags those rows rather than letting them rank anything.

---

## 2. Design language

Dark-first — SOC dashboards are used in dim rooms — with a light theme that is
warm off-white paper, not a blank monitor.

| Property | Value |
|---|---|
| Type — UI | Inter, system-ui fallback; letter-spacing `-0.005em` |
| Type — numerals & IDs | JetBrains Mono, `tabular-nums` |
| Density | Compact. Body copy runs 10.5–13px, not 14–16px |
| Corner radius | 4px (`rounded`), 3px on inline code, 6px on `pre` |
| Elevation | **No shadows.** Surfaces elevate by getting lighter within the same family, plus a 1px border |
| Motion | Colour-only transitions, 180ms ease. Layout never animates. `prefers-reduced-motion` kills keyframes too |
| Focus | 2px solid `border/focus`, 2px offset, 4px radius |
| Tables | Zebra stripe on even rows at 60% surface, 1px `border/subtle` column rules, `border/strong` under the header |
| Disabled | opacity 0.42, `cursor: not-allowed` |

### Layout skeleton

```
┌──────────┬──────────────────────────────────────────────┐
│ Sidebar  │ Topbar  h=48px, bg surface, 1px bottom border │
│ 216px    ├──────────────────────────────────────────────┤
│ (52px    │                                              │
│ collapsed│ main — scrolls, padding 24px x / 20px y      │
│ 248px    │                                              │
│ drawer   │                                              │
│ on mobile├──────────────────────────────────────────────┤
│ )        │ StatusBar — API dot + run facts              │
└──────────┴──────────────────────────────────────────────┘
```

Captured at 1920px viewport. The app is responsive: the sidebar becomes a drawer
on mobile, and the rail collapses to a 52px icon strip on demand.

### Component inventory

| Component | Anatomy |
|---|---|
| `Panel` | `bg surface-raised`, 1px `border/base`, radius 4. Header: 16px/12px padding, optional 9.5px uppercase eyebrow at `.18em` tracking, 13px semibold title, 11px mono subtitle, right-slot for controls, 1px `border/subtle` divider |
| `KpiCard` | Same shell, 16px padding. 10px uppercase label at `.16em`, **24px mono semibold value**, 10.5px mono sub-line. Optional sparkline bleeds into the right 40% at 15% opacity |
| `BarRow` | Label + coloured dot, proportional bar, right-aligned count, right-aligned percent, optional `LOW` badge |
| `Pill` / `ClassChip` | Small mono uppercase tags — `TEMPORAL SPLIT`, `TUNED`, `REQUIRED`, `EXPECTED`, `OPTIONAL`, `LOW` |
| `Sev` | Severity dot/text on the 5-step severity scale |
| `Donut`, `Sparkline` | Inline SVG, chart palette |
| `RunSelector` | `<select>` in the sidebar — switches the whole dashboard's active bundle |
| `Nil` | Renders the em-dash for a null metric. This is a real component, on purpose |

---

## 3. Colour tokens

Source of truth: `web/src/app/globals.css`. Machine-readable Tokens Studio
version: `tokens.json` (two sets, `dark` and `light`, one Figma theme switching
between them).

| Token | Dark | Light |
|---|---|---|
| `color/canvas` | `#080D17` | `#EDEAE4` |
| `color/surface` | `#0D1422` | `#F8F6F2` |
| `color/surface-raised` | `#111A2A` | `#F9F7F4` |
| `color/surface-elevated` | `#172236` | `#FDFCF9` |
| `color/surface-hover` | `#162135` | `#F2F0EA` |
| `color/ink-0` | `#DDE3F0` | `#16192A` |
| `color/ink-1` | `#A8B2C8` | `#444C62` |
| `color/ink-2` | `#6A7090` | `#6A7090` |
| `color/ink-3` | `#394058` | `#9CA2B4` |
| `color/brand-blue` | `#3B82F6` | `#1D6FEB` |
| `color/brand-cyan` | `#2AB8D8` | `#1578A8` |
| `color/brand-indigo` | `#6366F1` | `#4F46E5` |
| `color/ok` | `#22C77E` | `#0D6E48` |
| `color/info` | `#2AB8D8` | `#1260A0` |
| `color/warn` | `#E0A020` | `#8A5F08` |
| `color/danger` | `#EB6565` | `#A02020` |
| `color/critical` | `#F04070` | `#AF1437` |
| `border/subtle` | `#FFFFFF0D` | `#0000000F` |
| `border/base` | `#FFFFFF14` | `#0000001A` |
| `border/strong` | `#FFFFFF26` | `#00000033` |
| `border/focus` | `#3DAEFF` | `#1056A8` |
| `text/primary` | `#DDE3F0` | `#16192A` |
| `text/secondary` | `#A8B2C8` | `#444C62` |
| `text/muted` | `#6A7090` | `#7A8096` |

Status text tokens are separate from status dot tokens, because the dot colour
is chosen for visibility and the text colour is chosen to pass WCAG AA on
`bg/base` and `bg/surface`:

| Token | Dark | Light |
|---|---|---|
| `semantic/success-text` | `#4DCC9A` | `#0D6E48` |
| `semantic/warning-text` | `#F0AE3A` | `#8A5F08` |
| `semantic/danger-text` | `#FF7B7B` | `#A02020` |
| `semantic/info-text` | `#4FC5EA` | `#1260A0` |
| `semantic/critical-text` | `#FF6090` | `#AF1437` |

Each status also has a `-bg` (10% alpha) and `-border` (~28% alpha) variant for
callout blocks.

Chart palette — 8 categorical hues, bright on dark, deeper on light:

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| dark | `#34D899` | `#E86868` | `#F09040` | `#E8C840` | `#9B6CDD` | `#DD5BAA` | `#38B8E0` | `#6080E0` |
| light | `#0E8A56` | `#C03050` | `#C87018` | `#8A7000` | `#7040C0` | `#C02888` | `#1888C0` | `#3858C0` |

Severity scale — `info → low → medium → high → critical`:
dark `#4FC5EA · #34D899 · #E8C840 · #FF7B7B · #FF6090`,
light `#1888C0 · #0E8A56 · #8A7000 · #A02020 · #AF1437`.

**Accessibility floor to keep:** `text/primary` is 11.4:1 on `bg/base` (AAA),
`text/secondary` 6.9:1 (AA), every status text token ≥ 5.2:1. `text/muted` is
3.3:1 and is therefore only ever used for large or non-essential text. A
redesign may change hues but must re-check these ratios.

---

## 4. The eight screens

Screenshots in `screens/`, numbered to match. Dark unless noted.

### 1 — Overview (`/`) → `01-overview-dark.jpg`, `01-overview-light.jpg`
Topbar title "Overview". Four KPI cards: **Dataset** (`CICIDS2017`, sub
`9 classes · 77 features`), **Best F1 macro** (`0.8873`, sub `catboost`),
**Majority baseline** (`0.8296`, sub "accuracy of always predicting the largest
class"), **Lowest false-alarm rate** (`0.022%`, sub `lightgbm`).

Then panel *"How this run was produced"* — eyebrow `CICIDS2017_TEMPORAL_V1`,
subtitle "Provenance decides how the numbers may be read, so it sits above
them." A 3-column key/value grid: Split protocol · Train rows `1,748,579` ·
Test rows `749,401` · Random seed `42` · Class weighting `class_weight` ·
Hyperparameter search `yes`.

Then *"Models in this bundle"* — "Ranked by F1 macro. Models the bundle never
scored are not ranked." A 7-row list, each row a chart-colour dot + model name +
mono F1 + dimmed `acc 0.xxxx`.

### 2 — Dataset (`/dataset`) → `02-dataset-dark.jpg`
KPIs: Classes `9`, Test flows `749,401`, Train flows `1,748,579`, **Below
reliable support `2`** (sub "fewer than 30 test flows").

Warning callout in `warning-text`: *"2 classes cannot support inference"* —
`Infiltration (11)`, `Heartbleed (4)` in mono, then the explanation that a
single prediction moves F1 by a large fraction on these.

Panel *"Class distribution — test split"* — nine `BarRow`s: BENIGN 621,680
(82.96%), DoS 58,119 (7.76%), DDoS 38,405 (5.12%), PortScan 27,209 (3.63%),
Brute Force 2,745 (0.37%), Web Attack 643 (0.09%), Bot 585 (0.08%),
Infiltration 11 (0.00%, `LOW`), Heartbleed 4 (0.00%, `LOW`).

That distribution — one class at 83%, two classes in single digits — is the
whole design problem. It should be legible at a glance in any redesign.

### 3 — Distributions (`/eda`) → `03-distributions-dark.jpg`
Panel *"Class balance across the split"* — a 5-column table (Class, Train,
Train %, Test, Test %) with the colour dot in the class cell. Then *"EDA
figures"*, a large empty-state panel reading "Run `--stage eda` to generate this
plot." **The empty state is a first-class screen here**, not an edge case — most
bundles ship without figures.

### 4 — Model detail (`/performance`) → `04-model-detail-dark.jpg`, `04-model-detail-light.jpg`
A horizontal model-tab strip (catboost · lightgbm · logistic_regression · mlp ·
random_forest · stacking · xgboost), selected tab outlined in the accent.

KPIs for the selected model: Accuracy `0.9953`, F1 macro `0.8873`, Binary FPR
`0.065%` (sub `404 false alarms`), Train time `44.6s` (sub `gpu`).

Panel *"Per-class results"* — "Support sits next to every score, because a score
computed on a handful of flows is not a measurement." Table: Class, Precision,
Recall, F1, Support. Support values under 30 render in `warning-text` (Heartbleed
`4`, Infiltration `11`). Note Bot: precision 0.2875 / recall 0.1607 / F1 0.2061 —
a real failure sitting next to 0.99s, and it must not disappear.

Panel *"Confusion matrix"* — another empty state: this bundle stores the matrix
as a PNG, not as data, so it cannot be re-rendered.

### 5 — Comparison (`/compare`) → `05-comparison-dark.jpg`
Opens with a full-width caveat in `ink-0`: **"This run never tested whether the
gaps are real."** — the order below is point estimates alone, the distance
between first and second may be noise. Underneath, mono context line:
`CICIDS2017 · cicids2017_temporal_v1 · 749,401 test flows · 9 classes ·
predicting only the largest class scores 0.8296`.

Then *"Every recorded metric"* (right-aligned count `7 models`) — 8-column table:
Model (accent link), F1 macro (sort column, brighter), Accuracy, Recall macro,
MCC, FPR, False alarms, Missed, Train.

Real rows, best-to-worst by F1 macro: catboost `.8873`, lightgbm `.8774`,
xgboost `.8773`, random_forest `.8404`, stacking `.6822`, logistic_regression
`.5574`, mlp `.5369`. Note logistic_regression posts a 16.931% FPR = 105,258
false alarms — the cost column is the point of the screen.

### 6 — Explainability (`/shap`) → `06-explainability-dark.jpg`
Same model-tab strip, then *"Global feature importance"* — "Mean absolute SHAP
value across the explained sample." Currently an empty state: "No SHAP output is
stored for catboost in this run. Produce it with `python main.py --stage explain
--model catboost`." The populated version is a horizontal bar chart of the top-N
features.

### 7 — Batch predict (`/predict`) → `07-batch-predict-dark.jpg`
Page header block: eyebrow `INFERENCE`, 22px title **"Predict New Traffic"**,
mono sub `CICIDS2017 · 77 features · 9 attack classes · per-row probabilities`.

Two-panel wizard. **Step 1 — Select model** (accent top-border on the active
step). **Step 2 — Upload CSV** — dashed drop zone, upload glyph, "Drop CSV here
or click to browse", `CICIDS-format · .csv only`, then a full-width accent
**Run** button (disabled until both steps are satisfied). Results render below
as a predictions table with per-row class probabilities.

### 8 — Bundle contract (`/contract`) → `08-bundle-contract-dark.jpg`
Panel *"Fields the dashboard reads"* — "Checked against `cicids2017_temporal_v1`
— a field counts as present when at least one model records it." Table: Field
(mono), Requirement (`REQUIRED` red / `EXPECTED` amber / `OPTIONAL` grey pill),
Used by (which screen consumes it), In this bundle (`present` green /
`absent` red).

16 rows: accuracy, f1_macro (both REQUIRED); f1_weighted, recall_macro, mcc,
binary_fpr (EXPECTED); precision_macro, false_alarms_fp, missed_attacks_fn,
throughput_flows_per_sec, model_size_mb, train_seconds, cv_f1_macro_mean,
label_shuffle_f1_macro, f1_macro_ci_low, hp_tuned (OPTIONAL).
`label_shuffle_f1_macro` and `f1_macro_ci_low` are **absent** in this bundle.

Closing panel *"Why absent is not zero"* — the prose statement of rule one.
Keep it. It is the dashboard explaining its own integrity.

---

## 5. What a redesign should preserve, and where it is free

**Non-negotiable**
- Absent (—) never looks like zero.
- Support is never more than one glance away from a per-class score.
- Provenance (split protocol, seed, weighting, tuning) sits *above* the scores.
- The comparison screen keeps its "these gaps may be noise" caveat.
- Empty states are designed states, not error states — most bundles are partial.
- Contrast floors in §3 hold in both themes.
- Numerals stay monospace and tabular; columns of figures must align.

**Open to redesign**
- Overall visual weight and density — the current UI is deliberately austere.
- KPI card treatment, sparkline usage, chart forms.
- The class-distribution visual: nine bars where one is 83% is honest but flat.
- Confusion-matrix presentation once the data path exists.
- The predict wizard — currently two static panels, could be a real flow with
  progress, validation, and a results state.
- Sidebar: grouping is right, the styling is plain.
- Mobile. The drawer works, but no screen was designed phone-first.

---

## 6. Data shapes, if Make needs to mock realistically

Bundle list (`GET /api/bundles`) — 7 bundles available:

| id | dataset | models | classes | split | tuned |
|---|---|---|---|---|---|
| `cicids2017_temporal_v1` | CICIDS2017 | 7 | 9 | temporal | yes |
| `ids2018/300k` | CSE-CIC-IDS2018 | 7 | 15 | random stratified 70/30 | no |
| `ids2018/300k_balanced_weighting` | CSE-CIC-IDS2018 | 6 | 15 | random stratified 70/30 | no |
| `ids2018/300k_temporal` | CSE-CIC-IDS2018 | 7 | 15 | temporal | no |

Per-model metrics — **every field is nullable**: `accuracy`, `balanced_accuracy`,
`f1_macro`, `f1_weighted`, `precision_macro`, `precision_weighted`,
`recall_macro`, `recall_weighted`, `mcc`, `binary_fpr`, `binary_recall`,
`false_alarms_fp`, `missed_attacks_fn`, `train_seconds`, `predict_seconds`,
`throughput_flows_per_sec`, `model_size_mb`, `cv_f1_macro_mean`,
`cv_f1_macro_std`.

The 2018 bundles carry 15 classes, several with double-digit support — so any
table design must survive 15 rows with tiny numbers, not just 9.
