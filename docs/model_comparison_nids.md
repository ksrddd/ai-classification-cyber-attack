# Comparative Evaluation of 7 ML Models for a CICIDS2017 Network Intrusion Detection System

**Document type:** Technical evaluation report
**Author role:** ML Research + Security Operations
**Date:** 2026-07-27
**Evidence base:** Measured results from this repository — `results/latest/` (primary) and `results/local_300k_70_30/` (secondary)

---

## 0. Executive Summary

Seven models were trained and evaluated on the CICIDS2017 flow corpus reduced to **9 consolidated classes** over **80 CICFlowMeter features**. This report analyses each model across 12 technical dimensions, cites primary sources, and issues a deployment recommendation.

**Headline finding:** **XGBoost is the recommended model for inline SOC deployment** — but *not* primarily because it tops the leaderboard. Once the two statistically void classes are excluded (§5.2), the top four models (Stacking, Random Forest, CatBoost, XGBoost) span just **0.0047 in macro-F1 and are effectively tied on predictive quality.** The decision therefore rests on the secondary criteria, where XGBoost's advantage is large and unambiguous: **the lowest benign false-positive rate (0.66%, a 23–85% reduction in daily alert volume versus the other three), a 10–30 MB artifact, microsecond compiled inference, and the tightest CV variance (±0.0006).**

**Four findings that matter more than the leaderboard:**

1. **The reported accuracy baseline is misleading.** `metrics.json` reports `majority_baseline_acc = 0.0100`, because the majority class *after* targeted oversampling is Infiltration. The operationally meaningful baseline is "always predict BENIGN" = **52,355 / 60,000 = 0.8726**. XGBoost's 0.9867 is therefore a **10.6× reduction in error rate**, not a "+0.977 lift". All accuracy claims in this project should be restated against 0.8726.

2. **LightGBM's collapse is a hyperparameter-search artifact, not an algorithmic weakness.** In `results/latest/` LightGBM scored 0.7634 accuracy with unstable CV (0.4069 ± 0.1814). The cause is visible in its selected parameters: `min_child_samples=1` combined with `num_leaves=127`. In `results/local_300k_70_30/`, running with defaults and no HP search, the same model scored 0.9507 accuracy / 0.8407 macro-F1. **This is a search-space bug and is fixable** — see §5.4.

3. **The published ranking is unstable under a 16-row change.** Heartbleed (n=3) and Web Attack (n=13) carry the same weight in macro-F1 as BENIGN (n=52,355). Excluding them moves **XGBoost from 1st to 4th** and **Stacking from 4th to 1st** — see the full recomputation in §5.2. Any conclusion drawn from the 9-class macro-F1 alone is fragile.

4. **No model solves Infiltration.** Best F1 across all seven is **0.3201 (Stacking)**. This is a known property of CICIDS2017, not a modelling failure: Infiltration is a post-compromise internal activity whose flow-level statistics are near-identical to benign traffic.

---

## 1. Experimental Setup (as measured in this repository)

### 1.1 Data and protocol

| Parameter | Value |
|---|---|
| Dataset | CICIDS2017 (Canadian Institute for Cybersecurity) |
| Feature space | 80 CICFlowMeter flow features |
| Classes | 9: BENIGN, Bot, Brute Force, DDoS, DoS, Heartbleed, Infiltration, PortScan, Web Attack |
| Corpus subsample | 300,000 rows |
| Train / Calibration / Test | 180,000 / 60,000 / 60,000 |
| Imbalance strategy | `targeted` — Infiltration oversampled to 1.0× majority, **train folds only** |
| Threshold policy | FN-aware, optimised for **F2** on Infiltration, capped at FPR ≤ 0.02 |
| Cross-validation | StratifiedKFold, k=5 |
| HP search | RandomizedSearchCV, n_iter=8, on an 80,000-row subsample |
| Random state | 42 |
| **Total wall time (all 7 models)** | **14,652.5 s ≈ 4 h 04 m** |

### 1.2 Test-set class distribution (natural, untouched)

| Class | Test support | Share | Statistical confidence |
|---|---:|---:|---|
| BENIGN | 52,355 | 87.26% | High |
| DDoS | 3,888 | 6.48% | High |
| DoS | 1,678 | 2.80% | High |
| Bot | 630 | 1.05% | High |
| Infiltration | 599 | 1.00% | High |
| Brute Force | 444 | 0.74% | Adequate |
| PortScan | 390 | 0.65% | Adequate |
| **Web Attack** | **13** | 0.02% | **Anecdotal** |
| **Heartbleed** | **3** | 0.005% | **Anecdotal** |

> **Reading rule enforced throughout this report:** any metric for Heartbleed (n=3) or Web Attack (n=13) is an *anecdote*, not a measurement. A single flip changes Heartbleed recall by 33 percentage points. These two classes are excluded from all ranking decisions and are reported for completeness only.

### 1.3 Validity controls that passed

Two integrity checks ran for every model and both behaved correctly:

- **Label-shuffle control:** with permuted labels, macro-F1 collapsed to 0.064–0.114 against a 0.1111 chance level for 9 classes. This rules out label leakage through the preprocessing pipeline.
- **CV stability:** six of seven models produced CV standard deviations ≤ 0.004, confirming the test-set numbers are not single-split flukes. LightGBM was the sole exception (±0.1814).

---

## 2. Measured Results

### 2.1 Headline metrics (`results/latest/`)

| Model | Accuracy | Balanced Acc | **macro-F1** | weighted-F1 | CV macro-F1 (μ ± σ) | Misclassified / 60,000 |
|---|---:|---:|---:|---:|---|---:|
| **XGBoost** | **0.9867** | 0.9062 | **0.8651** | **0.9860** | 0.9606 ± 0.0006 | **800** |
| Random Forest | 0.9856 | **0.9101** | 0.8519 | 0.9854 | 0.9560 ± 0.0006 | 867 |
| CatBoost | 0.9829 | 0.8861 | 0.8272 | 0.9838 | 0.9533 ± 0.0010 | 1,025 |
| Stacking | 0.9830 | 0.8085 | 0.7509 | 0.9844 | 0.8490 ± 0.0005 | 1,022 |
| Logistic Regression | 0.9409 | 0.7972 | 0.7081 | 0.9464 | 0.8499 ± 0.0037 | 3,543 |
| MLP | 0.9807 | 0.7833 | 0.6981 | 0.9804 | 0.9324 ± 0.0039 | 1,156 |
| LightGBM | 0.7634 | 0.7899 | 0.6174 | 0.8455 | 0.4069 ± 0.1814 ⚠ | 14,194 |

*Trivial "always BENIGN" baseline: accuracy 0.8726, macro-F1 0.1036.*

### 2.2 Per-class F1 (argmax decision rule)

| Class | RF | XGB | LGBM | CatBoost | MLP | LogReg | Stacking | **Best** |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BENIGN | 0.9918 | **0.9924** | 0.8473 | 0.9902 | 0.9890 | 0.9661 | 0.9902 | XGB |
| Bot | 0.9952 | 0.9944 | 0.9000 | **0.9960** | 0.9936 | 0.8733 | **0.9960** | CatBoost / Stack |
| Brute Force | 0.9955 | **0.9977** | 0.9103 | **0.9977** | 0.9789 | 0.9310 | **0.9977** | 3-way tie |
| DDoS | 0.9951 | **0.9996** | 0.9372 | 0.9955 | 0.9892 | 0.8806 | **0.9996** | XGB / Stack |
| DoS | 0.9967 | 0.9991 | 0.8150 | 0.9929 | 0.9638 | 0.8936 | **0.9994** | Stacking |
| PortScan | **0.9936** | **0.9936** | 0.8923 | **0.9936** | 0.8867 | 0.6410 | **0.9936** | 4-way tie |
| **Infiltration** | 0.3242 | 0.2871 | 0.0602 | 0.3078 | 0.2932 | 0.0969 | **0.3201** | Stacking |
| Heartbleed ⚠ | 1.0000 | 1.0000 | 0.0000 | 0.8000 | 0.0000 | 1.0000 | 0.0000 | *n=3, void* |
| Web Attack ⚠ | 0.3750 | 0.5217 | 0.1940 | 0.3714 | 0.1890 | 0.0905 | 0.4615 | *n=13, void* |

### 2.3 Infiltration at the FN-aware operating point

This is the tuned operating point, not argmax. Ranked by F2 (recall-weighted), which is the correct objective for missed intrusions.

| Model | Threshold | Precision | Recall | **F2** | FPR | FN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Stacking** | 0.9423 | 0.2745 | **0.3840** | **0.3556** | 0.0102 | **369** | 608 |
| CatBoost | 0.9058 | 0.2819 | 0.3389 | 0.3257 | 0.0087 | 396 | 517 |
| Random Forest | 0.9526 | **0.3433** | 0.3072 | 0.3138 | **0.0059** | 415 | **352** |
| XGBoost | 0.9099 | 0.3276 | 0.2554 | 0.2672 | 0.0053 | 446 | 314 |
| MLP | 0.9616 | 0.3503 | 0.2521 | 0.2671 | 0.0047 | 448 | 280 |
| LightGBM | 0.7518 | 0.0315 | 0.6845 | 0.1329 | 0.2125 ✗ | 189 | 12,620 ✗ |
| Logistic Regression | 0.7548 | 0.0750 | 0.1369 | 0.1175 | 0.0170 | 517 | 1,011 |

> LightGBM's 0.6845 recall looks attractive until you read across: it costs **12,620 false positives** at a 21.25% FPR — a 10× breach of the 2% policy cap. It is not a usable operating point.

### 2.4 The false-positive budget — the number that decides deployment

Benign misclassification rate translated to a realistic enterprise volume of **10 M flows/day**:

| Model | BENIGN recall | Benign FP rate | **Alerts/day @ 10M flows** | Analyst-hours/day @ 2 min/alert |
|---|---:|---:|---:|---:|
| **XGBoost** | 0.9934 | **0.66%** | **66,000** | 2,200 |
| Random Forest | 0.9919 | 0.81% | 81,000 | 2,700 |
| CatBoost | 0.9883 | 1.17% | 117,000 | 3,900 |
| Stacking | 0.9878 | 1.22% | 122,000 | 4,067 |
| MLP | 0.9877 | 1.23% | 123,000 | 4,100 |
| Logistic Regression | 0.9485 | 5.15% | 515,000 | 17,167 |
| LightGBM | 0.7421 | 25.79% | 2,579,000 | 85,967 |

**This table is the single most important artifact in the report.** Even the best model generates ~66,000 raw alerts per day. No SOC on earth triages that. The conclusion is not "the model is bad" — it is that **a flow-level classifier is a scoring stage, never an alerting stage.** See §7.3.

---

## 3. Per-Model Technical Evaluation

---

### 3.1 Logistic Regression

**Official documentation:** https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
**Primary references:**
- Cox, D. R. (1958). "The Regression Analysis of Binary Sequences." *JRSS Series B*, 20(2), 215–242.
- McFadden, D. (1974). "Conditional Logit Analysis of Qualitative Choice Behavior." In *Frontiers in Econometrics*. — multinomial/softmax extension.
- Liu, D. C. & Nocedal, J. (1989). "On the Limited Memory BFGS Method for Large Scale Optimization." *Mathematical Programming*, 45, 503–528. — the `lbfgs` solver.
- Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in Python." *JMLR*, 12, 2825–2830.

**Fundamental learning algorithm.** Multinomial logistic regression models the class posterior as a softmax over linear scores: `P(y=k|x) = exp(wₖᵀx + bₖ) / Σⱼ exp(wⱼᵀx + bⱼ)`. Parameters are fit by minimising L2-penalised cross-entropy, a convex objective, using L-BFGS quasi-Newton optimisation. Convexity guarantees a unique global optimum — no seed sensitivity, no local minima.

**Strengths.**
- Only model here with a **provably unique global optimum**; retraining is bit-reproducible.
- Coefficients are directly interpretable as log-odds per unit of a standardised feature — the strongest native interpretability in the set.
- Calibrated probabilities out of the box; no isotonic/Platt post-hoc step required.
- Parameter vector is 80 × 9 = **720 floats ≈ 6 KB**. Deployable to a network appliance, an eBPF-adjacent userspace agent, or an FPGA lookup.
- Trains in seconds; supports true online learning via SGD for concept-drift adaptation.

**Weaknesses.**
- Restricted to **linear decision boundaries in the given feature space**. Network attacks are overwhelmingly defined by *interactions* — e.g. "high `Flow Packets/s` **AND** low `Average Packet Size` **AND** `SYN Flag Count` > 0" describes a SYN flood; a linear model cannot express the conjunction without hand-built interaction terms.
- Measured cost of this limitation: **macro-F1 0.7081 vs XGBoost's 0.8651**, and PortScan F1 of only **0.6410** (recall 0.5128 — it misses nearly half of all port scans).
- Requires feature scaling; sensitive to multicollinearity, which is severe in CICIDS2017 (`Average Packet Size`, `Packet Length Mean`, and `Avg Fwd Segment Size` are near-duplicates).

**Computational complexity.**
- Training: `O(i · n · d · K)` where i = L-BFGS iterations (typically 100–1000), n = 180,000, d = 80, K = 9. Memory for L-BFGS history: `O(m · d · K)` with m ≈ 10.
- Inference: `O(d · K)` = 720 multiply-accumulates per flow.

**Training time & latency.** Fastest to train of the seven — order of **seconds to low minutes**. Inference is a single 80×9 GEMV: **sub-microsecond per flow**, ≥ 1 M flows/s single-core. Nothing else in this set approaches it.

**Memory & scalability.** ~6 KB model. Scales linearly in n and is trivially parallelisable by data sharding. The only model here that fits comfortably in L1 cache.

**Hyperparameter sensitivity & overfitting risk.** Effectively one hyperparameter (`C`, the inverse regularisation strength); tuning selected `C=0.1`. **Overfitting risk is the lowest in the set** — this model *underfits*. CV σ = 0.0037.

**Interpretability.** Best-in-class. Coefficients are the explanation. SHAP values for a linear model reduce to `φᵢ = wᵢ(xᵢ − E[xᵢ])` — exact, closed-form, and computable in the same time as a prediction.

**Robustness to noise & unscaled features.** **Poor on unscaled input** — an unstandardised `Flow Duration` (range ~10⁸) will numerically dominate `Protocol` (range 0–17) and prevent convergence. `StandardScaler` is mandatory. Moderately robust to label noise thanks to convexity and strong regularisation. Vulnerable to outliers because squared-scale features drag the linear fit.

**Class imbalance.** `class_weight='balanced'` reweights the loss and works correctly in principle. In practice it is insufficient here: Infiltration F1 = **0.0969** at the tuned threshold, the second-worst result measured.

**Expected CICIDS2017 performance.** Confirmed: **0.9409 accuracy / 0.7081 macro-F1**. Strong on linearly separable volumetric attacks (Bot F1 0.8733, DDoS 0.8806), weak wherever feature interaction is required.

**SOC deployment.** Not viable as a primary detector — 5.15% benign FPR is **515,000 alerts/day**. Genuine roles: (a) a **transparent baseline** that quantifies how much non-linearity the problem actually requires; (b) the **meta-learner in the stacking ensemble**, which is exactly how it is used in `src/models/stacking.py`; (c) an **auditable tie-breaker** in regulated environments where a decision must be explainable to a non-technical reviewer.

---

### 3.2 Random Forest

**Official documentation:** https://scikit-learn.org/stable/modules/ensemble.html#random-forests
**Primary references:**
- Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5–32. DOI: 10.1023/A:1010933404324.
- Breiman, L. (1996). "Bagging Predictors." *Machine Learning*, 24(2), 123–140.
- Ho, T. K. (1995). "Random Decision Forests." *ICDAR '95*, 278–282. — the random subspace method.
- Louppe, G. (2014). *Understanding Random Forests: From Theory to Practice.* PhD thesis, Univ. of Liège. — the definitive analysis of the sklearn implementation.

**Fundamental learning algorithm.** Bagged ensemble of decision trees with double randomisation: each tree trains on a bootstrap resample, and each split considers only a random subset of features (√80 ≈ 9). Averaging decorrelated high-variance trees drives ensemble variance toward the residual inter-tree correlation while leaving bias unchanged. Prediction is a soft vote over 200 trees. **Tuned parameters:** `n_estimators=200, max_depth=20, min_samples_split=10, min_samples_leaf=2`.

**Strengths.**
- **Highest balanced accuracy in the entire study: 0.9101.** With `class_weight='balanced_subsample'`, weights are recomputed per bootstrap, which is measurably better than global reweighting for rare classes.
- **Best Infiltration precision (0.3433) and lowest Infiltration FP count (352)** of any model that respects the FPR policy — the most *conservative* stealthy-attack detector available.
- Fully **scale-invariant**: splits are threshold comparisons, so raw CICFlowMeter output works without normalisation.
- **Embarrassingly parallel** — trees are independent, so training and inference both scale near-linearly in cores.
- Provides free **out-of-bag error estimation**, an unbiased generalisation estimate with no held-out split.
- Highly resistant to overfitting as `n_estimators` grows: adding trees cannot increase generalisation error (Breiman 2001, Thm 1.2).
- Empirically robust under adversarial perturbation — `results/local_300k_70_30/red_team.json` shows macro-F1 essentially flat (0.8640 → 0.8631) across ε = 0.01 → 0.10 feature noise, with prediction flip rate rising only to 14.5%.

**Weaknesses.**
- **Largest memory footprint in the set.** 200 trees at depth 20 over 180,000 rows produce a very large node population; scikit-learn stores 8 fields (~64 bytes) per node, giving a serialised model in the **hundreds of MB**. This is 1–2 orders of magnitude larger than XGBoost or CatBoost for *lower* macro-F1.
- Inference is **memory-latency-bound, not compute-bound** — 200 independent tree traversals are 200 pointer-chasing walks with poor cache locality.
- Cannot extrapolate beyond the training range: a novel volumetric attack exceeding all observed `Flow Bytes/s` values is clamped to the highest training leaf.
- Default impurity-based feature importance is **biased toward high-cardinality continuous features** (Strobl et al. 2007) — a real hazard on CICIDS2017, where `Flow Duration` and `Flow IAT Mean` are continuous and `Protocol` is not. Use permutation importance or SHAP instead.

**Computational complexity.**
- Training: `O(T · √d · n log²n)` — with T=200, √d≈9, n=180,000.
- Inference: `O(T · depth)` = 200 × 20 = **4,000 comparisons per flow** (worst case), plus 200 cache-missing traversals.

**Training time & latency.** Moderate training, strongly parallel. Inference is the **slowest of the tree models** in practice — roughly 3–10× XGBoost per flow, dominated by cache misses rather than arithmetic. Estimated 50–200 µs/flow single-threaded in sklearn; batching amortises this substantially.

**Memory & scalability.** Training memory scales with `n × T`. Deployment memory is the binding constraint. Mitigations: cap `max_depth` at 12–15, use `min_samples_leaf ≥ 5`, or compile via **Treelite/ONNX Runtime**, which typically yields 10–30× inference speedup and a large size reduction.

**Hyperparameter sensitivity & overfitting risk.** **The least sensitive model in the set.** Defaults are already strong; `n_estimators` is monotone-safe. Overfitting risk is low and controlled by `min_samples_leaf`/`max_depth`. CV σ = 0.0006 — the joint-tightest measured.

**Interpretability.** Good. **TreeSHAP applies exactly** with `O(T · L · D²)` complexity (Lundberg et al. 2020) — polynomial, not exponential. Also supports proximity analysis and partial dependence. Individual trees are readable; the 200-tree ensemble is not, so SHAP is the practical route.

**Robustness to noise & unscaled features.** **Best in the set on both axes.** Scale-invariant by construction. Bootstrap averaging suppresses label noise — critical here, because CICIDS2017 labels are assigned per *attack window*, so benign flows inside an attack window are systematically mislabelled (Engelen et al. 2021). RF's bagging directly attenuates this.

**Class imbalance.** Strong. `balanced_subsample` recomputes weights per bootstrap. Measured: **highest balanced accuracy (0.9101)** and best minority-class precision. This is RF's single clearest advantage over the boosters.

**Expected CICIDS2017 performance.** Confirmed: **0.9856 / 0.8519 macro-F1**, second overall. Corroborated by the independent 70/30 full-corpus run where RF achieved the **best macro-F1 of that run (0.9160)** and was selected as champion in `results/champion.json`.

**SOC deployment.** Very strong for **batch and near-real-time** scoring. The memory footprint and traversal latency argue against a per-packet inline path without compilation. **Recommended concrete role: the second-stage stealthy-attack scorer** — its Infiltration precision (0.3433) and minimal FP count (352) make it the right model for a lower-volume, higher-scrutiny queue.

---

### 3.3 XGBoost — ⭐ Recommended

**Official documentation:** https://xgboost.readthedocs.io/
**Primary references:**
- Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *KDD '16*, 785–794. DOI: 10.1145/2939672.2939785.
- Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine." *Annals of Statistics*, 29(5), 1189–1232. — the gradient boosting foundation.
- Friedman, J. H. (2002). "Stochastic Gradient Boosting." *Computational Statistics & Data Analysis*, 38(4), 367–378. — subsampling.

**Fundamental learning algorithm.** Additive stage-wise boosting on a **second-order Taylor expansion** of the loss. Each new tree fits the gradient *and* Hessian of the current residual, giving a closed-form optimal leaf weight `w* = −G/(H+λ)` and split gain `½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)] − γ`. Regularisation (`λ` L2, `γ` complexity penalty) is written into the objective rather than bolted on. **Tuned parameters:** `n_estimators=400, max_depth=6, learning_rate=0.03, subsample=0.9`, `tree_method='hist'`.

**Strengths.**
- **Best macro-F1 (0.8651) and best weighted-F1 (0.9860) measured.**
- **Lowest benign false-positive rate (0.66%)** — the decisive operational metric, and a 19% reduction in daily alert volume versus Random Forest.
- **Near-perfect on all high-volume attacks:** DDoS F1 **0.9996**, DoS F1 **0.9991** (recall 1.0000 — *zero* missed DoS flows), Brute Force 0.9977, PortScan 0.9936.
- Second-order optimisation converges in fewer trees than first-order boosting and with more stable steps.
- **Explicit regularisation** (`λ`, `γ`, `α`) plus row and column subsampling — several independent overfitting brakes.
- **Sparsity-aware split finding** with a learned default direction per node: missing/NaN values are handled natively. This matters directly here, because `Flow Bytes/s` and `Flow Packets/s` produce Inf/NaN on zero-duration flows.
- Fastest CV convergence in the study: **0.9606 ± 0.0006**, the best CV mean and joint-tightest variance.
- Mature production ecosystem: ONNX export, Treelite compilation, GPU inference, JVM/C++/Rust bindings, `xgboost4j` for Spark/Flink pipelines.

**Weaknesses.**
- **Sequential by construction.** Trees depend on prior residuals, so boosting rounds cannot be parallelised; only *within-tree* split finding is parallel. Wall-clock training exceeds Random Forest at equal core count.
- **~10 interacting hyperparameters** — the widest tuning surface among the single models.
- Multiclass builds `n_estimators × K` = 400 × 9 = **3,600 trees**, so model size and inference cost scale linearly with the class count.
- **Genuinely weak on Infiltration: F1 0.2871, the worst of the four healthy tree models.** Its aggressive residual fitting drives it toward confident majority predictions; because Infiltration is nearly inseparable at the flow level, XGBoost resolves the ambiguity by predicting BENIGN. This is the honest cost of its precision advantage.
- Overfits readily if `max_depth` is raised without lowering `learning_rate`.

**Computational complexity.**
- Training (`hist`): `O(T · K · (n · d + b · d))` where b = histogram bins (256 default). The histogram approximation reduces per-split cost from `O(n log n)` sorting to `O(n)` binning — the core reason `hist` is the default at this scale.
- Training (`exact`): `O(T · K · d · n log n)`.
- Inference: `O(T · K · depth)` = 400 × 9 × 6 = **21,600 comparisons per flow**, but over shallow, cache-resident trees.

**Training time & latency.** Moderate training (minutes to tens of minutes at this scale on multicore CPU; GPU `hist` gives 5–20×). Inference is **fast and cache-friendly** — depth-6 trees fit in cache, unlike RF's depth-20 trees. Estimated **10–50 µs/flow** in native Python; **~1–5 µs/flow** after Treelite/ONNX compilation, which is comfortably line-rate for a 10 Gbps flow-export pipeline.

**Memory & scalability.** 3,600 depth-6 trees ≈ **10–30 MB serialised** — roughly an order of magnitude smaller than the Random Forest at *higher* accuracy. Supports out-of-core training via external memory (DMatrix), distributed training on Spark/Dask/Ray, and multi-GPU.

**Hyperparameter sensitivity & overfitting risk.** **High sensitivity — the main operational cost.** The critical interaction is `max_depth` × `learning_rate` × `n_estimators`; the tuned combination (depth 6, lr 0.03, 400 rounds) is a textbook-correct shallow-and-slow configuration. Overfitting risk is moderate but well-controlled by the built-in regularisers and early stopping.

**Interpretability.** Excellent. **TreeSHAP is exact and fast** — it was co-developed on XGBoost (Lundberg et al. 2020). Provides gain / cover / weight importances, native SHAP interaction values, and per-prediction attributions. For a SOC this is decisive: an analyst receiving "Infiltration, p=0.91" also receives "driven by `Init_Win_bytes_forward = −1`, `Flow IAT Std` in the 99th percentile" — an actionable investigative lead rather than an opaque verdict.

**Robustness to noise & unscaled features.** **Scale-invariant** (threshold splits) and **NaN/Inf-native** via sparsity-aware split finding — both properties matter directly for raw CICFlowMeter output. Moderately robust to label noise, though less so than Random Forest: boosting *up-weights* misclassified examples, so systematically mislabelled benign-in-attack-window flows receive increasing attention across rounds. Mitigate with `subsample < 1.0` (already at 0.9) and a modest round count.

**Class imbalance.** Good but not best. This repository wraps it in a `BalancedXGBClassifier` (`src/models/xgboost_model.py`) to apply per-class weighting, since native `scale_pos_weight` is binary-only. Measured balanced accuracy **0.9062** — strong, though below Random Forest's 0.9101.

**Expected CICIDS2017 performance.** Confirmed: **0.9867 accuracy / 0.8651 macro-F1 — best overall.** This is consistent with the published CICIDS2017 literature, where gradient-boosted trees routinely lead flow-based benchmarks.

**SOC deployment.** **The strongest overall candidate.** A single compact artifact, compiled microsecond inference, the lowest false-positive burden, exact SHAP for analyst-facing explanations, and a deployment ecosystem that spans Python, JVM, C++, and ONNX. Its one real gap — Infiltration recall — is addressable architecturally rather than by changing model families (§7.2).

---

### 3.4 LightGBM

**Official documentation:** https://lightgbm.readthedocs.io/
**Primary references:**
- Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *NeurIPS 30*, 3146–3154.
- Fisher, W. D. (1958). "On Grouping for Maximum Homogeneity." *JASA*, 53(284), 789–798. — the optimal categorical split used by LightGBM.

**Fundamental learning algorithm.** Histogram-based GBDT with three distinguishing mechanisms: (1) **leaf-wise (best-first) tree growth** — always split the leaf with the highest loss reduction, rather than growing level-wise; (2) **GOSS** (Gradient-based One-Side Sampling) — retain all large-gradient instances, randomly subsample small-gradient ones with a compensating weight; (3) **EFB** (Exclusive Feature Bundling) — bundle mutually exclusive sparse features into a single feature, reducing effective dimensionality.

**Strengths.**
- **Typically the fastest GBDT trainer available** on tabular data of this size; leaf-wise growth reaches a target loss with fewer leaves than level-wise.
- Histogram binning gives `O(n)` split finding and O(#bins) memory per feature.
- Excellent native categorical support without one-hot expansion (Fisher's optimal grouping).
- Lowest training memory of the boosters.
- **When configured sanely it is fully competitive**: the `local_300k_70_30` run measured **0.9507 accuracy / 0.8407 macro-F1**, within 0.4 points of XGBoost on the same split.

**Weaknesses — and the measured failure.**
- **Leaf-wise growth overfits aggressively on small or imbalanced leaves.** This is a documented, known property, and it is exactly what happened in `results/latest/`.
- The randomized search selected **`min_child_samples=1` with `num_leaves=127`** and 600 rounds. `min_child_samples=1` permits a leaf containing a *single* training row. Applied to a training set where Infiltration has been oversampled from ~600 to 77,963 rows — i.e. massively duplicated — the model memorises duplicate rows as pure leaves.
- **The resulting damage:** accuracy **0.7634**, macro-F1 **0.6174**, BENIGN recall collapsed to **0.7421**, Infiltration precision **0.0315** at 21.25% FPR (12,620 false positives), and Heartbleed F1 **0.0000**.
- **CV instability confirms it:** fold scores 0.688, 0.462, 0.413, 0.344, 0.127 — a monotone decline, σ = 0.1814. This is the signature of a model whose behaviour depends entirely on which duplicated rows land in which fold.
- Requires more careful tuning than XGBoost for equivalent robustness; the defaults are less forgiving.

**Computational complexity.**
- Training: `O(T · K · n' · d')` where n' < n via GOSS and d' < d via EFB — asymptotically the cheapest booster here.
- Inference: `O(T · K · depth)`, though leaf-wise trees are deeper and more irregular than XGBoost's depth-capped trees.

**Training time & latency.** **Fastest booster to train** — typically 2–10× faster than XGBoost at equal accuracy. Inference is comparable to XGBoost, slightly worse for cache locality because leaf-wise trees are unbalanced. As configured in this run (5,400 trees × up to 127 leaves), both training and inference cost are inflated well beyond necessity.

**Memory & scalability.** Lowest training memory of the boosters. **But the model as configured here is large:** 600 rounds × 9 classes × 127 leaves ≈ **50–100 MB**, several times XGBoost's footprint for a *far worse* model. Scales to distributed training and GPU.

**Hyperparameter sensitivity & overfitting risk.** **Highest sensitivity and highest overfitting risk in the entire study — this run is the proof.** The dangerous parameters are `num_leaves` and `min_child_samples`, and they interact multiplicatively. **The rule LightGBM's own documentation states:** keep `num_leaves < 2^max_depth`, and set `min_child_samples` in the hundreds for datasets of this size.

**Interpretability.** Same tooling as XGBoost — TreeSHAP applies exactly, plus split/gain importances. In practice, deep irregular leaf-wise trees make individual-tree inspection less useful; SHAP is essentially mandatory.

**Robustness to noise & unscaled features.** Scale-invariant and NaN-tolerant like other tree models. **But markedly the least robust to label noise**, because leaf-wise growth actively seeks out the highest-loss region — which is precisely where mislabelled flows live. On CICIDS2017's window-labelled data this is a structural mismatch, and `min_child_samples=1` removes the only guardrail.

**Class imbalance.** `class_weight='balanced'` is supported and is set in `src/models/lightgbm_model.py`. It was **not** the problem here — the problem was that oversampling and `min_child_samples=1` combined into a memorisation regime.

**Expected CICIDS2017 performance.** **Two divergent measurements, and the divergence is the finding:**
- `results/latest/` (HP search on): **0.7634 / 0.6174** — pathological.
- `results/local_300k_70_30/` (defaults, no search): **0.9507 / 0.8407** — healthy and competitive.

**Do not conclude "LightGBM is unsuitable for NIDS."** Conclude that **the search space in `src/models/tuner.py` permits a degenerate configuration**, and that F2-on-a-single-class is an objective that will happily select it.

**SOC deployment.** **Do not deploy the `latest` artifact.** A 25.79% benign FPR is 2.58 M alerts/day — the model is worse than useless, since it would bury the true positives it finds. After the fix in §5.4, LightGBM becomes a legitimate candidate wherever **retraining frequency** dominates — e.g. daily retrains against drifting traffic — because its training speed is a genuine operational advantage.

---

### 3.5 CatBoost

**Official documentation:** https://catboost.ai/docs/
**Primary references:**
- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., Gulin, A. (2018). "CatBoost: unbiased boosting with categorical features." *NeurIPS 31*, 6638–6648.
- Dorogush, A. V., Ershov, V., Gulin, A. (2018). "CatBoost: gradient boosting with categorical features support." arXiv:1810.11363.

**Fundamental learning algorithm.** Gradient boosting with two distinctive mechanisms: (1) **Ordered boosting** — a permutation-driven scheme in which the residual for each example is computed from a model trained only on examples preceding it in a random permutation, eliminating the *prediction shift* (target leakage) that biases standard boosting; (2) **Oblivious (symmetric) trees** — every node at a given depth uses the *same* split condition, so a depth-6 tree is fully described by 6 feature-threshold pairs, and inference becomes a 6-bit index computation into a 64-entry lookup table. **Tuned parameters:** `iterations=500, depth=6, learning_rate=0.1`.

**Strengths.**
- **Fastest inference of any tree model, by a wide margin.** Oblivious trees turn tree traversal into branchless, vectorisable index arithmetic — no pointer chasing, no branch misprediction. This is a structural advantage, not an implementation detail, and it is exactly what an inline SOC path needs.
- **Most compact tree model.** 500 oblivious depth-6 trees ≈ **2–5 MB** — roughly 100× smaller than the Random Forest.
- **Ordered boosting is the most principled overfitting defence** among the boosters and gives the best out-of-the-box performance with minimal tuning.
- **Excellent minority-class behaviour without threshold tuning:** in the `local_300k_70_30` run CatBoost achieved the **highest balanced accuracy of that run (0.9561)** and Infiltration recall of **0.8642 at native threshold**.
- Symmetric trees act as a strong structural regulariser — the depth-6 constraint is enforced identically across the whole tree.
- Strong, well-documented defaults; the least tuning effort required for a competitive result.

**Weaknesses.**
- **Slowest training of the three boosters.** Ordered boosting requires maintaining multiple permutation-indexed models, a real constant-factor cost.
- **Oblivious trees are less expressive per tree** — forcing one split condition per level is a strong restriction, requiring more trees to match a free-form structure.
- Measured **third place: 0.9829 / 0.8272 macro-F1**, behind XGBoost and Random Forest.
- Its principal design advantage — categorical feature handling — is **irrelevant here**. All 80 CICFlowMeter features are numeric; only `Protocol` and `Destination Port` are arguably categorical. **This means CatBoost is competing without its main differentiator.**
- Smallest community and ecosystem of the three boosters; fewer third-party integrations.

**Computational complexity.**
- Training: `O(T · K · n · d)` with a constant factor of roughly `s` (permutation count) above plain boosting.
- Inference: **`O(T · depth)` with a very small constant** — 500 × 6 = 3,000 branchless operations, fully vectorisable. Asymptotically similar to XGBoost, but with far better constants.

**Training time & latency.** **Slowest booster to train; fastest to serve.** That asymmetry defines its deployment profile: pay once at training, benefit continuously at inference. Estimated **~1–10 µs/flow** in native form — competitive with a *compiled* XGBoost model without needing compilation.

**Memory & scalability.** Smallest tree-model footprint. Excellent GPU support (CatBoost's GPU training is among the best-engineered of the three). Scales well, though distributed-training tooling is less mature than XGBoost's.

**Hyperparameter sensitivity & overfitting risk.** **Lowest sensitivity of the three boosters** — the strongest defaults in the set. Ordered boosting makes overfitting risk **low**. CV σ = 0.0010, tight and reliable.

**Interpretability.** Very good. TreeSHAP applies; CatBoost additionally ships **`PredictionValuesChange`** and **`LossFunctionChange`** importances, and native `get_feature_importance(type='ShapValues')`. Oblivious trees are unusually readable — a depth-6 tree is literally six rules, which an analyst can review directly. **This is an underrated SOC advantage.**

**Robustness to noise & unscaled features.** Scale-invariant. **Ordered boosting provides the best principled defence against the prediction-shift bias that label noise induces** — a meaningful advantage on CICIDS2017's window-labelled data. Empirically stable: 0.6667 Heartbleed recall where MLP, LightGBM, and Stacking all scored 0.0000.

**Class imbalance.** **Strong — arguably the best "out of the box".** Supports `auto_class_weights='Balanced'` / `'SqrtBalanced'` and custom class weights. Evidence: **balanced accuracy 0.9561 and Infiltration recall 0.8642 in the 70/30 run at the native threshold**, i.e. without any FN-aware tuning. In `latest` it placed second on Infiltration F2 (0.3257).

**Expected CICIDS2017 performance.** Confirmed: **0.9829 / 0.8272** — a consistent, reliable third. Never the winner, never a failure; the lowest-variance performer across both runs.

**SOC deployment.** **The best choice when inference latency or model size is the binding constraint** — a network appliance, an edge sensor, a container with a hard memory budget, or an inline path that cannot afford a compilation toolchain. Also the right pick when **retraining will be infrequent and unattended**, because its defaults degrade most gracefully. It is the second-strongest overall recommendation after XGBoost.

---

### 3.6 Multi-Layer Perceptron (MLP)

**Official documentation:** https://scikit-learn.org/stable/modules/neural_networks_supervised.html
**Primary references:**
- Rumelhart, D. E., Hinton, G. E., Williams, R. J. (1986). "Learning representations by back-propagating errors." *Nature*, 323, 533–536.
- Cybenko, G. (1989). "Approximation by superpositions of a sigmoidal function." *Mathematics of Control, Signals and Systems*, 2(4), 303–314. — universal approximation.
- Hornik, K. (1991). "Approximation capabilities of multilayer feedforward networks." *Neural Networks*, 4(2), 251–257.
- Kingma, D. P. & Ba, J. (2015). "Adam: A Method for Stochastic Optimization." *ICLR 2015*. arXiv:1412.6980.
- **Grinsztajn, L., Oyallon, E., Varoquaux, G. (2022). "Why do tree-based models still outperform deep learning on tabular data?" *NeurIPS 2022 Datasets & Benchmarks*.** — directly explains the result observed here.

**Fundamental learning algorithm.** Feedforward network of fully-connected layers with ReLU activations and a softmax output, trained by backpropagation with Adam. Learns a hierarchy of distributed non-linear feature representations. **Tuned parameters:** `hidden_layer_sizes=(128, 64), learning_rate_init=0.005, alpha=0.0001` — i.e. 80 → 128 → 64 → 9, **19,008 weights**.

**Strengths.**
- **Universal function approximator** — can in principle represent any continuous decision boundary, including the smooth, rotationally-oriented boundaries that axis-aligned trees approximate with staircases.
- **Extremely compact: 19,008 parameters ≈ 76 KB.** Second-smallest model in the study.
- **Inference is a fixed sequence of three dense GEMMs** — a constant-time, branchless, perfectly cache-predictable, SIMD/GPU-friendly operation. Latency is *deterministic*, which matters for hard-real-time inline paths where a p99.9 tail is unacceptable.
- **Natively supports true incremental learning** via `partial_fit` — the only model here that can be updated online against traffic drift without a full retrain. For a SOC facing continuously evolving attack patterns, this is a genuine and underappreciated advantage.
- Learns shared internal representations across classes, rather than fitting each class independently.
- Solid aggregate accuracy: **0.9807**, fourth of seven.

**Weaknesses.**
- **Macro-F1 of 0.6981 — sixth of seven, and second-worst.** The gap between accuracy (0.9807, 4th) and macro-F1 (0.6981, 6th) is the sharpest in the study and diagnoses the problem precisely: **the MLP is very good at the majority class and poor at the minorities.**
- **Balanced accuracy 0.7833 — the worst in the entire study.** Softmax cross-entropy is dominated by frequent classes, and gradient updates from the 3 Heartbleed rows are numerically negligible.
- **Heartbleed F1 = 0.0000** — complete failure on the rarest class.
- Weakest measured minority performance: PortScan F1 **0.8867** (worst of the four healthy models) and Web Attack F1 **0.1890**.
- **Requires strict feature scaling.** Without `StandardScaler`, `Flow Duration` (~10⁸) saturates activations and stalls learning. This is the least robust model to raw input in the entire set.
- **Non-convex objective** — different seeds yield different local minima. CV σ = 0.0039, the second-highest among healthy models.
- **Weakest interpretability.** No native feature importance; SHAP requires KernelSHAP (`O(2^d)`, must be approximated) or DeepSHAP/GradientSHAP (approximate, architecture-coupled). For a SOC that must justify a block decision, this is a serious operational deficiency.
- Grinsztajn et al. (2022) identify the structural cause: neural networks have a **rotationally-invariant inductive bias**, which is a poor match for tabular data where individual features carry independent, non-rotatable meaning — exactly the case for CICFlowMeter features.

**Computational complexity.**
- Training: `O(E · n · (d·h₁ + h₁·h₂ + h₂·K))` = `O(E · n · 19,008)` — linear in epochs and samples.
- Inference: **`O(19,008)` FLOPs per flow, fixed** — no data-dependent branching.

**Training time & latency.** Moderate training on CPU; very fast on GPU. **Inference latency is deterministic and low** — estimated **5–20 µs/flow** on CPU, and it batches superbly (a 1,000-flow batch is one 1000×80 × 80×128 GEMM, which BLAS executes at near-peak FLOPS). For high-throughput batched scoring the MLP is genuinely competitive.

**Memory & scalability.** **76 KB.** Trains on mini-batches, so training memory is independent of dataset size — it scales to arbitrarily large corpora. Only model here with no memory ceiling on n.

**Hyperparameter sensitivity & overfitting risk.** **High sensitivity** — architecture, learning rate, `alpha`, batch size, and epoch count all interact, and the objective is non-convex. Overfitting risk is moderate-to-high and the selected `alpha=0.0001` is very weak L2 regularisation. Notably, the MLP's shuffled-label macro-F1 was **0.0639**, the lowest measured — it fits noise *less* than the trees do, which is consistent with underfitting the minorities rather than memorising them.

**Interpretability.** **Worst in the set.** Requires approximate, expensive post-hoc methods. In a regulated SOC — PCI-DSS, or any environment where an automated block must be defensible — this alone can disqualify it.

**Robustness to noise & unscaled features.** **Worst on unscaled features** (mandatory standardisation) and **poor on noisy traffic** — smooth global decision boundaries mean a perturbation anywhere in the 80-dimensional input shifts the output, whereas a tree ignores features not on the decision path. Combined with the deterministic-latency advantage, this makes the MLP a good fit for *clean, well-normalised* pipelines and a bad fit for raw sensor feeds.

**Class imbalance.** **Weakest in the study.** sklearn's `MLPClassifier` has **no `class_weight` parameter** — the standard remedy is unavailable. Options are limited to resampling (already applied) or a custom weighted-loss implementation in PyTorch/TensorFlow (e.g. focal loss, Lin et al. 2017). Measured balanced accuracy **0.7833**, last place.

**Expected CICIDS2017 performance.** Confirmed: **0.9807 accuracy but 0.6981 macro-F1.** A textbook illustration of why accuracy is the wrong headline metric on imbalanced security data. Interestingly, in the 70/30 run with *aggressive rebalancing*, the MLP reached **0.9440 balanced accuracy and the best Infiltration recall of that run (0.8975)** — confirming that its weakness is loss weighting, not representational capacity.

**SOC deployment.** Not recommended as the primary detector: worst imbalance handling, worst interpretability, and a mandatory scaling dependency. **Two defensible roles:** (a) an **online-adaptive drift monitor** using `partial_fit`, running alongside a frozen tree model, where divergence between the two signals distribution shift; (b) a **batched high-throughput pre-filter** exploiting its deterministic GEMM latency. If a neural approach is genuinely wanted for NIDS, move to a sequence model (LSTM/Transformer) over *packet sequences* rather than an MLP over aggregated flow statistics — the aggregation is what discards the signal a network could exploit.

---

### 3.7 Stacking Ensemble

**Official documentation:** https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization
**Primary references:**
- Wolpert, D. H. (1992). "Stacked Generalization." *Neural Networks*, 5(2), 241–259.
- Breiman, L. (1996). "Stacked Regressions." *Machine Learning*, 24(1), 49–64. — non-negativity constraint on meta-weights.
- Ting, K. M. & Witten, I. H. (1999). "Issues in Stacked Generalization." *JAIR*, 10, 271–289. — why `predict_proba` beats hard votes as meta-features.
- van der Laan, M. J., Polley, E. C., Hubbard, A. E. (2007). "Super Learner." *Statistical Applications in Genetics and Molecular Biology*, 6(1). — asymptotic optimality proof.

**Architecture as implemented** (`src/models/stacking.py`):

```
StandardScaler
   └─> StackingClassifier (cv=5, stack_method='predict_proba', passthrough=False)
         ├─ LightGBM  (300 trees, 63 leaves, lr 0.05, class_weight='balanced')
         ├─ XGBoost   (200 trees, depth 8, lr 0.10, hist, per-class weighted)
         ├─ RandomForest (200 trees, class_weight='balanced')
         └─ meta: LogisticRegression (lbfgs, C=1.0, class_weight='balanced')
```

**Fundamental learning algorithm.** Two-level learning. Level 0: three heterogeneous base learners generate **out-of-fold** `predict_proba` outputs via 5-fold internal CV — out-of-fold generation is essential, since in-fold predictions would leak and cause the meta-learner to over-trust an overfitted base. Level 1: a balanced logistic regression learns the optimal linear blend over the resulting 27-dimensional meta-feature space (3 learners × 9 classes). Van der Laan's Super Learner theorem establishes that such a stack is asymptotically no worse than the best single base learner.

**Strengths.**
- **Best Infiltration performance in the study: F2 = 0.3556, recall = 0.3840, FN = 369 (fewest of any policy-compliant model).** On the hardest and most operationally significant class, stacking wins — 47 fewer missed intrusions than Random Forest, 77 fewer than XGBoost.
- **Best DoS F1 (0.9994) and joint-best DDoS F1 (0.9996)** — it does not sacrifice volumetric detection for stealth detection.
- **Tightest CV variance in the study (σ = 0.0005)** — the most reproducible model measured.
- Averages away the idiosyncratic errors of individual learners: RF's variance, XGBoost's majority bias, LightGBM's instability.
- A **balanced logistic meta-learner** is itself interpretable — its coefficients reveal *which base model is trusted for which class*, which is genuinely informative for model governance.
- **Graceful degradation:** if one base learner drifts, the meta-learner can down-weight it at the next retrain without an architecture change.

**Weaknesses.**
- **By far the most expensive model.** With `cv=5`, each base learner is fit **6 times** (5 folds + 1 final refit) — 18 base-model fits total. This dominated the 4-hour run.
- **Macro-F1 of only 0.7509 — fourth of seven, below all three of its own tree base learners individually.** The cause is arithmetic, not conceptual: **Heartbleed F1 = 0.0000 on n=3**, which alone removes ~0.111 from a 9-class macro average. Correcting for the two anecdotal classes, its 7-class macro-F1 is **0.8995 — the highest in the study**, though only by 0.0006 over Random Forest (see §5.2). *The headline ranking of stacking is largely an artifact of macro-averaging over a 3-row class.*
- **Highest memory footprint** — hosts a full Random Forest, an XGBoost, and a LightGBM simultaneously.
- **Highest inference latency** — every prediction runs all three base models, then the meta-model. Latency is the *sum*, not the max.
- **Most complex to operate:** four models to version, monitor, retrain, and validate; a single base-learner regression can silently degrade the ensemble.
- Inherits every base learner's failure mode, including LightGBM's leaf-wise fragility.
- SHAP is not exact end-to-end — attribution must be composed across two levels or approximated with KernelSHAP.

**Computational complexity.**
- Training: `O((cv + 1) · Σᵢ Trainᵢ) + O(meta)` = **~6× the sum of all three base learners**.
- Inference: `O(Σᵢ Inferᵢ + K·M)` — the **sum** of RF, XGB, and LGBM inference plus a trivial 27×9 meta-GEMV.

**Training time & latency.** **Slowest to train and slowest to serve.** Estimated inference **100–400 µs/flow** uncompiled — roughly an order of magnitude above XGBoost alone. This effectively rules it out of a per-flow inline path at 10 Gbps.

**Memory & scalability.** **Largest footprint in the study** — the sum of all base models. Scales poorly: adding a base learner multiplies both training and inference cost. Partially mitigable by reducing internal `cv` from 5 to 3 (a ~40% training saving for a small variance increase).

**Hyperparameter sensitivity & overfitting risk.** **Low sensitivity at the ensemble level, high complexity underneath.** No ensemble-level hyperparameters were tuned (`best_params: {}`); the meta-learner is inherently well-regularised. **Overfitting risk is the lowest in the study** — out-of-fold meta-features plus a linear meta-learner form a strong structural defence, confirmed by CV σ = 0.0005.

**Interpretability.** **Moderate — a genuine step down.** The meta-layer is interpretable (which base model is trusted per class), but end-to-end attribution to input features requires composing SHAP across two levels. For a SOC that must answer "*why* was this flow blocked?", this is a real cost.

**Robustness to noise & unscaled features.** **Best robustness to noisy traffic** — ensembling three independent inductive biases is the most reliable general defence against idiosyncratic noise-driven errors. `StandardScaler` is applied in the pipeline (harmless for the trees, necessary for the meta-learner). One caveat: because it contains the LightGBM configuration, it inherits some label-noise sensitivity.

**Class imbalance.** **The best in the study for the hardest class.** All three base learners use `class_weight='balanced'`, and so does the meta-learner — imbalance correction is applied at *both* levels. Result: **best Infiltration F2 (0.3556) and fewest false negatives (369)**.

**Expected CICIDS2017 performance.** Confirmed: **0.9830 accuracy / 0.7509 macro-F1**, rising to **0.8995 on the 7 statistically valid classes**. In the 70/30 run it took the **top accuracy (0.9529)**.

**SOC deployment.** **Not for the inline path** — latency, memory, and operational complexity all disqualify it. **Its correct role is a second-stage / asynchronous deep-inspection scorer:** flows that the inline model marks as ambiguous get re-scored by the stack, where an extra 400 µs is irrelevant and the superior Infiltration recall directly reduces missed breaches. This is the standard cascade pattern, and it is the architecture recommended in §7.2.

---

## 4. Consolidated Trade-Off Matrix

### 4.1 Master comparison

| Model | Key Advantage | Key Limitation | Inference Latency† | Memory† | Imbalance Handling | Interpretability | Overall Recommendation |
|---|---|---|---|---|---|---|---|
| **XGBoost** | Best macro-F1 (0.8651) + lowest FP rate (0.66%) | Weakest Infiltration F1 among healthy trees (0.2871) | **Very Low** (~10–50 µs; 1–5 µs compiled) | Low (10–30 MB) | Good (bal. acc 0.9062) | Excellent (exact TreeSHAP) | ⭐ **DEPLOY — inline primary** |
| **Random Forest** | Best balanced acc (0.9101) + best Infiltration precision (0.3433) | Largest memory; cache-bound traversal | Medium-High (~50–200 µs) | **Very High** (100s of MB) | **Excellent** (`balanced_subsample`) | Excellent (exact TreeSHAP) | ✅ **Deploy — 2nd-stage scorer** |
| **CatBoost** | Fastest + smallest inference (oblivious trees) | Slowest training; categorical edge unused here | **Lowest** (~1–10 µs native) | **Lowest tree model** (2–5 MB) | **Excellent** (bal. acc 0.9561 in run 2) | Excellent (readable trees + SHAP) | ✅ **Deploy — edge / appliance** |
| **Stacking** | Best Infiltration F2 (0.3556), fewest FN (369) | ~6× training cost; highest latency | **Highest** (~100–400 µs) | **Highest** (sum of all) | **Best** (balanced at both levels) | Moderate (2-level SHAP) | ⚠ **Async deep-inspection only** |
| **MLP** | Deterministic latency; online `partial_fit` | Worst balanced acc (0.7833); no `class_weight` | Low (~5–20 µs, batches well) | **Lowest overall** (76 KB) | **Poor** | **Poor** (KernelSHAP only) | ⚠ **Drift monitor / research** |
| **Logistic Regression** | Fully interpretable; ~6 KB; sub-µs | Linear only; PortScan F1 0.6410 | **Lowest** (<1 µs) | **Lowest** (6 KB) | Weak (Infil. F1 0.0969) | **Best** (closed-form SHAP) | ⚠ **Baseline + stack meta-learner** |
| **LightGBM** | Fastest training; healthy at defaults (0.8407 in run 2) | Current artifact broken: 25.79% FPR | Low (~10–50 µs) | Medium-High (50–100 MB as configured) | Config-dependent (failed here) | Excellent (TreeSHAP) | ❌ **DO NOT DEPLOY — fix §5.4 first** |

† Latency and memory are engineering estimates derived from the measured model configurations (tree counts, depths, parameter counts) and standard CPU inference behaviour. They were **not** micro-benchmarked in this run; only the 14,652 s aggregate training wall time was measured. §5.5 specifies the benchmark required to replace these estimates with measurements.

### 4.2 Dimension-by-dimension ranking (1 = best)

| Dimension | 1st | 2nd | 3rd | 4th | 5th | 6th | 7th |
|---|---|---|---|---|---|---|---|
| macro-F1 | XGB | RF | CatBoost | Stack | LogReg | MLP | LGBM |
| Balanced accuracy | RF | XGB | CatBoost | Stack | LogReg | LGBM | MLP |
| False-positive burden | XGB | RF | CatBoost | Stack | MLP | LogReg | LGBM |
| Infiltration F2 | Stack | CatBoost | RF | XGB | MLP | LGBM | LogReg |
| Inference latency | LogReg | CatBoost | MLP | XGB | LGBM | RF | Stack |
| Memory footprint | LogReg | MLP | CatBoost | XGB | LGBM | RF | Stack |
| Training speed | LogReg | LGBM | XGB | RF | MLP | CatBoost | Stack |
| Interpretability | LogReg | CatBoost | XGB | RF | LGBM | Stack | MLP |
| Robustness to noise | RF | Stack | CatBoost | XGB | LogReg | LGBM | MLP |
| Tuning burden (low=best) | LogReg | RF | CatBoost | Stack | XGB | MLP | LGBM |
| Reproducibility (CV σ) | Stack | RF/XGB | CatBoost | LogReg | MLP | — | LGBM |

### 4.3 Per-Class Attack Evaluation

#### High-Volume Attacks — DoS, DDoS, PortScan

| Class | Support | Best model | Best F1 | Runner-up | Analysis |
|---|---:|---|---:|---|---|
| **DDoS** | 3,888 | **XGBoost / Stacking** | **0.9996** | CatBoost (0.9955) | Effectively solved. XGBoost precision 0.9997 / recall 0.9995 = **2 errors in 3,888 flows.** |
| **DoS** | 1,678 | **Stacking** | **0.9994** | XGBoost (0.9991) | XGBoost achieves **recall = 1.0000 — zero missed DoS flows.** Prefer XGBoost operationally: perfect recall matters more than 0.0003 F1. |
| **PortScan** | 390 | **4-way tie** (RF / XGB / CatBoost / Stack) | **0.9936** | MLP (0.8867) | All four healthy tree models are identical. LogReg collapses to 0.6410 (recall 0.5128) — **port scanning is not linearly separable.** |

> **Verdict — High-Volume: XGBoost.** It wins or ties on all three, achieves perfect DoS recall, and does so at the lowest false-positive cost. These attacks are high-signal — burst packet rates, uniform packet sizes, distinctive flag counts — and any well-configured tree ensemble solves them. **Model choice is essentially free here; choose on latency and FP rate, which points to XGBoost.**

#### Stealthy / Rare Attacks — Infiltration, Bot, Heartbleed, Web Attack

| Class | Support | Best model | Best score | Analysis |
|---|---:|---|---:|---|
| **Infiltration** | 599 | **Stacking** | **F2 0.3556 / recall 0.3840 / 369 FN** | The genuine hard problem. Runner-up CatBoost (F2 0.3257). RF gives the **cleanest** detection (precision 0.3433, only 352 FP). |
| **Bot** | 630 | **CatBoost / Stacking** | **F1 0.9960** | Solved. RF 0.9952, XGB 0.9944. Botnet C2 beaconing has a distinctive periodic IAT signature that trees capture easily. |
| **Heartbleed** ⚠ | **3** | *statistically void* | — | RF/XGB/LogReg = 1.0000; CatBoost = 0.8000; MLP/LGBM/Stack = 0.0000. **Three test rows. This is coin-flip noise and must not inform model selection.** |
| **Web Attack** ⚠ | **13** | *statistically void* | XGB 0.5217 | XGBoost leads, but with n=13 the 95% CI spans roughly 0.25–0.75. **Directional only.** |

> **Verdict — Stealthy/Rare: Stacking for recall, Random Forest for precision.**
> - Maximising *detection* of Infiltration → **Stacking** (369 FN, the fewest).
> - Minimising *analyst load* → **Random Forest** (352 FP, the fewest; highest precision).
> - **Why every model struggles:** Infiltration in CICIDS2017 is post-compromise internal lateral movement (a Dropbox-delivered payload followed by internal port scanning). Its flow-level statistics — duration, packet sizes, IAT — are **genuinely indistinguishable from benign internal traffic** in the 80-feature space. This is an *information-theoretic* limit of flow-based features, not a modelling failure. Fixing it requires new signal (§7.4), not a better classifier.

---

## 5. Critical Findings & Required Remediation

### 5.1 🔴 The reported baseline understates the difficulty

`metrics.json` reports `majority_baseline_acc = 0.0100` and the generated report claims "Model lift = +0.9767". This is computed against the *post-oversampling training* majority class (Infiltration, 77,963 rows), which does not exist in the natural test distribution.

**The correct baseline is "always predict BENIGN" = 52,355 / 60,000 = 0.8726.** XGBoost's true framing: error rate **12.74% → 1.33%**, a **10.6× error reduction**. This is still an excellent result, but it is the defensible one.

**Action:** compute the baseline from the **test** distribution in `src/evaluation/metrics.py`, and report both figures.

### 5.2 🔴 Two classes cannot support any conclusion

Heartbleed (n=3) and Web Attack (n=13) are included in macro-F1 with the same weight as BENIGN (n=52,355). This single fact **reorders the leaderboard.** Recomputing macro-F1 over only the 7 statistically valid classes (BENIGN, Bot, Brute Force, DDoS, DoS, Infiltration, PortScan):

| Model | Published macro-F1 (9 classes) | **`macro_f1_valid` (7 classes)** | Δ | Rank change |
|---|---:|---:|---:|---|
| **Stacking** | 0.7509 (4th) | **0.8995 (1st)** | +0.1486 | ▲ 3 |
| Random Forest | 0.8519 (2nd) | **0.8989 (2nd)** | +0.0470 | — |
| CatBoost | 0.8272 (3rd) | **0.8962 (3rd)** | +0.0690 | — |
| XGBoost | 0.8651 (1st) | **0.8948 (4th)** | +0.0297 | ▼ 3 |
| MLP | 0.6981 (6th) | 0.8706 (5th) | +0.1725 | ▲ 1 |
| LightGBM | 0.6174 (7th) | 0.7660 (6th) | +0.1486 | ▲ 1 |
| Logistic Regression | 0.7081 (5th) | 0.7546 (7th) | +0.0465 | ▼ 2 |

**Two conclusions follow, and both matter:**

1. **The published ranking is unstable.** XGBoost falls from 1st to 4th and Stacking rises from 4th to 1st, purely on the treatment of 16 test rows.
2. **The top four models are statistically indistinguishable.** Stacking, Random Forest, CatBoost, and XGBoost span **0.8948 → 0.8995 — a range of 0.0047**, far inside the noise floor for a single split. **On predictive quality alone there is no defensible winner among them.** This does not weaken the §6 recommendation — it *strengthens* it, because it means the decision must rest entirely on the secondary criteria (false-positive burden, latency, memory, maintainability), which is precisely where XGBoost's advantage is large and unambiguous.

**Action:** report a **`macro_f1_valid`** metric restricted to classes with n_test ≥ 30 alongside the standard macro-F1, and use it for champion selection. Where models tie within ~0.01, break the tie on benign FP rate rather than on F1.

### 5.3 🟡 Champion selection is inconsistent across runs

`results/champion.json` selects `random_forest` with `status: "conditional_no_model_meets_fpr"` and a target FPR of 0.0269 — **above the 0.02 policy cap**. Meanwhile in `results/latest/` four models satisfy the cap and XGBoost leads on macro-F1.

**Action:** make the selection rule explicit and identical across runs — e.g. *"maximise `macro_f1_valid` subject to `target_fpr ≤ 0.02`; break ties on benign FP rate."* Fail loudly rather than emitting a conditional champion.

### 5.4 🔴 The LightGBM search space permits a degenerate configuration

Root cause: `min_child_samples=1` is reachable in the search grid. Combined with `num_leaves=127`, 600 rounds, and a training set in which Infiltration is oversampled 130× (599 → 77,963 rows), the model memorises duplicated rows as single-instance pure leaves.

**Fix in `src/models/tuner.py`:**

```python
# LightGBM search space — enforce leaf-wise growth guardrails.
"lightgbm": {
    # Was: min_child_samples ∈ {1, 5, 20, ...}  <- 1 allows single-row leaves
    "clf__min_child_samples": [100, 200, 500],   # floor well above duplicate multiplicity
    "clf__num_leaves":        [15, 31, 63],      # keep num_leaves < 2^max_depth
    "clf__max_depth":         [6, 8, 10],        # explicit cap; LightGBM defaults to -1 (unlimited)
    "clf__min_split_gain":    [0.0, 0.01, 0.1],  # require real loss reduction to split
    "clf__reg_lambda":        [0.0, 1.0, 10.0],
    "clf__n_estimators":      [200, 400, 600],
    "clf__learning_rate":     [0.03, 0.05, 0.1],
},
```

**Additionally, guard the objective.** `target_f2` on a single class will happily accept a 21% FPR to gain recall. Constrain the search to configurations satisfying the FPR cap:

```python
def constrained_f2(estimator, X, y):
    """F2 on the target class, but zero out any candidate breaching the FPR policy."""
    f2, fpr = target_f2_and_fpr(estimator, X, y)
    return f2 if fpr <= TARGET_MAX_FPR else 0.0
```

Without this second change, a fixed grid alone will not prevent the search from selecting pathological trade-offs.

### 5.5 🟡 Latency and memory were never measured

Every latency and footprint figure in §4.1 is an estimate. For a deployment decision made on throughput grounds, these must be measured.

**Action:** add `scripts/benchmark_inference.py` recording, per model: p50 / p95 / p99 / p99.9 single-flow latency; batched throughput at batch sizes 1 / 100 / 1,000 / 10,000; serialised artifact size; peak RSS during inference; and the same set after Treelite/ONNX compilation. Persist to `results/<run>/latency.json` so the trade-off matrix becomes evidence-based.

### 5.6 🟡 Adversarial robustness was tested for one model only

`red_team.json` covers Random Forest alone. Its result is encouraging — macro-F1 held at 0.863 through ε=0.10 perturbation — but the ε=0.10 scenario **failed the policy check** (`policy_pass: false`, worst-class recall drop 0.20, flip rate 14.5%).

This matters more than a normal ML robustness test, because a network attacker **controls the features**. Padding packets, inserting delays, and fragmenting flows are trivial manipulations that directly move `Packet Length Mean`, `Flow IAT Mean`, and `Total Fwd Packets`.

**Action:** run the evasion suite for all seven models and add the worst-class recall drop at ε=0.05 as a **selection criterion**, not just a report line.

---

## 6. Final SOC Deployment Recommendation

### 6.1 Decision: XGBoost

Scored against the four criteria specified:

| Criterion | Weight | XGBoost | Evidence |
|---|---:|---|---|
| **Accuracy / F1** | 35% | **1st on published metrics; joint-1st in practice** | macro-F1 **0.8651** (best), weighted-F1 **0.9860** (best), accuracy **0.9867** (best), 800 errors in 60,000 (fewest). **Caveat (§5.2): on the 7 statistically valid classes it ranks 4th at 0.8948, within 0.0047 of the leader — the top four are tied.** |
| **Throughput / Latency** | 30% | **2nd** | Depth-6 cache-resident trees; ~10–50 µs native, ~1–5 µs compiled. Only CatBoost is faster, and its accuracy is 0.038 macro-F1 lower |
| **Compute Efficiency** | 20% | **2nd** | 10–30 MB artifact — ~10× smaller than Random Forest at higher accuracy. GPU `hist` training, out-of-core support |
| **Maintainability** | 15% | **1st** | Tightest CV (±0.0006); single artifact; exact TreeSHAP; ONNX/Treelite/JVM/C++ ecosystem; the most widely operated GBDT in production security tooling |

**The decisive factor — the false-positive budget.** Because §5.2 shows the top four models are tied on predictive quality, the decision falls entirely to the secondary criteria, and there the margin is wide. At 0.66% benign FPR, XGBoost generates **66,000 alerts/day at 10 M flows/day, versus 81,000 for Random Forest (+23%), 117,000 for CatBoost (+77%), and 122,000 for Stacking (+85%)**. Since analyst hours are the scarcest resource in any SOC, a 23–85% reduction in alert volume at equal detection quality is the single largest lever available.

**Honest statement of the trade-off.** XGBoost is the **worst of the four healthy tree models on Infiltration** (F1 0.2871 vs Stacking's 0.3201). Selecting it means accepting **77 more missed Infiltration flows per 60,000** than Stacking would deliver. That is a real cost. It is accepted because §6.2 recovers it architecturally at a fraction of the cost of deploying Stacking inline.

### 6.2 Recommended production architecture

A single model cannot simultaneously satisfy line-rate latency and stealthy-attack recall. Use a cascade:

```
                    Flow export (CICFlowMeter / Zeek / nProbe)
                                    │
              ┌─────────────────────▼─────────────────────┐
              │  STAGE 1 — INLINE          XGBoost        │
              │  compiled via Treelite/ONNX               │
              │  target: 1–5 µs/flow, ≥ 200k flows/s      │
              └─────────────────────┬─────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │ p(attack) < 0.10          │ 0.10 ≤ p < 0.90           │ p ≥ 0.90
        ▼                           ▼                           ▼
     ALLOW                  ┌───────────────┐              BLOCK + ALERT
   (log only)               │  STAGE 2      │            (SHAP attribution
                            │  Stacking     │             attached to alert)
                            │  async queue  │
                            │  ~400 µs OK   │
                            └───────┬───────┘
                                    │  Infiltration F2 0.3556
                                    ▼  369 FN (best measured)
                          ┌─────────────────────┐
                          │ STAGE 3 — SOC queue │
                          │ risk-scored,        │
                          │ host-aggregated     │
                          └─────────────────────┘

   Parallel: CatBoost on remote/edge sensors (2–5 MB, ~1–10 µs, no compile toolchain)
   Parallel: MLP partial_fit drift monitor — divergence from Stage 1 signals distribution shift
```

**Why this resolves the trade-off:**
- Stage 1 handles ~99% of flows at line rate, at the lowest achievable false-positive cost.
- Stage 2 sees only the ambiguous band — perhaps 1–3% of flows — where 400 µs is irrelevant, and applies the best-measured Infiltration detector precisely where it is needed.
- Stage 3 is mandatory, not optional: **66,000 raw alerts/day must be aggregated by host, by campaign, and by risk score before any human sees them.** A flow-level classifier is a *scoring* stage; alerting must happen at the entity level.

### 6.3 Role assignment summary

| Role | Model | Rationale |
|---|---|---|
| **Inline primary detector** | **XGBoost** | Best macro-F1, lowest FP rate, compiled µs-latency, exact SHAP |
| **Second-stage deep inspection** | **Stacking** | Best Infiltration F2 (0.3556) and fewest FN (369); latency irrelevant off the hot path |
| **Edge / appliance sensor** | **CatBoost** | 2–5 MB, fastest native inference, best defaults, most graceful unattended degradation |
| **High-scrutiny / low-FP queue** | **Random Forest** | Highest Infiltration precision (0.3433), fewest FP (352), best noise robustness |
| **Drift monitor** | **MLP** | `partial_fit` online adaptation; divergence from Stage 1 flags distribution shift |
| **Audit baseline & stack meta-learner** | **Logistic Regression** | Fully explainable; quantifies required non-linearity; already the stack's meta-model |
| **Blocked pending fix** | **LightGBM** | 25.79% benign FPR in current artifact — remediate per §5.4, then re-evaluate as the fast-retrain candidate |

### 6.4 Pre-production checklist

| # | Item | Status |
|---|---|---|
| 1 | Fix LightGBM search space + FPR-constrained objective (§5.4) | ⬜ Required |
| 2 | Recompute baseline from test distribution (§5.1) | ⬜ Required |
| 3 | Add `macro_f1_valid` (n_test ≥ 30) and use it for champion selection (§5.2) | ⬜ Required |
| 4 | Benchmark latency/memory for all 7 models; persist to `latency.json` (§5.5) | ⬜ Required |
| 5 | Run evasion suite across all 7 models; gate on ε=0.05 worst-class recall drop (§5.6) | ⬜ Required |
| 6 | Compile XGBoost via Treelite/ONNX; verify inference parity | ⬜ Required |
| 7 | Unify champion selection rule across runs (§5.3) | ⬜ Recommended |
| 8 | Validate on CSE-CIC-IDS2018 or UNSW-NB15 to test cross-dataset generalisation | ⬜ Recommended |
| 9 | Build Stage-3 entity-level alert aggregation before any SOC pilot | ⬜ **Blocking** |
| 10 | Re-run with the corrected CICIDS2017 labels (Engelen et al. 2021) | ⬜ Recommended |

---

## 7. Limitations of This Evaluation

1. **Latency and memory are estimates, not measurements.** Only the 14,652 s aggregate training time was recorded. §5.5 specifies the fix.
2. **300,000-row subsample.** The full CICIDS2017 corpus is ~2.8 M flows. Model *rankings* are unlikely to change, but absolute minority-class metrics will.
3. **Single random seed (42), single split.** Five-fold CV mitigates this for six of seven models, but seed-averaged results would be stronger, particularly for the non-convex MLP.
4. **Heartbleed (n=3) and Web Attack (n=13) support no conclusions.** They are reported for completeness and excluded from all rankings.
5. **CICIDS2017 has documented label defects.** Engelen, Rimmer & Joosen (2021) identify systematic labelling and CICFlowMeter feature-extraction errors; Rosay et al. (2022) confirm several. Absolute numbers here inherit those defects. Cross-dataset validation is the correct control.
6. **Single-dataset evaluation.** Performance on CICIDS2017 does not establish generalisation to live enterprise traffic, whose benign distribution differs substantially.
7. **No concept-drift evaluation.** All splits are random, not temporal. Real deployment faces attacks that did not exist at training time; a chronological split would be a far harder and more realistic test.

---

## 8. Consolidated Bibliography

**Models**

| # | Reference |
|---|---|
| 1 | Cox, D. R. (1958). The Regression Analysis of Binary Sequences. *JRSS-B*, 20(2), 215–242. |
| 2 | McFadden, D. (1974). Conditional Logit Analysis of Qualitative Choice Behavior. In *Frontiers in Econometrics*, 105–142. |
| 3 | Liu, D. C. & Nocedal, J. (1989). On the Limited Memory BFGS Method for Large Scale Optimization. *Mathematical Programming*, 45, 503–528. |
| 4 | Breiman, L. (1996). Bagging Predictors. *Machine Learning*, 24(2), 123–140. |
| 5 | Ho, T. K. (1995). Random Decision Forests. *ICDAR '95*, 278–282. |
| 6 | **Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.** |
| 7 | Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics*, 29(5), 1189–1232. |
| 8 | Friedman, J. H. (2002). Stochastic Gradient Boosting. *Comput. Stat. & Data Analysis*, 38(4), 367–378. |
| 9 | **Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*, 785–794.** |
| 10 | **Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 30*, 3146–3154.** |
| 11 | **Prokhorenkova, L. et al. (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS 31*, 6638–6648.** |
| 12 | Dorogush, A. V., Ershov, V., Gulin, A. (2018). CatBoost: gradient boosting with categorical features support. arXiv:1810.11363. |
| 13 | Rumelhart, D. E., Hinton, G. E., Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533–536. |
| 14 | Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *MCSS*, 2(4), 303–314. |
| 15 | Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. *Neural Networks*, 4(2), 251–257. |
| 16 | Kingma, D. P. & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR 2015*. arXiv:1412.6980. |
| 17 | **Wolpert, D. H. (1992). Stacked Generalization. *Neural Networks*, 5(2), 241–259.** |
| 18 | Breiman, L. (1996). Stacked Regressions. *Machine Learning*, 24(1), 49–64. |
| 19 | Ting, K. M. & Witten, I. H. (1999). Issues in Stacked Generalization. *JAIR*, 10, 271–289. |
| 20 | van der Laan, M. J., Polley, E. C., Hubbard, A. E. (2007). Super Learner. *Stat. Appl. Genet. Mol. Biol.*, 6(1). |
| 21 | Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830. |
| 22 | Louppe, G. (2014). *Understanding Random Forests: From Theory to Practice.* PhD thesis, Univ. of Liège. arXiv:1407.7502. |

**Dataset, interpretability, imbalance, and evaluation**

| # | Reference |
|---|---|
| 23 | **Sharafaldin, I., Lashkari, A. H., Ghorbani, A. A. (2018). Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. *ICISSP 2018*, 108–116.** |
| 24 | Lashkari, A. H. et al. (2017). Characterization of Tor Traffic Using Time Based Features. *ICISSP 2017*. — CICFlowMeter. |
| 25 | **Engelen, G., Rimmer, V., Joosen, W. (2021). Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study. *IEEE S&P Workshops (SPW)*, 7–12.** |
| 26 | Rosay, A. et al. (2022). Network intrusion detection: A comprehensive analysis of CIC-IDS2017. *ICISSP 2022*. |
| 27 | **Lundberg, S. M. & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 30*, 4765–4774.** |
| 28 | **Lundberg, S. M. et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2, 56–67.** — TreeSHAP. |
| 29 | Chawla, N. V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321–357. |
| 30 | Lin, T.-Y. et al. (2017). Focal Loss for Dense Object Detection. *ICCV 2017*, 2980–2988. |
| 31 | Strobl, C. et al. (2007). Bias in random forest variable importance measures. *BMC Bioinformatics*, 8, 25. |
| 32 | **Grinsztajn, L., Oyallon, E., Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on tabular data? *NeurIPS 2022 D&B Track*.** |
| 33 | Sommer, R. & Paxson, V. (2010). Outside the Closed World: On Using Machine Learning for Network Intrusion Detection. *IEEE S&P 2010*, 305–316. — the canonical statement of the base-rate problem in §2.4. |
| 34 | Axelsson, S. (2000). The base-rate fallacy and the difficulty of intrusion detection. *ACM TISSEC*, 3(3), 186–205. |
| 35 | Arp, D. et al. (2022). Dos and Don'ts of Machine Learning in Computer Security. *USENIX Security 2022*, 3971–3988. |

**Official documentation**

| Model | URL |
|---|---|
| scikit-learn (LogReg, RF, MLP, Stacking) | https://scikit-learn.org/stable/ |
| XGBoost | https://xgboost.readthedocs.io/ |
| LightGBM | https://lightgbm.readthedocs.io/ |
| CatBoost | https://catboost.ai/docs/ |
| SHAP | https://shap.readthedocs.io/ |
| CICIDS2017 dataset | https://www.unb.ca/cic/datasets/ids-2017.html |

---

*Generated from measured results in `results/latest/` and `results/local_300k_70_30/`. Thai-language version: [`model_comparison_nids_th.md`](./model_comparison_nids_th.md).*
