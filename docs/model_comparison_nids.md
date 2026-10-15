# Comparative Evaluation of 7 ML Models for a CICIDS2017 Network Intrusion Detection System

**Document type:** Technical evaluation report
**Author role:** ML Research + Security Operations
**Date:** 15 October 2026 (revised from the 27 July 2026 edition)
**Evidence base:** measured results from run `results/cicids2017_temporal_v1/`, the run used in the final project report and the one `results/champion.json` points to

> Thai-language version: [`model_comparison_nids_th.md`](./model_comparison_nids_th.md)

> **What changed from the previous edition** — the 27 July edition used an older run: a 300,000-row subsample, a random split and targeted oversampling of Infiltration. That run broke LightGBM and led to an XGBoost recommendation. This edition uses the full 2,497,980-row corpus, a chronological 70/30 split, identical class weighting for every model, and measured latency. The results and the recommendation change accordingly, and several issues the previous edition raised are now fixed (section 5.1).

---

## 0. Executive Summary

All seven models were trained and evaluated on CICIDS2017 collapsed to **9 classes**, using **77 flow-level features** from the full 2,497,980-row corpus. Train and test are split **chronologically, 70/30, inside every (capture file, class) group**. Every model gets the same tuning budget and the same imbalance handling (`class_weight`).

**Headline:** **CatBoost is the recommended model.** It wins all three pre-declared rankings:
- **Overall:** highest macro-F1, 0.8873.
- **Security:** highest recall_macro under FPR ≤ 1%, 0.8970.
- **Deployment:** lowest p95 latency (8.39 ms per 1,000 flows) among models within 0.02 macro-F1 of the best.

It also misses the fewest attacks (2,049 of 127,721 attack flows) and its model file is only 2.5 MB.

**Five findings that matter more than the ranking table:**

1. **The top three boosting models are very close.**
   - Macro-F1: CatBoost 0.8873, LightGBM 0.8774, XGBoost 0.8773.
   - Counting only the 7 classes with ≥ 30 test rows (`f1_macro_reportable`), LightGBM leads slightly, 0.8627 versus 0.8613.
   - Predictive quality is therefore close. CatBoost wins on secondary criteria: it misses about 3× fewer attacks (2,049 vs 5,483–6,220) and is 6–17× faster.
2. **Bot is the hardest class for every model.** Recall is 0.15–0.16 in all seven, because the chronological split puts late Bot traffic in test that differs from the early Bot traffic used for training. This is distribution shift inside one capture, not a weakness of any one model (section 4.3).
3. **LightGBM and XGBoost give the fewest false positives.** Their FPR is 0.022% and 0.038%, against 0.065% for CatBoost. Where false alarms cost more than missed attacks, these two are the alternatives, at the price of roughly 3× more missed attacks.
4. **Stacking is worse than its own base learners.** Its macro-F1 of 0.6822 is below LightGBM, XGBoost and Random Forest, the models it stacks. The main cause is many BENIGN flows predicted as Brute Force (Brute Force precision only 0.108). It also takes the longest to train (2,755 s).
5. **Accuracy still misleads.** The "always BENIGN" baseline already reaches 0.8296 accuracy. MLP reaches 0.9331 accuracy, yet its macro-F1 is only 0.5369 and it misses 19,677 attack flows. Decisions therefore rest on macro-F1, per-class recall and FPR.

---

## 1. Experimental Setup

### 1.1 Data and protocol

| Parameter | Value |
|---|---|
| Dataset | CICIDS2017, MachineLearningCVE release (Canadian Institute for Cybersecurity) |
| Rows after cleaning | 2,497,980 from 8 files |
| Feature space | 77 CICFlowMeter flow-level features |
| Classes | 9: BENIGN, Bot, Brute Force, DDoS, DoS, Heartbleed, Infiltration, PortScan, Web Attack |
| Split | Chronological 70/30 inside every (capture file, class) group, ordered by original CSV row position |
| Train / Test | 1,748,579 / 749,401 rows (no calibration part) |
| Imbalance strategy | `class_weight`, identical for every model; resampling only inside train/CV folds |
| Cross-validation | StratifiedKFold, k=5, scored on macro-F1 |
| HP search | RandomizedSearchCV, n_iter=20 on a 200,000-row subset, search space of 144 for every model (13.9% of each space explored) |
| Random state | 42 |
| Hardware | XGBoost, CatBoost and Stacking trained on GPU; the rest on CPU · **latency measured on CPU for every model** |

**Why not a source holdout?** Each attack class lives in exactly one file, so holding out a file would leave its class with no training rows.

**Why row order instead of time?** This release has no Timestamp column. Row order was verified against the published attack schedule.

### 1.2 Test-set class distribution (natural, untouched)

| Class | n_train | n_test | Test share | Statistical confidence |
|---|---:|---:|---:|---|
| BENIGN | 1,450,574 | 621,680 | 82.96% | High |
| DoS | 135,611 | 58,119 | 7.76% | High |
| DDoS | 89,609 | 38,405 | 5.12% | High |
| PortScan | 63,485 | 27,209 | 3.63% | High |
| Brute Force | 6,405 | 2,745 | 0.37% | High |
| Web Attack | 1,500 | 643 | 0.09% | High |
| Bot | 1,363 | 585 | 0.08% | High |
| **Infiltration** | **25** | **11** | 0.001% | **Anecdotal** |
| **Heartbleed** | **7** | **4** | 0.0005% | **Anecdotal** |

> **Reading rule used throughout:** any Heartbleed (n=4) or Infiltration (n=11) figure is an *observation*, not a *measurement*. One misclassified row moves Heartbleed recall by 25 points. The pipeline therefore reports `f1_macro_reportable`, which averages only the 7 classes with n_test ≥ 30, alongside plain macro-F1.

### 1.3 Validity checks

- **Correct baseline:** "always BENIGN" scores **0.8296** accuracy, computed from the test distribution. The previous edition computed it from the oversampled train set and got 0.0100, which was wrong.
- **CV:**
  - 5-fold on the training set, with σ between 0.0135 and 0.0538.
  - CV scores are higher than test scores for every model (e.g. XGBoost 0.9697 vs 0.8773), because stratified CV samples from the same time period as training while the test set comes later.
  - That gap is the price of a more realistic evaluation.
- **Label-shuffle:**
  - **Not run on this run** (`skipped`).
  - The previous run scored macro-F1 0.064–0.114 against a random level of 0.111.
  - It should be re-run here (section 5.2).
- **Tuning fairness:** same method, objective, budget and search-space size for every model.

---

## 2. Measured Results

### 2.1 Headline metrics

| Model | Accuracy | Balanced Acc | **macro-F1** | macro-F1 (reportable) | weighted-F1 | MCC | CV macro-F1 (μ ± σ) |
|---|---:|---:|---:|---:|---:|---:|---|
| **CatBoost** | **0.9953** | **0.8970** | **0.8873** | 0.8613 | **0.9952** | **0.9844** | 0.9179 ± 0.0225 |
| LightGBM | 0.9903 | 0.8576 | 0.8774 | **0.8627** | 0.9898 | 0.9676 | 0.9666 ± 0.0168 |
| XGBoost | 0.9906 | 0.8889 | 0.8773 | 0.8542 | 0.9902 | 0.9687 | 0.9697 ± 0.0155 |
| Random Forest | 0.9820 | 0.8165 | 0.8404 | 0.8378 | 0.9832 | 0.9397 | 0.9163 ± 0.0135 |
| Stacking | 0.9581 | 0.8852 | 0.6822 | 0.7092 | 0.9712 | 0.8757 | 0.7987 ± 0.0279 |
| Logistic Regression | 0.8428 | 0.8338 | 0.5574 | 0.6348 | 0.9030 | 0.6651 | 0.4988 ± 0.0278 |
| MLP | 0.9331 | 0.7112 | 0.5369 | 0.6767 | 0.9443 | 0.7895 | 0.5697 ± 0.0538 |

*"Always BENIGN" baseline: accuracy 0.8296*

### 2.2 Per-class F1

| Class | CatBoost | LightGBM | XGBoost | RF | Stacking | LogReg | MLP | **Best** |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BENIGN | **0.9980** | 0.9949 | 0.9954 | 0.9900 | 0.9757 | 0.9049 | 0.9603 | CatBoost |
| DoS | **0.9769** | 0.9406 | 0.9429 | 0.9103 | 0.9499 | 0.8751 | 0.7881 | CatBoost |
| DDoS | 0.9994 | **0.9995** | 0.9994 | 0.9990 | 0.9988 | 0.9776 | 0.9967 | tie |
| PortScan | 0.9821 | **0.9832** | 0.9791 | 0.9812 | 0.9764 | 0.8710 | 0.8899 | tie |
| Brute Force | 0.9984 | 0.9978 | **0.9993** | 0.9880 | 0.1946 | 0.7175 | 0.7956 | XGBoost |
| Web Attack | 0.8680 | 0.8716 | 0.8218 | **0.9500** | 0.8125 | 0.0948 | 0.2968 | Random Forest |
| **Bot** | 0.2061 | **0.2514** | 0.2413 | 0.0459 | 0.0562 | 0.0023 | 0.0096 | LightGBM (still low) |
| Infiltration ⚠ | 0.9565 | 1.0000 | 0.9167 | 0.8421 | 0.1760 | 0.0015 | 0.0947 | *n=11, anecdotal* |
| Heartbleed ⚠ | 1.0000 | 0.8571 | 1.0000 | 0.8571 | 1.0000 | 0.5714 | 0.0000 | *n=4, anecdotal* |

### 2.3 Attack detection (attack vs BENIGN)

| Model | Detection rate | Precision | FPR | FNR | False alerts | Attacks missed |
|---|---:|---:|---:|---:|---:|---:|
| **CatBoost** | **0.9840** | 0.9968 | 0.00065 | **0.01604** | 404 | **2,049** |
| Logistic Regression | 0.9746 | 0.5418 | 0.16931 | 0.02540 | 105,258 | 3,244 |
| Stacking | 0.9661 | 0.8298 | 0.04072 | 0.03389 | 25,317 | 4,329 |
| XGBoost | 0.9571 | 0.9981 | 0.00038 | 0.04293 | 236 | 5,483 |
| LightGBM | 0.9513 | **0.9989** | **0.00022** | 0.04870 | **136** | 6,220 |
| Random Forest | 0.9277 | 0.9729 | 0.00530 | 0.07228 | 3,298 | 9,232 |
| MLP | 0.8459 | 0.7864 | 0.04721 | 0.15406 | 29,347 | 19,677 |

### 2.4 The false-positive budget

FPR on benign traffic converted to alert volume for an organisation with **10 million benign flows per day**.

| Model | FPR | **False alerts/day @ 10M flows** | Analyst-hours/day @ 2 min/alert |
|---|---:|---:|---:|
| LightGBM | 0.022% | **2,200** | 73 |
| XGBoost | 0.038% | 3,800 | 127 |
| **CatBoost** | 0.065% | 6,500 | 217 |
| Random Forest | 0.53% | 53,000 | 1,767 |
| Stacking | 4.07% | 407,000 | 13,567 |
| MLP | 4.72% | 472,000 | 15,733 |
| Logistic Regression | 16.93% | 1,693,000 | 56,433 |

Even the best model raises thousands of false alerts a day, more than analysts can triage. A flow classifier should therefore be a **scoring stage**, not the alerting stage. Alerts need entity-level aggregation by host or campaign before a human sees them (section 6.2).

### 2.5 Computational efficiency (measured)

Measured on CPU for every model: batches of 1,000 flows, 30 repeats from random offsets, after one discarded warm-up batch.

| Model | Trained on | Fit time | p50 latency | **p95 latency** | Throughput | Model size |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | CPU | 653.7 s | 4.30 ms | **5.15 ms** | 232,391 flows/s | 0.0 MB |
| **CatBoost** | GPU | 44.6 s | 7.58 ms | **8.39 ms** | 131,975 flows/s | 2.5 MB |
| MLP | CPU | 818.7 s | 10.05 ms | 12.50 ms | 99,545 flows/s | 0.7 MB |
| XGBoost | GPU | 154.6 s | 43.42 ms | 54.10 ms | 23,030 flows/s | 15.8 MB |
| Random Forest | CPU | 314.6 s | 85.94 ms | 104.12 ms | 11,636 flows/s | 124.5 MB |
| Stacking | GPU | 2,754.7 s | 119.79 ms | 137.95 ms | 8,348 flows/s | 109.3 MB |
| LightGBM | CPU | 549.6 s | 111.40 ms | 144.02 ms | 8,977 flows/s | 54.9 MB |

> Fit times are not directly comparable, because some models trained on GPU. Latency and throughput are comparable, because all were measured on CPU.

---

## 3. Per-Model Technical Assessment

---

### 3.1 Logistic Regression

**Official docs:** https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
**Primary references:** Cox, D. R. (1958) *JRSS-B* 20(2):215–242 · McFadden, D. (1974) — multinomial/softmax extension · Liu & Nocedal (1989) *Mathematical Programming* 45:503–528 — the `lbfgs` solver

**Algorithm** — softmax over linear scores, `P(y=k|x) = exp(wₖᵀx+bₖ) / Σⱼ exp(wⱼᵀx+bⱼ)`, fitted by minimising L2-penalised cross-entropy. The objective is convex, so there is a single global optimum. **Tuned parameters:** `C=300, tol=1e-4, fit_intercept=True`.

**Strengths**
- The only model in the set with a **unique global optimum**; retraining reproduces it.
- Coefficients read directly as log-odds per standardised feature unit — **the most interpretable model here**.
- **Fastest measured:** p95 5.15 ms per 1,000 flows (232,391 flows/s), and a model file that rounds to 0.0 MB.
- Overall attack recall of 0.9746, with DDoS recall 0.9979 and PortScan 0.9941.

**Weaknesses (measured)**
- Linear boundaries only. It cannot express feature interactions without hand-built terms.
- **FPR 16.93%**, or 105,258 false alerts on the test set — the cost of reaching high recall with a linear boundary.
- macro-F1 0.5574, with very low precision on rare classes (Bot 0.0012, Infiltration 0.0008, Web Attack 0.0499).
- 653.7 s to train, long for a linear model. `C=300` means weak regularisation, so L-BFGS needs many iterations on 1.7M rows.

**Complexity** — train `O(i · n · d · K)`; predict `O(d · K)` = 77 × 9 = 693 multiply-adds per flow.

**Interpretability** — best. Linear SHAP reduces to `φᵢ = wᵢ(xᵢ − E[xᵢ])`: exact and as cheap as a prediction.

**Unscaled features** — very poor on raw data: `Flow Duration` (range ~10⁸) swamps small-range features. `StandardScaler` is mandatory.

**Class imbalance** — `class_weight` lifts recall, but with a linear boundary the price is a flood of false positives.

**SOC role** — not a primary detector. Fitting roles:
- a **transparent baseline** that measures how much non-linearity the problem needs;
- the **meta-learner of the stacking ensemble**, as used in `src/models/stacking.py`;
- an auditable decision-maker where compliance requires one.

---

### 3.2 Random Forest

**Official docs:** https://scikit-learn.org/stable/modules/ensemble.html#random-forests
**Primary references:** **Breiman, L. (2001) "Random Forests" *Machine Learning* 45(1):5–32** · Breiman (1996) Bagging · Ho (1995) Random Subspace · Louppe (2014) PhD thesis analysing the sklearn implementation

**Algorithm** — bagged decision trees with two layers of randomness: each tree trains on a bootstrap sample, and each split considers a random feature subset (√77 ≈ 9). Averaging high-variance, decorrelated trees lowers variance without raising bias. **Tuned parameters:** `n_estimators=300, max_depth=30, min_samples_split=2, min_samples_leaf=1`.

**Strengths**
- **Best Web Attack F1 in the study (0.9500)**, with Web Attack precision of 0.9866.
- **Tightest CV in the study (σ = 0.0135).**
- Scale-invariant; consumes raw CICFlowMeter output directly.
- Embarrassingly parallel, with a free out-of-bag estimate of generalisation error.

**Weaknesses (measured)**
- **Largest model file in the study (124.5 MB)**: 300 trees up to depth 30 on 1.7M rows.
- **p95 latency 104.12 ms per 1,000 flows**, 12× slower than CatBoost, because walking 300 trees is pointer-chasing with poor cache locality.
- macro-F1 0.8404, fourth. It misses 9,232 attack flows, second-worst, with DoS recall of only 0.8372.
- Bot F1 of only 0.0459, clearly behind the boosting models.
- Cannot extrapolate beyond the training range. Impurity importance is biased toward high-cardinality features (Strobl et al. 2007), so prefer SHAP.

**Complexity** — train `O(T · √d · n log²n)`; predict `O(T · depth)` = 300 trees × up to 30 comparisons per flow, plus many cache misses.

**Memory and scalability** — deployment memory is the binding constraint. Mitigate by capping `max_depth`, setting `min_samples_leaf ≥ 5`, or compiling with Treelite/ONNX Runtime.

**Hyperparameter sensitivity** — lowest among the tree models; defaults are good and more trees is always safe.

**Interpretability** — good. Exact TreeSHAP in `O(T · L · D²)` (Lundberg et al. 2020).

**Noise robustness** — good. Bagging averages out label noise, which matters on CICIDS2017 because labels follow *attack time windows* (Engelen et al. 2021).

**SOC role** — strong for batch scoring, but its size and latency rule out an inline path unless compiled. Fitting role: **a Web Attack specialist**, where it has both the highest F1 and the highest precision.

---

### 3.3 XGBoost

**Official docs:** https://xgboost.readthedocs.io/
**Primary references:** **Chen, T. & Guestrin, C. (2016) "XGBoost: A Scalable Tree Boosting System" *KDD '16* 785–794** · Friedman (2001) *Annals of Statistics* 29(5):1189–1232 — gradient boosting foundations · Friedman (2002) — stochastic subsampling

**Algorithm** — additive boosting on a **second-order Taylor expansion** of the loss:
- each new tree fits the gradient and Hessian of the current residual;
- optimal leaf weights have the closed form `w* = −G/(H+λ)`;
- split gain is `½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)] − γ`, with regularisation written into the objective.

**Tuned parameters:** `n_estimators=800, max_depth=10, learning_rate=0.05, subsample=0.8`, `tree_method='hist'` (GPU).

**Strengths**
- **Best Brute Force F1 (0.9993)**, and a macro-F1 of 0.8773, virtually equal to LightGBM.
- **Second-lowest FPR (0.038%)**: 236 false alerts on the test set.
- **Highest CV mean (0.9697)** on the training set.
- Explicit regularisation (`λ`, `γ`, `α`) plus row/column subsampling.
- Native sparsity-aware split finding. This matters here because `Flow Bytes/s` and `Flow Packets/s` produce Inf/NaN on zero-duration flows.
- Production-ready ecosystem: ONNX export, Treelite, GPU inference, JVM/C++ bindings.

**Weaknesses (measured)**
- **Misses 5,483 attack flows**, 2.7× more than CatBoost; DoS recall 0.8937 versus 0.9571.
- **p95 latency 54.10 ms per 1,000 flows**, 6.5× slower than CatBoost. Multiclass builds `n_estimators × K` = 800 × 9 = 7,200 trees up to depth 10.
- 15.8 MB model file, 6× larger than CatBoost.
- Many interacting hyperparameters; a wide tuning surface.

**Complexity** — train (`hist`): `O(T · K · (n · d + b · d))`, b = histogram bins; predict `O(T · K · depth)` = 800 × 9 × 10 = 72,000 comparisons per flow.

**Hyperparameter sensitivity** — high. The key interaction is `max_depth` × `learning_rate` × `n_estimators`. It overfits easily if depth rises without lowering the learning rate.

**Interpretability** — excellent. TreeSHAP was co-developed with XGBoost (Lundberg et al. 2020). It provides:
- gain/cover/weight importance;
- SHAP interaction values;
- per-prediction attribution.

**Noise robustness** — scale-invariant with native NaN/Inf handling. Less tolerant of label noise than Random Forest, because boosting up-weights misclassified rows; mitigated by `subsample < 1.0`.

**SOC role** — a good choice when **a low FPR matters more than missed attacks** and a compile toolchain (Treelite/ONNX) is available to cut latency. Second choice after CatBoost.

---

### 3.4 LightGBM

**Official docs:** https://lightgbm.readthedocs.io/
**Primary references:** **Ke, G. et al. (2017) "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" *NeurIPS 30* 3146–3154** · Fisher (1958) *JASA* 53(284):789–798 — optimal categorical grouping

**Algorithm** — histogram GBDT with three signature mechanisms:
1. **Leaf-wise growth:** always split the leaf with the largest loss reduction.
2. **GOSS:** keep all high-gradient rows, sample low-gradient rows with reweighting.
3. **EFB:** bundle mutually exclusive sparse features.

**Tuned parameters:** `n_estimators=1000, num_leaves=127, min_child_samples=1, learning_rate=0.1`.

**Strengths**
- **Lowest FPR in the study (0.022%)**: only 136 false alerts, and the highest binary precision, 0.9989.
- **Highest `f1_macro_reportable` (0.8627)**, 0.0014 ahead of CatBoost.
- Best DDoS and PortScan F1 (0.9995, 0.9832), and the best Bot F1 of all models (0.2514), though still low.
- Histogram binning gives `O(n)` splits and low training memory.

**Weaknesses (measured)**
- **Slowest p95 latency in the study (144.02 ms per 1,000 flows)**: 1,000 rounds × 9 classes × up to 127 leaves.
- 54.9 MB model file. It misses 6,220 attack flows, 3× more than CatBoost, with DoS recall 0.8891.
- **`min_child_samples=1` remains a risk.**
  - In the old run this value, combined with Infiltration oversampling, made the model memorise duplicated rows and collapse (accuracy 0.7634).
  - This run uses `class_weight` instead of oversampling, so it does not happen.
  - If oversampling comes back, `min_child_samples` must be raised (section 5.2).

**Complexity** — train `O(T · K · n' · d')` with n' < n from GOSS and d' < d from EFB; predict `O(T · K · depth)`, but leaf-wise trees are deep and asymmetric.

**Hyperparameter sensitivity** — highest among the boosting libraries. The dangerous pair is `num_leaves` × `min_child_samples`. The LightGBM docs advise `num_leaves < 2^max_depth` and `min_child_samples` in the hundreds at this data size.

**Interpretability** — exact TreeSHAP, but deep asymmetric leaf-wise trees make inspecting single trees of little use.

**Noise robustness** — scale-invariant and NaN-tolerant. Leaf-wise growth chases high-loss regions, which is where mislabelled flows sit, so it tolerates label noise less well than its peers.

**SOC role** — the best choice when **false-alert volume is the binding constraint** and the path does not need low latency, e.g. batch scoring or a secondary review queue.

---

### 3.5 CatBoost — ⭐ Recommended

**Official docs:** https://catboost.ai/docs/
**Primary references:** **Prokhorenkova, L. et al. (2018) "CatBoost: unbiased boosting with categorical features" *NeurIPS 31* 6638–6648** · Dorogush, Ershov & Gulin (2018) arXiv:1810.11363

**Algorithm** — gradient boosting with two signature mechanisms:
1. **Ordered boosting:** each row's residual comes from a model trained only on rows earlier in a permutation, which removes the *prediction shift* that biases ordinary boosting.
2. **Oblivious (symmetric) trees:** every node at a depth shares one split condition. A depth-6 tree is just six (feature, threshold) pairs, and prediction becomes a 6-bit index into a 64-entry table.

**Tuned parameters:** `iterations=500, depth=6, learning_rate=0.15, l2_leaf_reg=1` (GPU).

**Strengths (measured)**
- **Wins all three rankings:**
  - highest macro-F1 (0.8873);
  - highest recall_macro under FPR ≤ 1% (0.8970);
  - lowest p95 latency among models near the top macro-F1 (8.39 ms).
- **Fewest missed attacks (2,049 flows, FNR 1.60%)** and the highest detection rate, 0.9840.
- **Best DoS F1 (0.9769) and DoS recall of 0.9571**, against 0.84–0.89 for the other trees. This is the main reason it misses fewer attacks, since DoS is the largest attack class in the test set.
- **Highest accuracy, balanced accuracy, weighted-F1 and MCC** (0.9953, 0.8970, 0.9952, 0.9844).
- **Fastest tree model at inference:** p95 8.39 ms per 1,000 flows (131,975 flows/s). Oblivious trees turn tree-walking into branch-free, vectorisable index arithmetic.
- **Smallest tree model (2.5 MB)**, 50× smaller than Random Forest.
- Fastest measured fit (44.6 s on GPU).

**Weaknesses**
- FPR 0.065% is 3× LightGBM's and 1.7× XGBoost's (404 false alerts vs 136 and 236).
- CV σ = 0.0225, wider than the other tree models.
- `f1_macro_reportable` (0.8613) trails LightGBM slightly, so its predictive edge rests on DoS and the rare classes.
- Oblivious trees express less per tree; more trees are needed to match free-form structure.
- Its categorical-feature strength goes unused, because all 77 features are numeric.

**Complexity** — train `O(T · K · n · d)` with a permutation constant; predict `O(T · depth)` = 500 × 6 = 3,000 branch-free, vectorisable operations.

**Hyperparameter sensitivity** — lowest of the three boosting libraries; ordered boosting keeps overfitting risk low.

**Interpretability** — very good. TreeSHAP works, and a depth-6 oblivious tree is six rules an analyst can read.
- SHAP results for this model are in the final report, §4.1.4, and in `results/cicids2017_temporal_v1_shap/`.
- The top features are Destination Port, Init_Win_bytes_forward and Init_Win_bytes_backward.

**Noise robustness** — scale-invariant. Ordered boosting gives principled protection against prediction-shift bias from label noise.

**Class imbalance** — strong: the highest balanced accuracy (0.8970) with the same `class_weight` as every other model.

**SOC role** — **the primary inline detector**: low latency, a small file, no compilation, and the fewest missed attacks. The one trade-off is slightly more false alerts than LightGBM/XGBoost, which alert aggregation handles (section 6.2).

---

### 3.6 Multi-Layer Perceptron (MLP)

**Official docs:** https://scikit-learn.org/stable/modules/neural_networks_supervised.html
**Primary references:** **Rumelhart, Hinton & Williams (1986) *Nature* 323:533–536** · Cybenko (1989) and Hornik (1991) — universal approximation · Kingma & Ba (2015) *ICLR* — Adam · **Grinsztajn, Oyallon & Varoquaux (2022) *NeurIPS D&B*** — why trees still beat neural networks on tabular data

**Algorithm** — fully connected feed-forward network with ReLU and softmax, trained by backpropagation with Adam. **Tuned parameters:** `hidden_layer_sizes=(256, 128, 64, 32), learning_rate_init=0.0005, batch_size=256, alpha=0.0001`, i.e. 77 → 256 → 128 → 64 → 32 → 9.

**Strengths**
- A universal function approximator; it can draw smooth oblique boundaries that axis-aligned trees approximate as staircases.
- **Small model file (0.7 MB)** and **constant latency**: p95 12.50 ms per 1,000 flows, because prediction is a fixed sequence of matrix multiplications.
- True incremental learning via `partial_fit`, the only model here that can update online.

**Weaknesses (measured)**
- **Lowest macro-F1 (0.5369)** and lowest balanced accuracy (0.7112).
- **Most missed attacks (19,677 flows, FNR 15.4%)**; DoS recall only 0.6583, Heartbleed recall 0.
- Highest CV variance (σ = 0.0538), because the objective is non-convex.
- **sklearn's `MLPClassifier` has no `class_weight` parameter**, so the standard imbalance fix is unavailable — the structural reason the rare classes suffer.
- Requires strict feature scaling. It is the least interpretable model and needs approximate KernelSHAP.
- Grinsztajn et al. (2022) attribute this to a rotationally invariant inductive bias that does not suit tabular data, where each feature carries its own meaning.

**Complexity** — constant FLOPs per flow, no data-dependent branches; training is linear in epochs and samples (818.7 s on CPU).

**SOC role** — not recommended as a primary detector. Fitting role: an **online drift monitor** via `partial_fit` running alongside a frozen tree model. A neural NIDS should use sequence models over packet sequences, not an MLP over aggregated flow statistics.

---

### 3.7 Stacking Ensemble

**Official docs:** https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization
**Primary references:** **Wolpert, D. H. (1992) "Stacked Generalization" *Neural Networks* 5(2):241–259** · Breiman (1996) Stacked Regressions · Ting & Witten (1999) *JAIR* 10:271–289 · van der Laan et al. (2007) Super Learner

**Architecture as implemented** (`src/models/stacking.py`):

```
StandardScaler
   └─> StackingClassifier (cv=5, stack_method='predict_proba', passthrough=False)
         ├─ LightGBM      (300 trees, 63 leaves, lr 0.05, class_weight='balanced')
         ├─ XGBoost       (200 trees, depth 8, lr 0.10, hist, per-class weights)
         ├─ RandomForest  (200 trees, class_weight='balanced')
         └─ meta: LogisticRegression (lbfgs, C=1.0, class_weight='balanced')
```

**Algorithm** — two-level learning:
- **Level 0:** three base learners produce out-of-fold `predict_proba` via 5-fold internal CV.
- **Level 1:** a balanced logistic regression learns a linear blend of the 27 meta-features (3 models × 9 classes).

Super Learner theory guarantees a stack is no worse than its best base learner only *asymptotically*. It gives no guarantee on finite data or under test-set distribution shift.

**Strengths**
- Overall detection rate of 0.9661, third-best, with Heartbleed/Infiltration recall of 1.0 (on n=4 and 11).
- Second-best DoS F1 (0.9499).
- The logistic meta-learner is interpretable: its coefficients show *which base model is trusted for which class*.

**Weaknesses (measured)**
- **macro-F1 of 0.6822, below all three of its base learners measured alone** (LightGBM 0.8774, XGBoost 0.8773, RF 0.8404). The stack's base learners use smaller default settings than the tuned standalone models (`best_params: {}`, no ensemble-level tuning).
- **Brute Force F1 of only 0.1946 (precision 0.1078)**: the meta-learner predicts many BENIGN flows as Brute Force (Brute Force class FPR 3.03%). Every base learner separates Brute Force almost perfectly, so the fault is in the meta level, not the base level.
- **FPR 4.07% (25,317 false alerts)**, 63× CatBoost's.
- **Longest training (2,754.7 s on GPU)**, because each base learner is fit 6 times (5 folds + refit).
- p95 latency 137.95 ms and 109.3 MB, because every prediction runs all three base models then the meta-model.
- Hardest to operate: four models to version and monitor.

**Complexity** — train `O((cv + 1) · Σᵢ Trainᵢ) + O(meta)` ≈ 6× the base-learner total; predict `O(Σᵢ Inferᵢ + K·M)`, the *sum* of RF, XGB and LGBM.

**Interpretability** — moderate. The meta level is readable, but attributing to raw features needs SHAP composed across two levels.

**SOC role** — **not recommended in its current form.** Fix the meta level first, then re-evaluate (section 5.2). Options:
- use the tuned base learners;
- enable `passthrough`;
- tune the Brute Force threshold.

---

## 4. Comparison Tables and Trade-Offs

### 4.1 Master comparison

| Model | Key strength | Key limitation | p95 latency (per 1,000 flows) | Model size | Attacks missed | Interpretability | Overall |
|---|---|---|---:|---:|---:|---|---|
| **CatBoost** | Wins all 3 rankings; fewest missed attacks | FPR slightly above LGBM/XGB | **8.39 ms** | **2.5 MB** | **2,049** | Excellent (TreeSHAP + readable trees) | ⭐ **Deploy — primary inline** |
| **LightGBM** | Lowest FPR (0.022%); best reportable macro-F1 | Slowest; 3× CatBoost's missed attacks | 144.02 ms | 54.9 MB | 6,220 | Excellent (TreeSHAP) | ✅ **Deploy — low-FP queue / batch** |
| **XGBoost** | Best Brute Force F1; second-lowest FPR | 6.5× slower than CatBoost | 54.10 ms | 15.8 MB | 5,483 | Excellent (TreeSHAP) | ✅ **Alternative — with a compile toolchain** |
| **Random Forest** | Best Web Attack F1; tightest CV | Largest file; low DoS recall | 104.12 ms | 124.5 MB | 9,232 | Good (TreeSHAP) | ⚠ **Web Attack specialist** |
| **Stacking** | Detection rate 0.966 | Brute Force precision 0.108; longest training | 137.95 ms | 109.3 MB | 4,329 | Moderate (two-level SHAP) | ❌ **Fix the meta level first** |
| **Logistic Regression** | Fastest; fully interpretable | FPR 16.9% | **5.15 ms** | **0.0 MB** | 3,244 | **Best** (closed-form SHAP) | ⚠ **Baseline + stack meta-learner** |
| **MLP** | Constant latency; `partial_fit` | Lowest macro-F1; no `class_weight` | 12.50 ms | 0.7 MB | 19,677 | Weak (KernelSHAP) | ⚠ **Drift monitor / research** |

### 4.2 Rank by dimension (1 = best)

| Dimension | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| macro-F1 | CatBoost | LGBM | XGB | RF | Stack | LogReg | MLP |
| macro-F1 (reportable) | LGBM | CatBoost | XGB | RF | Stack | MLP | LogReg |
| Balanced accuracy | CatBoost | XGB | Stack | LGBM | LogReg | RF | MLP |
| Attacks missed (fewer = better) | CatBoost | LogReg | Stack | XGB | LGBM | RF | MLP |
| False-positive burden (lower = better) | LGBM | XGB | CatBoost | RF | Stack | MLP | LogReg |
| p95 latency | LogReg | CatBoost | MLP | XGB | RF | Stack | LGBM |
| Model size | LogReg | MLP | CatBoost | XGB | LGBM | Stack | RF |
| Reproducibility (CV σ) | RF | XGB | LGBM | CatBoost | LogReg | Stack | MLP |
| Interpretability | LogReg | CatBoost | XGB | RF | LGBM | Stack | MLP |

### 4.3 Per-attack-type assessment

#### High-volume attacks — DoS, DDoS, PortScan

| Class | n_test | Best | F1 | Analysis |
|---|---:|---|---:|---|
| **DDoS** | 38,405 | tie (LGBM/CatBoost/XGB) | 0.9994–0.9995 | Solved; every model except LogReg exceeds 0.996 |
| **DoS** | 58,119 | **CatBoost** | **0.9769** | The decisive class: CatBoost recall 0.9571 vs 0.84–0.89 for the other trees. DoS is the largest attack class, so this gap explains almost all of the difference in missed attacks. |
| **PortScan** | 27,209 | tie (LGBM/CatBoost/RF/XGB) | 0.979–0.983 | Trees are close; LogReg and MLP drop to 0.87–0.89 |

> **Verdict — high-volume: CatBoost**, winning DoS clearly and tying the other two.

#### Rare attacks — Brute Force, Web Attack, Bot, Infiltration, Heartbleed

| Class | n_test | Best | Score | Analysis |
|---|---:|---|---:|---|
| **Brute Force** | 2,745 | XGBoost | F1 0.9993 | Boosting models exceed 0.997; Stacking drops to 0.1946 from meta-level false positives |
| **Web Attack** | 643 | **Random Forest** | F1 0.9500 | RF leads clearly (precision 0.9866); boosting 0.82–0.87 |
| **Bot** | 585 | LightGBM | F1 0.2514 | **Every model fails**, recall 0.15–0.16 — see below |
| **Infiltration** ⚠ | 11 | — | — | LightGBM 1.0000, CatBoost 0.9565 · **n=11 cannot decide anything** |
| **Heartbleed** ⚠ | 4 | — | — | CatBoost/XGB/Stacking = 1.0000 · **n=4 is coincidence** |

> **Why Bot defeats every model:**
> - All Bot traffic sits in the single Friday-Morning file. The chronological split sends the first 70% to train and the last 30% to test.
> - Across all 1,948 Bot rows, the late part differs clearly from the early part:
>   - the share using port 8080 rises from about 52% to 92%;
>   - median Flow Duration rises from about 59,000 to 1,000,000 µs.
> - The models therefore face Bot behaviour they have not seen. The effect disappears under a random split (the old run scored Bot F1 ≈ 0.99), a clear example of how random splits overstate performance.

---

## 5. Findings and Open Work

### 5.1 ✅ Issues from the previous edition, now fixed

| Issue in the previous edition | Status in this run |
|---|---|
| 5.1 baseline computed from the oversampled train set (0.0100) | ✅ computed from the test set (0.8296) |
| 5.2 tiny classes destabilised the ranking | ✅ `f1_macro_reportable` (n_test ≥ 30) reported alongside; full data gives Web Attack 643 test rows instead of 13 |
| 5.3 inconsistent champion rule | ✅ three pre-declared rankings (Overall, Security, Deployment), with every exclusion logged |
| 5.4 LightGBM collapse from the search space | ✅ does not occur here, because `class_weight` replaces oversampling (residual risk in 5.2) |
| 5.5 latency and memory never measured | ✅ measured on CPU for every model (section 2.5) |
| Limitation: subsample and random split | ✅ full data and chronological split |
| Checklist #8: validate against CSE-CIC-IDS2018 | ✅ done in the final report's supplementary experiment (section 5.3 here) |

### 5.2 🟡 Open work

1. **Bot under a chronological split.**
   - Every model's recall is about 0.16.
   - Before claiming Bot detection, this needs either Bot data from other periods or sources, or features that capture C2 behaviour more stably.
2. **Run label-shuffle on this run.** It was skipped here and should be run to confirm no pipeline leakage under the new split.
3. **Constrain LightGBM's search space.**
   - The selected `min_child_samples=1` would again memorise duplicated rows if oversampling returns.
   - Floor `min_child_samples` in the hundreds and keep `num_leaves < 2^max_depth`.
4. **Fix the Stacking meta level.** Use the tuned base learners, consider `passthrough=True`, and check Brute Force false positives before re-evaluating.
5. **Run evasion tests on all seven models.** Attackers control flow-level features cheaply through padding, delay and flow splitting.
6. **Compile CatBoost/XGBoost with ONNX or Treelite** and verify prediction parity before any inline deployment.

### 5.3 Cross-dataset transfer (CSE-CIC-IDS2018)

The final report's supplementary experiment retrained all seven models on a shared 7-class schema and tested across CICIDS2017 and CSE-CIC-IDS2018.
- **Transfer is limited:** the best macro-F1 is 0.34 (2017→2018) and 0.44 (2018→2017), versus 0.86–0.88 within a single dataset.
- **Each dataset is identifiable:** a classifier tells which dataset a record came from with AUC ≥ 0.999.
- **What this means here:** the figures in this document describe performance *within CICIDS2017 only*. They do not guarantee performance on another network.

Details are in `docs/experiments/crossdataset/round6_split_70_30/report/paper_summary_pack.md`.

---

## 6. Final SOC Deployment Recommendation

### 6.1 Decision: CatBoost

| Criterion | CatBoost | Evidence |
|---|---|---|
| **Predictive quality** | **#1** | Highest macro-F1 0.8873, balanced accuracy 0.8970, MCC 0.9844. On reportable macro-F1 it trails LightGBM by 0.0014, a tie at the top. |
| **Attacks missed** | **#1** | 2,049 flows (FNR 1.60%), 2.7× fewer than XGBoost and 3× fewer than LightGBM |
| **Latency / size** | **#1 among trees** | p95 8.39 ms per 1,000 flows, 2.5 MB. Only LogReg is faster, at FPR 16.9%. |
| **False-positive burden** | #3 | FPR 0.065%, or 6,500 false alerts per 10M benign flows/day, vs 2,200 for LightGBM and 3,800 for XGBoost |

**The honest trade-off:**
- **What CatBoost costs:** about 4,300 more false alerts than LightGBM per 10 million benign flows.
- **What it buys:** about 4,200 fewer missed attacks on this test set, and 17× lower latency.
- **Why that is the right side:** in security, one missed intrusion costs more than one false alert. False alerts can also be reduced by aggregation (stage 3 below), while a missed intrusion cannot be recovered.

### 6.2 Recommended production architecture

```
                    Flow export (CICFlowMeter / Zeek / nProbe)
                                    │
              ┌─────────────────────▼─────────────────────┐
              │  STAGE 1 — INLINE          CatBoost        │
              │  p95 8.39 ms / 1,000 flows, 2.5 MB         │
              │  no compilation needed                     │
              └─────────────────────┬─────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │ p(attack) low             │ uncertain                 │ p(attack) high
        ▼                           ▼                           ▼
      Pass                    ┌───────────────┐             Alert
    (log only)                │  STAGE 2      │           (attach SHAP
                              │  LightGBM     │            attribution)
                              │  async queue  │
                              │  FPR 0.022%   │
                              └───────┬───────┘
                                      ▼
                            ┌─────────────────────┐
                            │ STAGE 3 — SOC queue │
                            │ risk-ranked,        │
                            │ grouped by host     │
                            └─────────────────────┘

   In parallel: Random Forest re-checks flows predicted as Web Attack (F1 0.95)
   In parallel: MLP partial_fit drift monitor — divergence from stage 1 flags distribution shift
```

**Why this works:**
- Stage 1 handles most flows with low latency and the fewest missed attacks.
- Stage 2 uses the lowest-FPR model to re-check only the uncertain band, where 144 ms per 1,000 flows does not matter because it is off the main path.
- Stage 3 is **required, not optional**. Even the best model raises thousands of false alerts a day. A flow classifier is the scoring stage; alerting must happen at the entity level.

### 6.3 Role assignment summary

| Role | Model | Justification |
|---|---|---|
| **Primary inline detector** | **CatBoost** | Wins all 3 rankings; fewest missed attacks; fastest and smallest tree model |
| **False-positive reduction re-check** | **LightGBM** | Lowest FPR (0.022%); best reportable macro-F1 |
| **Alternative with a compile toolchain** | **XGBoost** | Second-lowest FPR; best Brute Force F1; ONNX/Treelite ecosystem |
| **Web Attack specialist** | **Random Forest** | Web Attack F1 0.95, precision 0.99 |
| **Drift monitor** | **MLP** | Online adaptation via `partial_fit` |
| **Baseline and meta-learner** | **Logistic Regression** | Fully explainable; already the stack's meta-model |
| **Hold until fixed** | **Stacking** | Brute Force precision 0.108, FPR 4.07% · fix the meta level per section 5.2 first |

### 6.4 Pre-production checklist

| # | Item | Status |
|---|---|---|
| 1 | Baseline from the test distribution | ✅ done |
| 2 | `f1_macro_reportable` and the three-ranking champion rule | ✅ done |
| 3 | Measure latency/size for all 7 models on CPU | ✅ done |
| 4 | Validate against CSE-CIC-IDS2018 | ✅ done (transfer is limited) |
| 5 | Run label-shuffle on this run (5.2 item 2) | ⬜ required |
| 6 | Address Bot under the chronological split (5.2 item 1) | ⬜ required |
| 7 | Evasion tests on all 7 models (5.2 item 5) | ⬜ required |
| 8 | Compile CatBoost with ONNX and verify prediction parity | ⬜ recommended |
| 9 | Constrain LightGBM's search space (5.2 item 3) | ⬜ recommended |
| 10 | Build entity-level alert aggregation (stage 3) before any SOC pilot | ⬜ **blocking** |

---

## 7. Limitations of This Evaluation

1. **One split and one random seed (42).** 5-fold CV shows variance on the training set, not the variance of the chronological split.
2. **Heartbleed (n=4) and Infiltration (n=11) support no conclusion.** They are reported for completeness, and `f1_macro_reportable` excludes them.
3. **The chronological split uses file row order, not a Timestamp.** The MachineLearningCVE release has no time column. Row order is flow *completion* order, not start order, so long-lived attacks such as DoS GoldenEye can trail past their attack window.
4. **CICIDS2017 has documented label defects.** Engelen, Rimmer & Joosen (2021) found errors in both labelling and CICFlowMeter feature extraction, and Rosay et al. (2022) confirmed several. Read the absolute numbers as an upper bound.
5. **Within-dataset results do not guarantee cross-network performance.** The cross-dataset experiment shows a large drop (section 5.3).
6. **Fit times are not comparable across models**, because some trained on GPU and some on CPU.

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
| CSE-CIC-IDS2018 dataset | https://www.unb.ca/cic/datasets/ids-2018.html |

---

*Generated from measured results in `results/cicids2017_temporal_v1/` (the run used in the final report). Thai-language version: [`model_comparison_nids_th.md`](./model_comparison_nids_th.md).*
