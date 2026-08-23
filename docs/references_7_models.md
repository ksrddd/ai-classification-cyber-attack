# บรรณานุกรม 7 โมเดล — Bibliography for the 7-Model Benchmark

**ที่มา:** สกัดจาก `docs/model_comparison_nids.md` (รายงานฉบับเต็ม) · **โครงงาน:** CICIDS2017 / CSE-CIC-IDS2018 NIDS

เอกสารนี้รวมเฉพาะแหล่งอ้างอิงไว้ที่เดียว สำหรับใช้อ่านและใส่บรรณานุกรมของเล่มรายงาน

ส่วนที่ 1 คืออ้างอิงหลักแยกตามโมเดลทั้ง 7 ตัว ส่วนที่ 2 คือรายการรวมทั้งหมด 35 รายการพร้อมหมายเลขสำหรับใช้อ้างในเนื้อความ ปิดท้ายด้วยตาราง URL เอกสารทางการ

---

## ส่วนที่ 1 · อ้างอิงหลักแยกตามโมเดล (Per-model primary references)

### 1. Logistic Regression

**Official documentation:** https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
**Primary references:**
- Cox, D. R. (1958). "The Regression Analysis of Binary Sequences." *JRSS Series B*, 20(2), 215–242.
- McFadden, D. (1974). "Conditional Logit Analysis of Qualitative Choice Behavior." In *Frontiers in Econometrics*. — multinomial/softmax extension.
- Liu, D. C. & Nocedal, J. (1989). "On the Limited Memory BFGS Method for Large Scale Optimization." *Mathematical Programming*, 45, 503–528. — the `lbfgs` solver.
- Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in Python." *JMLR*, 12, 2825–2830.

### 2. Random Forest

**Official documentation:** https://scikit-learn.org/stable/modules/ensemble.html#random-forests
**Primary references:**
- Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5–32. DOI: 10.1023/A:1010933404324.
- Breiman, L. (1996). "Bagging Predictors." *Machine Learning*, 24(2), 123–140.
- Ho, T. K. (1995). "Random Decision Forests." *ICDAR '95*, 278–282. — the random subspace method.
- Louppe, G. (2014). *Understanding Random Forests: From Theory to Practice.* PhD thesis, Univ. of Liège. — the definitive analysis of the sklearn implementation.

### 3. XGBoost

**Official documentation:** https://xgboost.readthedocs.io/
**Primary references:**
- Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *KDD '16*, 785–794. DOI: 10.1145/2939672.2939785.
- Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine." *Annals of Statistics*, 29(5), 1189–1232. — the gradient boosting foundation.
- Friedman, J. H. (2002). "Stochastic Gradient Boosting." *Computational Statistics & Data Analysis*, 38(4), 367–378. — subsampling.

### 4. LightGBM

**Official documentation:** https://lightgbm.readthedocs.io/
**Primary references:**
- Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *NeurIPS 30*, 3146–3154.
- Fisher, W. D. (1958). "On Grouping for Maximum Homogeneity." *JASA*, 53(284), 789–798. — the optimal categorical split used by LightGBM.

### 5. CatBoost

**Official documentation:** https://catboost.ai/docs/
**Primary references:**
- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., Gulin, A. (2018). "CatBoost: unbiased boosting with categorical features." *NeurIPS 31*, 6638–6648.
- Dorogush, A. V., Ershov, V., Gulin, A. (2018). "CatBoost: gradient boosting with categorical features support." arXiv:1810.11363.

### 6. Multi-Layer Perceptron (MLP)

**Official documentation:** https://scikit-learn.org/stable/modules/neural_networks_supervised.html
**Primary references:**
- Rumelhart, D. E., Hinton, G. E., Williams, R. J. (1986). "Learning representations by back-propagating errors." *Nature*, 323, 533–536.
- Cybenko, G. (1989). "Approximation by superpositions of a sigmoidal function." *Mathematics of Control, Signals and Systems*, 2(4), 303–314. — universal approximation.
- Hornik, K. (1991). "Approximation capabilities of multilayer feedforward networks." *Neural Networks*, 4(2), 251–257.
- Kingma, D. P. & Ba, J. (2015). "Adam: A Method for Stochastic Optimization." *ICLR 2015*. arXiv:1412.6980.
- **Grinsztajn, L., Oyallon, E., Varoquaux, G. (2022). "Why do tree-based models still outperform deep learning on tabular data?" *NeurIPS 2022 Datasets & Benchmarks*.** — directly explains the result observed here.

### 7. Stacking Ensemble

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

---

## ส่วนที่ 2 · รายการรวม (Consolidated bibliography)

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

