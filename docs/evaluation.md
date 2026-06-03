# Evaluation

How models are scored and compared.

## Metrics

| Metric | Why it's in the table |
|--------|-----------------------|
| Accuracy | Familiar baseline — but misleading on imbalanced data |
| Precision (weighted) | Average precision weighted by class support |
| Recall (weighted) | Average recall weighted by class support |
| F1 (weighted) | **Primary metric** — balances P and R, weighted by support |
| F1 (macro) | **Fairness metric** — equal weight per class, exposes minority-class failure |
| ROC-AUC (One-vs-Rest) | Threshold-independent quality of probability scores |
| Matthews Correlation Coefficient | Robust to class imbalance; -1 to 1 |
| Per-class precision/recall/F1 | Lives in the classification report |

## Why F1-weighted is primary, not Accuracy

CICIDS2017 BENIGN traffic outnumbers attacks roughly 30:1. A model that
always predicts BENIGN gets ~97% accuracy and detects 0% of attacks.
F1-weighted penalizes the missed attacks; F1-macro penalizes them even
harder.

In security operations the asymmetric cost is real — a missed attack is
worse than a false alarm. Recall (especially per-attack-class recall) is
the metric we eyeball, F1-weighted is the metric we rank by.

## Confusion matrix conventions

- Rows = true class, columns = predicted class
- Normalized by row (i.e. recall per class) for the "interpretation" view
- Raw counts for the "diagnosis" view
- Both versions saved as `cm_<model>_raw.png` and `cm_<model>_norm.png`

## Comparison report

`results/metrics/comparison.csv` shape::

    model               | variant   | accuracy | precision_w | recall_w | f1_w | f1_macro | roc_auc | mcc | train_time_s
    logistic_regression | baseline  | ...
    logistic_regression | tuned     | ...
    random_forest       | baseline  | ...
    random_forest       | tuned     | ...
    mlp                 | baseline  | ...
    mlp                 | tuned     | ...

The narrative `reports/comparison.md` is generated alongside the CSV and
embeds the headline plot.
