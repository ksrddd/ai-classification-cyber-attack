# Evaluation

How models are scored and compared.

## Metric set

| Metric | Why it's in the table |
|--------|-----------------------|
| Accuracy | Familiar baseline, but misleading on imbalanced data |
| Precision (weighted) | Average precision weighted by class support |
| Recall (weighted) | Average recall weighted by class support |
| F1 (weighted) | **Primary ranking metric** -- balances P and R, weighted by support |
| Precision (macro) | Equal weight per class -- exposes minority-class failure |
| Recall (macro) | Equal weight per class -- exposes minority-class failure |
| F1 (macro) | **Fairness metric** -- equal weight per class |
| ROC-AUC | Binary classification -- AUC of the positive class. Multi-class -- One-vs-Rest, weighted |
| Matthews Correlation Coefficient | Robust to class imbalance; -1 to 1 |
| Per-class precision/recall/F1 | Lives in the classification report (one row per class) |

## Why F1-weighted is primary, not Accuracy

CICIDS2017 BENIGN traffic outnumbers attacks roughly **30:1**. A model
that always predicts BENIGN gets ~97% accuracy and detects 0% of
attacks. F1-weighted penalizes missed attacks; F1-macro penalizes them
even harder.

In security operations the asymmetric cost is real -- a missed attack
is worse than a false alarm. Recall (especially per-attack-class
recall) is the metric we eyeball; F1-weighted is the metric we rank by.

## Confusion matrix conventions

- Rows = true class, columns = predicted class.
- Normalized by row (i.e. recall per class) for the
  "interpretation" view -- this is what `--stage evaluate` writes by
  default to `results/figures/confusion_matrix_<model>.png`.
- Raw counts can be computed by passing `normalize=None` to
  `plot_confusion_matrix`.

## Comparison report

`--stage evaluate` writes three artefacts:

- `results/metrics/model_comparison.csv` -- one row per model with the
  full metric set.
- `results/metrics/model_comparison.png` -- side-by-side bar chart of
  accuracy, F1-weighted, F1-macro, ROC-AUC.
- `reports/model_comparison.md` -- the narrative version with the
  ranking table at the top and per-class breakdown for each model
  underneath.

## Per-class breakdown

For each model the `per_class` block of `results/metrics/<model>_test.json`
contains precision / recall / F1 per attack family:

```json
{
  "per_class": {
    "BENIGN":    {"precision": 0.999, "recall": 0.998, "f1": 0.998},
    "DoS":       {"precision": 0.991, "recall": 0.993, "f1": 0.992},
    "DDoS":      {"precision": 0.998, "recall": 0.999, "f1": 0.998},
    ...
  }
}
```

The dashboard's Model Performance page renders this table directly.

## Best-model selection

`src.evaluation.comparison.best_model` picks the model with the highest
F1-weighted on the test set. That model is what the dashboard's
"Predict New CSV" page defaults to.
