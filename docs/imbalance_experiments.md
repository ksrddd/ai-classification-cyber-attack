# Class-imbalance experiments

The canonical trainer now keeps the test set at the natural class
distribution and applies every imbalance treatment to training data only.
Sampler-based strategies are part of an `imblearn.Pipeline`, so resampling is
also restricted to the training portion of every cross-validation fold.

The same strategy switch is supported by every canonical model:

- Random Forest
- XGBoost
- LightGBM
- CatBoost
- MLP
- Logistic Regression
- Stacking (LightGBM + XGBoost + Random Forest)

With `class_weight`, each model uses its native weight option or per-fold
sample weights. With any resampling strategy, those weights are disabled so
the imbalance is not corrected twice.

## Recommended Infiltration comparison

Run each experiment under a different run name:

```powershell
# FN-aware run for all seven models (recommended)
.\.venv\Scripts\python.exe main.py --stage train --model all `
  --run-name fn_aware --preset 8gb `
  --imbalance-strategy targeted --target-ratio 1.00 `
  --primary-metric target_f2 --target-max-fpr 0.02 `
  --threshold-validation-size 0.20

# Existing class-weight baseline
.\.venv\Scripts\python.exe main.py --stage train --model xgboost `
  --run-name infiltration_weight --preset 8gb `
  --imbalance-strategy class_weight

# Use more genuine Infiltration rows
.\.venv\Scripts\python.exe main.py --stage train --model xgboost `
  --run-name infiltration_targeted --preset 8gb `
  --imbalance-strategy targeted --target-ratio 1.00 `
  --primary-metric target_f2 --target-max-fpr 0.02

# Duplicate genuine target rows inside each training fold
.\.venv\Scripts\python.exe main.py --stage train --model xgboost `
  --run-name infiltration_random_over --preset 8gb `
  --imbalance-strategy random_over --target-ratio 0.20

# Generate synthetic points near the class boundary
.\.venv\Scripts\python.exe main.py --stage train --model xgboost `
  --run-name infiltration_borderline_smote --preset 8gb `
  --imbalance-strategy borderline_smote --target-ratio 0.20
```

`target_ratio` is `target class count / majority class count` after training
sampling. For the FN-aware protocol, `1.00` retains roughly 104k genuine
Infiltration rows in the 300k budget instead of synthesising replacements.
The test and threshold-calibration sets keep their natural distributions.

`target_max_fpr` bounds the target-class false-positive rate on the separate
calibration set. Increasing it usually reduces false negatives but creates
more false alarms; report both FN and FP whenever this value changes.

## Strategies

| Strategy | Treatment | Class weights |
|---|---|---|
| `class_weight` | Natural train sample | Enabled |
| `targeted` | More genuine target rows, fewer majority rows | Disabled |
| `random_over` | Duplicate genuine target rows in each fold | Disabled |
| `borderline_smote` | Target-only BorderlineSMOTE in each fold | Disabled |
| `smoteenn` | Target-only SMOTE followed by ENN cleaning | Disabled |

Class weighting is disabled for resampled strategies to avoid correcting the
same imbalance twice.

Compare `Infiltration` precision, recall, F1, the confusion matrix, macro-F1,
and balanced accuracy. Do not select the winner by accuracy alone. All outputs
are saved under `results/<run-name>/`.

To re-evaluate an existing model and keep its CSV metrics synchronized with
the regenerated confusion matrix:

```powershell
.\.venv\Scripts\python.exe main.py --stage train `
  --run-name <run-name> --refresh-plots
```
