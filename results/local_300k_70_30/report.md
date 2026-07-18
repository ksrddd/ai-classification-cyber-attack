# CICIDS/CIC-IDS raw corpus -- training run `local_300k_70_30`

- subsample_n: 1500000
- rare_threshold (keep-all-rows below this): 5000
- imbalance_strategy: class_weight (target=Infiltration, target/majority=1.000)
- FN-aware threshold: validation_size=0.00, max target FPR=0.020
- resampling scope: TRAIN/CV folds only; test distribution untouched
- test_size: 0.3, min_test_per_class: 3
- CV: skipped
- HP search: False (n_iter=12, subsample=150000)
- Random state: 42
- Classes (9): BENIGN, Bot, Brute Force, DDoS, DoS, Heartbleed, Infiltration, PortScan, Web Attack

## Per-class sample sizes (REPORT-CRITICAL)

| class | n_train | n_test | granularity warning |
|---|---|---|---|
| BENIGN | 32308 | 13846 |  |
| Bot | 25847 | 11077 |  |
| Brute Force | 25847 | 11077 |  |
| DDoS | 25847 | 11077 |  |
| DoS | 25846 | 11077 |  |
| Heartbleed | 3 | 2 | very low confidence -- treat as anecdote |
| Infiltration | 22251 | 9536 |  |
| PortScan | 51347 | 22006 |  |
| Web Attack | 704 | 302 |  |

> Per-class recall for any class with `n_test < 10` should be read as an upper-bound estimate, not a stable metric. This is most visible for classes with only a handful of rows after subsampling, especially Heartbleed.

## Headline metrics

| model | accuracy | balanced acc | f1_macro | f1_weighted | CV f1_macro (mean +/- std) | majority baseline acc | shuffled-labels f1_macro |
|---|---|---|---|---|---|---|---|
| stacking | 0.9529 | 0.8491 | 0.8438 | 0.9532 | skipped | 0.2445 | skipped |
| random_forest | 0.9507 | 0.8987 | 0.9160 | 0.9507 | nan +/- nan | 0.2445 | 0.1138 |
| xgboost | 0.9523 | 0.8478 | 0.8448 | 0.9526 | nan +/- nan | 0.2445 | 0.1088 |
| lightgbm | 0.9507 | 0.8442 | 0.8407 | 0.9509 | nan +/- nan | 0.2445 | 0.1119 |
| catboost | 0.9467 | 0.9561 | 0.9070 | 0.9471 | nan +/- nan | 0.2445 | 0.1002 |
| mlp | 0.9284 | 0.9440 | 0.8822 | 0.9281 | nan +/- nan | 0.2445 | 0.0784 |
| logistic_regression | 0.8994 | 0.8602 | 0.8181 | 0.8991 | nan +/- nan | 0.2445 | 0.0799 |

## Infiltration false-negative metrics

| model | threshold | precision | recall | F2 | FPR | FN | FN to BENIGN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp | native | 0.6290 | 0.8975 | 0.8269 | 0.0627 | 977 | 869 | 5048 |
| catboost | native | 0.7112 | 0.8642 | 0.8285 | 0.0416 | 1295 | 1273 | 3347 |
| stacking | native | 0.7448 | 0.8525 | 0.8285 | 0.0346 | 1407 | 1405 | 2785 |
| xgboost | native | 0.7433 | 0.8482 | 0.8249 | 0.0347 | 1448 | 1447 | 2793 |
| lightgbm | native | 0.7520 | 0.8095 | 0.7973 | 0.0316 | 1817 | 1815 | 2545 |
| logistic_regression | native | 0.5618 | 0.7787 | 0.7229 | 0.0720 | 2110 | 1501 | 5792 |
| random_forest | native | 0.7728 | 0.7707 | 0.7711 | 0.0269 | 2187 | 2173 | 2161 |

## Verdict on accuracy

- `stacking` test accuracy 0.9529 -- in the expected range.
-   - majority-class baseline accuracy = 0.2445. Model lift = +0.7084. macro_f1 = 0.8438 is the load-bearing number on this imbalanced dataset.
-   - CV trust check was skipped for this run to keep full-corpus training practical. Use a subsampled run with CV when you need fold-stability evidence.

- `random_forest` test accuracy 0.9507 -- in the expected range.
-   - majority-class baseline accuracy = 0.2445. Model lift = +0.7062. macro_f1 = 0.9160 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = nan +/- nan (**UNSTABLE** (std >= 0.10). Likely cause: some CV folds contained too few rows of a minority class (Heartbleed has only 8 train rows total). The test-set number is still valid (verified by the shuffled-labels check) but this model has high variance across splits and may not generalise well to new minority-class instances).
-   - shuffled-labels f1_macro = 0.1138 (chance level = 0.1111). Collapse to ~chance confirms the pipeline is NOT leaking labels through preprocessing.

- `xgboost` test accuracy 0.9523 -- in the expected range.
-   - majority-class baseline accuracy = 0.2445. Model lift = +0.7078. macro_f1 = 0.8448 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = nan +/- nan (**UNSTABLE** (std >= 0.10). Likely cause: some CV folds contained too few rows of a minority class (Heartbleed has only 8 train rows total). The test-set number is still valid (verified by the shuffled-labels check) but this model has high variance across splits and may not generalise well to new minority-class instances).
-   - shuffled-labels f1_macro = 0.1088 (chance level = 0.1111). Collapse to ~chance confirms the pipeline is NOT leaking labels through preprocessing.

- `lightgbm` test accuracy 0.9507 -- in the expected range.
-   - majority-class baseline accuracy = 0.2445. Model lift = +0.7062. macro_f1 = 0.8407 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = nan +/- nan (**UNSTABLE** (std >= 0.10). Likely cause: some CV folds contained too few rows of a minority class (Heartbleed has only 8 train rows total). The test-set number is still valid (verified by the shuffled-labels check) but this model has high variance across splits and may not generalise well to new minority-class instances).
-   - shuffled-labels f1_macro = 0.1119 (chance level = 0.1111). Collapse to ~chance confirms the pipeline is NOT leaking labels through preprocessing.

- `catboost` test accuracy 0.9467 -- in the expected range.
-   - majority-class baseline accuracy = 0.2445. Model lift = +0.7022. macro_f1 = 0.9070 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = nan +/- nan (**UNSTABLE** (std >= 0.10). Likely cause: some CV folds contained too few rows of a minority class (Heartbleed has only 8 train rows total). The test-set number is still valid (verified by the shuffled-labels check) but this model has high variance across splits and may not generalise well to new minority-class instances).
-   - shuffled-labels f1_macro = 0.1002 (chance level = 0.1111). Collapse to ~chance confirms the pipeline is NOT leaking labels through preprocessing.

- `mlp` test accuracy 0.9284 -- in the expected range.
-   - majority-class baseline accuracy = 0.2445. Model lift = +0.6839. macro_f1 = 0.8822 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = nan +/- nan (**UNSTABLE** (std >= 0.10). Likely cause: some CV folds contained too few rows of a minority class (Heartbleed has only 8 train rows total). The test-set number is still valid (verified by the shuffled-labels check) but this model has high variance across splits and may not generalise well to new minority-class instances).
-   - shuffled-labels f1_macro = 0.0784 (chance level = 0.1111). Collapse to ~chance confirms the pipeline is NOT leaking labels through preprocessing.

- `logistic_regression` test accuracy 0.8994 -- in the expected range.
-   - majority-class baseline accuracy = 0.2445. Model lift = +0.6549. macro_f1 = 0.8181 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = nan +/- nan (**UNSTABLE** (std >= 0.10). Likely cause: some CV folds contained too few rows of a minority class (Heartbleed has only 8 train rows total). The test-set number is still valid (verified by the shuffled-labels check) but this model has high variance across splits and may not generalise well to new minority-class instances).
-   - shuffled-labels f1_macro = 0.0799 (chance level = 0.1111). Collapse to ~chance confirms the pipeline is NOT leaking labels through preprocessing.

## Top weaknesses + concrete improvements

1. **Minority-class metric variance** -- Heartbleed still has only 11 rows in the combined raw corpus, so its per-class metric is anecdotal. Improvement: report per-class metrics with sample-size caveat (this report already does this).
2. **Training-sample bias** -- if `subsample_n` is below the full corpus, some BENIGN application protocols may be underrepresented. The held-out test remains naturally distributed, but final research numbers should also include a larger-RAM sensitivity run.
3. **CICIDS/CIC-IDS labelling noise** -- labels are assigned per attack window, not per flow, so BENIGN flows during an attack window may be mislabelled. Improvement: keep a separate cross-dataset validation run when reporting final research numbers.

## Verifying the clean run

```
python -W error::Warning train.py
```

Any warning becomes a hard exception. Exit code 0 = clean.
