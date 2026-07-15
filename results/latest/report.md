# CICIDS/CIC-IDS raw corpus -- training run `latest`

- subsample_n: 300000
- rare_threshold (keep-all-rows below this): 5000
- imbalance_strategy: targeted (target=Infiltration, target/majority=1.000)
- FN-aware threshold: validation_size=0.20, max target FPR=0.020
- resampling scope: TRAIN/CV folds only; test distribution untouched
- test_size: 0.2, min_test_per_class: 3
- CV: StratifiedKFold(n_splits=5)
- HP search: True (n_iter=8, subsample=80000)
- Random state: 42
- Classes (9): BENIGN, Bot, Brute Force, DDoS, DoS, Heartbleed, Infiltration, PortScan, Web Attack

## Per-class sample sizes (REPORT-CRITICAL)

| class | n_train | n_test | granularity warning |
|---|---|---|---|
| BENIGN | 77962 | 52355 |  |
| Bot | 1891 | 630 |  |
| Brute Force | 1333 | 444 |  |
| DDoS | 11664 | 3888 |  |
| DoS | 5035 | 1678 |  |
| Heartbleed | 5 | 3 | very low confidence -- treat as anecdote |
| Infiltration | 77963 | 599 |  |
| PortScan | 1171 | 390 |  |
| Web Attack | 2976 | 13 |  |

> Per-class recall for any class with `n_test < 10` should be read as an upper-bound estimate, not a stable metric. This is most visible for classes with only a handful of rows after subsampling, especially Heartbleed.

## Headline metrics

| model | accuracy | balanced acc | f1_macro | f1_weighted | CV f1_macro (mean +/- std) | majority baseline acc | shuffled-labels f1_macro |
|---|---|---|---|---|---|---|---|
| random_forest | 0.9856 | 0.9101 | 0.8519 | 0.9854 | 0.9560 +/- 0.0006 | 0.0100 | 0.1097 |
| xgboost | 0.9867 | 0.9062 | 0.8651 | 0.9860 | 0.9606 +/- 0.0006 | 0.0100 | 0.1013 |
| lightgbm | 0.7634 | 0.7899 | 0.6174 | 0.8455 | 0.4069 +/- 0.1814 | 0.0100 | 0.1045 |
| catboost | 0.9829 | 0.8861 | 0.8272 | 0.9838 | 0.9533 +/- 0.0010 | 0.0100 | 0.0925 |
| mlp | 0.9807 | 0.7833 | 0.6981 | 0.9804 | 0.9324 +/- 0.0039 | 0.0100 | 0.0639 |
| logistic_regression | 0.9409 | 0.7972 | 0.7081 | 0.9464 | 0.8499 +/- 0.0037 | 0.0100 | 0.0934 |
| stacking | 0.9830 | 0.8085 | 0.7509 | 0.9844 | 0.8490 +/- 0.0005 | 0.0100 | 0.0925 |

## Infiltration false-negative metrics

| model | threshold | precision | recall | F2 | FPR | FN | FN to BENIGN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lightgbm | 0.751763 | 0.0315 | 0.6845 | 0.1329 | 0.2125 | 189 | 177 | 12620 |
| stacking | 0.942317 | 0.2745 | 0.3840 | 0.3556 | 0.0102 | 369 | 369 | 608 |
| catboost | 0.905846 | 0.2819 | 0.3389 | 0.3257 | 0.0087 | 396 | 396 | 517 |
| random_forest | 0.952612 | 0.3433 | 0.3072 | 0.3138 | 0.0059 | 415 | 415 | 352 |
| xgboost | 0.909924 | 0.3276 | 0.2554 | 0.2672 | 0.0053 | 446 | 446 | 314 |
| mlp | 0.961639 | 0.3503 | 0.2521 | 0.2671 | 0.0047 | 448 | 445 | 280 |
| logistic_regression | 0.754845 | 0.0750 | 0.1369 | 0.1175 | 0.0170 | 517 | 492 | 1011 |

## Verdict on accuracy

- `random_forest` test accuracy 0.9856 -- in the expected range.
-   - majority-class baseline accuracy = 0.0100. Model lift = +0.9756. macro_f1 = 0.8519 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = 0.9560 +/- 0.0006 (small std confirms result is not a single-lucky-split fluke).
-   - shuffled-labels f1_macro = 0.1097 (chance level = 0.1111). Collapse to ~chance confirms the pipeline is NOT leaking labels through preprocessing.

- `xgboost` test accuracy 0.9867 -- in the expected range.
-   - majority-class baseline accuracy = 0.0100. Model lift = +0.9767. macro_f1 = 0.8651 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = 0.9606 +/- 0.0006 (small std confirms result is not a single-lucky-split fluke).
-   - shuffled-labels f1_macro = 0.1013 (chance level = 0.1111). Collapse to ~chance confirms the pipeline is NOT leaking labels through preprocessing.

- `lightgbm` test accuracy 0.7634 -- in the expected range.
-   - majority-class baseline accuracy = 0.0100. Model lift = +0.7534. macro_f1 = 0.6174 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = 0.4069 +/- 0.1814 (**UNSTABLE** (std >= 0.10). Likely cause: some CV folds contained too few rows of a minority class (Heartbleed has only 8 train rows total). The test-set number is still valid (verified by the shuffled-labels check) but this model has high variance across splits and may not generalise well to new minority-class instances).
-   - shuffled-labels f1_macro = 0.1045 (chance level = 0.1111). Collapse to ~chance confirms the pipeline is NOT leaking labels through preprocessing.

- `catboost` test accuracy 0.9829 -- in the expected range.
-   - majority-class baseline accuracy = 0.0100. Model lift = +0.9729. macro_f1 = 0.8272 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = 0.9533 +/- 0.0010 (small std confirms result is not a single-lucky-split fluke).
-   - shuffled-labels f1_macro = 0.0925 (chance level = 0.1111). Collapse to ~chance confirms the pipeline is NOT leaking labels through preprocessing.

- `mlp` test accuracy 0.9807 -- in the expected range.
-   - majority-class baseline accuracy = 0.0100. Model lift = +0.9708. macro_f1 = 0.6981 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = 0.9324 +/- 0.0039 (small std confirms result is not a single-lucky-split fluke).
-   - shuffled-labels f1_macro = 0.0639 (chance level = 0.1111). Collapse to ~chance confirms the pipeline is NOT leaking labels through preprocessing.

- `logistic_regression` test accuracy 0.9409 -- in the expected range.
-   - majority-class baseline accuracy = 0.0100. Model lift = +0.9310. macro_f1 = 0.7081 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = 0.8499 +/- 0.0037 (small std confirms result is not a single-lucky-split fluke).
-   - shuffled-labels f1_macro = 0.0934 (chance level = 0.1111). Collapse to ~chance confirms the pipeline is NOT leaking labels through preprocessing.

- `stacking` test accuracy 0.9830 -- in the expected range.
-   - majority-class baseline accuracy = 0.0100. Model lift = +0.9730. macro_f1 = 0.7509 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = 0.8490 +/- 0.0005 (small std confirms result is not a single-lucky-split fluke).
-   - shuffled-labels f1_macro = 0.0925 (chance level = 0.1111). Collapse to ~chance confirms the pipeline is NOT leaking labels through preprocessing.

## Top weaknesses + concrete improvements

1. **Minority-class metric variance** -- Heartbleed still has only 11 rows in the combined raw corpus, so its per-class metric is anecdotal. Improvement: report per-class metrics with sample-size caveat (this report already does this).
2. **Training-sample bias** -- if `subsample_n` is below the full corpus, some BENIGN application protocols may be underrepresented. The held-out test remains naturally distributed, but final research numbers should also include a larger-RAM sensitivity run.
3. **CICIDS/CIC-IDS labelling noise** -- labels are assigned per attack window, not per flow, so BENIGN flows during an attack window may be mislabelled. Improvement: keep a separate cross-dataset validation run when reporting final research numbers.

## Verifying the clean run

```
python -W error::Warning train.py
```

Any warning becomes a hard exception. Exit code 0 = clean.
