# CICIDS2017 -- training run `latest`

- subsample_n: 300000
- rare_threshold (keep-all-rows below this): 5000
- test_size: 0.2, min_test_per_class: 3
- CV: StratifiedKFold(n_splits=5)
- HP search: True (n_iter=8, subsample=80000)
- Random state: 42
- Classes (9): BENIGN, Bot, Brute Force, DDoS, DoS, Heartbleed, Infiltration, PortScan, Web Attack

## Per-class sample sizes (REPORT-CRITICAL)

| class | n_train | n_test | granularity warning |
|---|---|---|---|
| BENIGN | 197038 | 49260 |  |
| Bot | 1558 | 390 |  |
| Brute Force | 861 | 215 |  |
| DDoS | 12040 | 3010 |  |
| DoS | 18222 | 4555 |  |
| Heartbleed | 8 | 3 | very low confidence -- treat as anecdote |
| Infiltration | 29 | 7 | indicative only -- small N |
| PortScan | 8530 | 2132 |  |
| Web Attack | 1714 | 429 |  |

> Per-class recall for any class with `n_test < 10` should be read as an upper-bound estimate, not a stable metric. This is an inherent limitation of CICIDS2017 -- the dataset ships with only 11 Heartbleed and 36 Infiltration flows.

## Headline metrics

| model | accuracy | balanced acc | f1_macro | f1_weighted | CV f1_macro (mean +/- std) | majority baseline acc | shuffled-labels f1_macro |
|---|---|---|---|---|---|---|---|
| random_forest | 0.9964 | 0.9802 | 0.9715 | 0.9966 | 0.9587 +/- 0.0256 | 0.8210 | 0.1042 |
| xgboost | 0.9986 | 0.9836 | 0.9778 | 0.9986 | 0.9787 +/- 0.0200 | 0.8210 | 0.0923 |
| lightgbm | 0.9984 | 0.9808 | 0.9851 | 0.9984 | 0.9759 +/- 0.0141 | 0.8210 | 0.1067 |

## Verdict on accuracy

- `random_forest` reports test accuracy 0.9964 (>= 0.99). For CICIDS2017 this is plausible -- the dataset is highly separable for tree ensembles. Trust checks:
-   - majority-class baseline accuracy = 0.8210. Model lift = +0.1754. macro_f1 = 0.9715 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = 0.9587 +/- 0.0256 (std is acceptable; result is reasonably stable).
-   - shuffled-labels f1_macro = 0.1042 (chance level = 0.1111). Collapse to ~chance confirms the pipeline is NOT leaking labels through preprocessing.

- `xgboost` reports test accuracy 0.9986 (>= 0.99). For CICIDS2017 this is plausible -- the dataset is highly separable for tree ensembles. Trust checks:
-   - majority-class baseline accuracy = 0.8210. Model lift = +0.1776. macro_f1 = 0.9778 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = 0.9787 +/- 0.0200 (std is acceptable; result is reasonably stable).
-   - shuffled-labels f1_macro = 0.0923 (chance level = 0.1111). Collapse to ~chance confirms the pipeline is NOT leaking labels through preprocessing.

- `lightgbm` reports test accuracy 0.9984 (>= 0.99). For CICIDS2017 this is plausible -- the dataset is highly separable for tree ensembles. Trust checks:
-   - majority-class baseline accuracy = 0.8210. Model lift = +0.1774. macro_f1 = 0.9851 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = 0.9759 +/- 0.0141 (small std confirms result is not a single-lucky-split fluke).
-   - shuffled-labels f1_macro = 0.1067 (chance level = 0.1111). Collapse to ~chance confirms the pipeline is NOT leaking labels through preprocessing.

## Top weaknesses + concrete improvements

1. **Minority-class metric variance** -- Heartbleed (11 total rows) and Infiltration (36) cannot give stable per-class metrics with any pipeline. Improvement: report per-class metrics with sample-size caveat (this report already does this).
2. **Subsample bias for the majority** -- if `subsample_n` < full corpus, BENIGN sub-flows from specific application protocols may be underrepresented. Improvement: retrain with `subsample_n=None` on a higher-RAM machine for the final reported numbers.
3. **CICIDS2017 labelling noise** -- labels are assigned per attack window, not per flow, so BENIGN flows during an attack window may be mislabelled. Improvement: cross-validate against CIC-IDS2018 to estimate the noise floor.

## Verifying the clean run

```
python -W error::Warning train.py
```

Any warning becomes a hard exception. Exit code 0 = clean.
