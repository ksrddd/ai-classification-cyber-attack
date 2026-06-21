# CICIDS/CIC-IDS raw corpus -- training run `latest`

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
| BENIGN | 207365 | 51841 |  |
| Bot | 2497 | 624 |  |
| Brute Force | 1759 | 440 |  |
| DDoS | 15400 | 3850 |  |
| DoS | 6648 | 1662 |  |
| Heartbleed | 8 | 3 | very low confidence -- treat as anecdote |
| Infiltration | 2375 | 594 |  |
| PortScan | 1546 | 386 |  |
| Web Attack | 2402 | 600 |  |

> Per-class recall for any class with `n_test < 10` should be read as an upper-bound estimate, not a stable metric. This is most visible for classes with only a handful of rows after subsampling, especially Heartbleed.

## Headline metrics

| model | accuracy | balanced acc | f1_macro | f1_weighted | CV f1_macro (mean +/- std) | majority baseline acc | shuffled-labels f1_macro |
|---|---|---|---|---|---|---|---|
| random_forest | 0.9878 | 0.8898 | 0.8979 | 0.9848 | 0.8991 +/- 0.0029 | 0.8640 | 0.1119 |
| xgboost | 0.9767 | 0.9205 | 0.8917 | 0.9801 | 0.8984 +/- 0.0017 | 0.8640 | 0.1090 |
| lightgbm | 0.9817 | 0.9104 | 0.8906 | 0.9825 | 0.9035 +/- 0.0015 | 0.8640 | 0.1097 |

## Verdict on accuracy

- `random_forest` test accuracy 0.9878 -- in the expected range.
-   - majority-class baseline accuracy = 0.8640. Model lift = +0.1238. macro_f1 = 0.8979 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = 0.8991 +/- 0.0029 (small std confirms result is not a single-lucky-split fluke).
-   - shuffled-labels f1_macro = 0.1119 (chance level = 0.1111). Collapse to ~chance confirms the pipeline is NOT leaking labels through preprocessing.

- `xgboost` test accuracy 0.9767 -- in the expected range.
-   - majority-class baseline accuracy = 0.8640. Model lift = +0.1127. macro_f1 = 0.8917 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = 0.8984 +/- 0.0017 (small std confirms result is not a single-lucky-split fluke).
-   - shuffled-labels f1_macro = 0.1090 (chance level = 0.1111). Collapse to ~chance confirms the pipeline is NOT leaking labels through preprocessing.

- `lightgbm` test accuracy 0.9817 -- in the expected range.
-   - majority-class baseline accuracy = 0.8640. Model lift = +0.1176. macro_f1 = 0.8906 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = 0.9035 +/- 0.0015 (small std confirms result is not a single-lucky-split fluke).
-   - shuffled-labels f1_macro = 0.1097 (chance level = 0.1111). Collapse to ~chance confirms the pipeline is NOT leaking labels through preprocessing.

## Top weaknesses + concrete improvements

1. **Minority-class metric variance** -- Heartbleed still has only 11 rows in the combined raw corpus, so its per-class metric is anecdotal. Improvement: report per-class metrics with sample-size caveat (this report already does this).
2. **Subsample bias for the majority** -- if `subsample_n` < full corpus, BENIGN sub-flows from specific application protocols may be underrepresented. Improvement: retrain with `subsample_n=None` on a higher-RAM machine for the final reported numbers.
3. **CICIDS/CIC-IDS labelling noise** -- labels are assigned per attack window, not per flow, so BENIGN flows during an attack window may be mislabelled. Improvement: keep a separate cross-dataset validation run when reporting final research numbers.

## Verifying the clean run

```
python -W error::Warning train.py
```

Any warning becomes a hard exception. Exit code 0 = clean.
