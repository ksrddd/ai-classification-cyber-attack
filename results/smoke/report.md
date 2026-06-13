# CICIDS2017 -- training run `smoke`

- subsample_n: 10000
- rare_threshold (keep-all-rows below this): 5000
- test_size: 0.2, min_test_per_class: 3
- CV: StratifiedKFold(n_splits=3)
- HP search: False (n_iter=8, subsample=80000)
- Random state: 42
- Classes (9): BENIGN, Bot, Brute Force, DDoS, DoS, Heartbleed, Infiltration, PortScan, Web Attack

## Per-class sample sizes (REPORT-CRITICAL)

| class | n_train | n_test | granularity warning |
|---|---|---|---|
| BENIGN | 3904 | 976 |  |
| Bot | 1558 | 390 |  |
| Brute Force | 17 | 4 | very low confidence -- treat as anecdote |
| DDoS | 238 | 60 |  |
| DoS | 361 | 90 |  |
| Heartbleed | 8 | 3 | very low confidence -- treat as anecdote |
| Infiltration | 29 | 7 | indicative only -- small N |
| PortScan | 169 | 42 |  |
| Web Attack | 1714 | 429 |  |

> Per-class recall for any class with `n_test < 10` should be read as an upper-bound estimate, not a stable metric. This is an inherent limitation of CICIDS2017 -- the dataset ships with only 11 Heartbleed and 36 Infiltration flows.

## Headline metrics

| model | accuracy | balanced acc | f1_macro | f1_weighted | CV f1_macro (mean +/- std) | majority baseline acc | shuffled-labels f1_macro |
|---|---|---|---|---|---|---|---|
| random_forest | 0.9915 | 0.9597 | 0.9758 | 0.9914 | 0.9716 +/- 0.0011 | 0.4878 | 0.1132 |
| xgboost | 0.9960 | 0.9821 | 0.9890 | 0.9960 | 0.9739 +/- 0.0091 | 0.4878 | 0.1081 |
| lightgbm | 0.9955 | 0.9818 | 0.9888 | 0.9955 | 0.9837 +/- 0.0123 | 0.4878 | 0.1113 |

## Verdict on accuracy

- `random_forest` reports test accuracy 0.9915 (>= 0.99). For CICIDS2017 this is plausible -- the dataset is highly separable for tree ensembles. Trust checks:
-   - majority-class baseline accuracy = 0.4878. Model lift = +0.5037. macro_f1 = 0.9758 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = 0.9716 +/- 0.0011 (small std confirms result is not a single-lucky-split fluke).
-   - shuffled-labels f1_macro = 0.1132 (chance level = 0.1111). Collapse to ~chance confirms the pipeline is NOT leaking labels through preprocessing.

- `xgboost` reports test accuracy 0.9960 (>= 0.99). For CICIDS2017 this is plausible -- the dataset is highly separable for tree ensembles. Trust checks:
-   - majority-class baseline accuracy = 0.4878. Model lift = +0.5082. macro_f1 = 0.9890 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = 0.9739 +/- 0.0091 (small std confirms result is not a single-lucky-split fluke).
-   - shuffled-labels f1_macro = 0.1081 (chance level = 0.1111). Collapse to ~chance confirms the pipeline is NOT leaking labels through preprocessing.

- `lightgbm` reports test accuracy 0.9955 (>= 0.99). For CICIDS2017 this is plausible -- the dataset is highly separable for tree ensembles. Trust checks:
-   - majority-class baseline accuracy = 0.4878. Model lift = +0.5077. macro_f1 = 0.9888 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = 0.9837 +/- 0.0123 (small std confirms result is not a single-lucky-split fluke).
-   - shuffled-labels f1_macro = 0.1113 (chance level = 0.1111). Collapse to ~chance confirms the pipeline is NOT leaking labels through preprocessing.

## Top weaknesses + concrete improvements

1. **Minority-class metric variance** -- Heartbleed (11 total rows) and Infiltration (36) cannot give stable per-class metrics with any pipeline. Improvement: report per-class metrics with sample-size caveat (this report already does this).
2. **Subsample bias for the majority** -- if `subsample_n` < full corpus, BENIGN sub-flows from specific application protocols may be underrepresented. Improvement: retrain with `subsample_n=None` on a higher-RAM machine for the final reported numbers.
3. **CICIDS2017 labelling noise** -- labels are assigned per attack window, not per flow, so BENIGN flows during an attack window may be mislabelled. Improvement: cross-validate against CIC-IDS2018 to estimate the noise floor.

## Verifying the clean run

```
python -W error::Warning train.py
```

Any warning becomes a hard exception. Exit code 0 = clean.
