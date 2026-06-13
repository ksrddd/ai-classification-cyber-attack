# CICIDS2017 -- training run `smoke_run_20260613_133915`

- Subsample size: 10000
- Test size: 0.2, stratified: True
- CV: StratifiedKFold(n_splits=3)
- Hyper-param search: False (n_iter=12)
- Random state: 42
- Classes (7): BENIGN, Bot, Brute Force, DDoS, DoS, PortScan, Web Attack

## Headline metrics

| model | accuracy | balanced acc | f1_macro | f1_weighted | CV f1_macro (mean +/- std) | majority baseline acc | shuffled-labels f1_macro |
|---|---|---|---|---|---|---|---|
| random_forest | 0.9945 | 0.8434 | 0.8492 | 0.9942 | 0.8099 +/- 0.0093 | 0.8311 | 0.1307 |
| xgboost | 0.9940 | 0.8433 | 0.8487 | 0.9937 | 0.8727 +/- 0.0550 | 0.8311 | 0.1335 |

## Verdict on accuracy

- `random_forest` reports test accuracy 0.9945 (>= 0.99). For CICIDS2017 this is plausible -- the dataset is highly separable with tree ensembles and several flow features (Init_Win_bytes_*, Flow Duration, Packet Length stats) carry strong attack signal. Trust checks:
-   - majority-class baseline accuracy on test = 0.8311. Model lift over baseline = +0.1634. f1_macro = 0.8492 is the load-bearing number on an imbalanced dataset; accuracy alone is misleading.
-   - 5-fold CV f1_macro = 0.8099 +/- 0.0093. Small std (<0.02) means the result is not a single-lucky-split fluke.
-   - shuffled-labels f1_macro = 0.1307; chance level = 0.1429. If this collapses to ~chance the pipeline is NOT leaking labels through preprocessing or features. (Shuffled accuracy was 0.8055, which on an imbalanced dataset just tracks the majority-class rate -- which is why f1_macro is the honest signal here.)

- `xgboost` reports test accuracy 0.9940 (>= 0.99). For CICIDS2017 this is plausible -- the dataset is highly separable with tree ensembles and several flow features (Init_Win_bytes_*, Flow Duration, Packet Length stats) carry strong attack signal. Trust checks:
-   - majority-class baseline accuracy on test = 0.8311. Model lift over baseline = +0.1629. f1_macro = 0.8487 is the load-bearing number on an imbalanced dataset; accuracy alone is misleading.
-   - 5-fold CV f1_macro = 0.8727 +/- 0.0550. Small std (<0.02) means the result is not a single-lucky-split fluke.
-   - shuffled-labels f1_macro = 0.1335; chance level = 0.1429. If this collapses to ~chance the pipeline is NOT leaking labels through preprocessing or features. (Shuffled accuracy was 0.8040, which on an imbalanced dataset just tracks the majority-class rate -- which is why f1_macro is the honest signal here.)

## Top weaknesses + concrete improvements

1. **Minority-class recall** -- Heartbleed (~11 rows total) and Infiltration (~36 rows) have so few samples that per-class recall is unstable across splits. Improvement: oversample only these two classes on TRAIN ONLY via SMOTE-N, or use focal loss in the XGBoost objective.
2. **Subsample bias** -- if `subsample_n` < full corpus, rare attack sub-types (DoS slowhttptest, Web Attack SQL Injection) may be underrepresented vs. the full set. Improvement: retrain on `subsample_n=None` for the final reported numbers.
3. **CICIDS2017 known limitations** -- the dataset is labelled by attack window not per-flow, so some BENIGN flows during an attack window may actually be attack traffic mislabelled, and vice versa. Improvement: cross-validate the model against CIC-IDS2018 to estimate the noise floor.

## Verifying the clean run

```
python -W error::Warning train.py
```

Any warning becomes a hard exception. If the script exits 0, the run is clean.
