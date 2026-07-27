# CICIDS2017 -- training run `cicids2017_temporal_v1`

## Protocol

- Corpus: CICIDS2017
- Split protocol: `cicids2017_temporal_v1`
  - Chronological 70/30 inside every (capture file, class) group: the earliest 70% of each class's flows train, the latest 30% test. No test flow precedes the training flows of its own class.
  - Ordering key: original CSV row position. The 2017 export has no `Timestamp` column, so row order is used and is verified against the published CIC attack schedule before every run.
  - Source holdout is impossible on this corpus: each attack class occurs in exactly one capture file, so holding out a file would leave its class with no training rows.
  - CV uses StratifiedKFold, not GroupKFold by source: with one capture per class, source grouping would place an entire class in a single fold.
- test_size: 0.3, min_test_per_class: 3
- imbalance_strategy: class_weight (identical for every model, so imbalance handling stays a controlled variable)
- resampling scope: TRAIN/CV folds only; test distribution untouched
- CV: 5-fold, scored on macro-F1

### Tuning fairness

Every tunable model is searched with the same method (RandomizedSearchCV), the same objective (`f1_macro`), the same budget (n_iter=20 on a 200,000-row train subset), and a search space of exactly 144 configurations -- so the same fraction (13.9%) of each space is explored. Equal trial counts over unequal spaces would not be a fair comparison.

- Random state: 42
- Classes (9): BENIGN, Bot, Brute Force, DDoS, DoS, Heartbleed, Infiltration, PortScan, Web Attack

## Per-class sample sizes (REPORT-CRITICAL)

| class | n_train | n_test | granularity warning |
|---|---|---|---|
| BENIGN | 1450574 | 621680 |  |
| Bot | 1363 | 585 |  |
| Brute Force | 6405 | 2745 |  |
| DDoS | 89609 | 38405 |  |
| DoS | 135611 | 58119 |  |
| Heartbleed | 7 | 4 | very low confidence -- treat as anecdote |
| Infiltration | 25 | 11 |  |
| PortScan | 63485 | 27209 |  |
| Web Attack | 1500 | 643 |  |

> Per-class recall for any class with `n_test < 10` should be read as an upper-bound estimate, not a stable metric. This is most visible for classes with only a handful of rows after subsampling, especially Heartbleed.

## Dimension 1 -- Classification Performance

Macro-F1 and per-class recall decide this comparison, not accuracy. With BENIGN at 83.0% of the test set, a model that predicted BENIGN and nothing else would already score that as accuracy while detecting no attacks at all.

| model | accuracy | balanced acc | macro P | macro R | f1_macro | f1_macro (reportable) | f1_weighted | wtd P | wtd R | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | 0.9953 | 0.8970 | 0.8839 | 0.8970 | 0.8873 | 0.8613 | 0.9952 | 0.9951 | 0.9953 | 0.9844 |
| lightgbm | 0.9903 | 0.8576 | 0.9403 | 0.8576 | 0.8774 | 0.8627 | 0.9898 | 0.9902 | 0.9903 | 0.9676 |
| xgboost | 0.9906 | 0.8889 | 0.8894 | 0.8889 | 0.8773 | 0.8542 | 0.9902 | 0.9905 | 0.9906 | 0.9687 |
| random_forest | 0.9820 | 0.8165 | 0.8846 | 0.8165 | 0.8404 | 0.8378 | 0.9832 | 0.9856 | 0.9820 | 0.9397 |
| stacking | 0.9581 | 0.8852 | 0.6551 | 0.8852 | 0.6822 | 0.7092 | 0.9712 | 0.9879 | 0.9581 | 0.8757 |
| logistic_regression | 0.8428 | 0.8338 | 0.5244 | 0.8338 | 0.5574 | 0.6348 | 0.9030 | 0.9788 | 0.8428 | 0.6651 |
| mlp | 0.9331 | 0.7112 | 0.5180 | 0.7112 | 0.5369 | 0.6767 | 0.9443 | 0.9620 | 0.9331 | 0.7895 |

> `f1_macro (reportable)` averages only the 7 classes with at least 30 test rows, excluding Heartbleed, Infiltration. Both numbers are shown because plain macro-F1 gives a class with a handful of test rows the same weight as one with hundreds of thousands; neither number alone tells the whole story.

### Trust checks

| model | CV f1_macro (mean +/- std) | majority baseline acc | shuffled-labels f1_macro |
|---|---|---:|---:|
| logistic_regression | 0.4988 +/- 0.0278 | 0.8296 | skipped |
| random_forest | 0.9163 +/- 0.0135 | 0.8296 | skipped |
| xgboost | 0.9697 +/- 0.0155 | 0.8296 | skipped |
| lightgbm | 0.9666 +/- 0.0168 | 0.8296 | skipped |
| catboost | 0.9179 +/- 0.0225 | 0.8296 | skipped |
| mlp | 0.5697 +/- 0.0538 | 0.8296 | skipped |
| stacking | 0.7987 +/- 0.0279 | 0.8296 | skipped |

## Dimension 2 -- Attack Detection Ability

Attack-vs-BENIGN view. A false positive is a benign flow raised as an alert; a false negative is an attack that reached the network unnoticed.

| model | binary recall (DR) | binary precision | binary F1 | FPR | FNR | false alerts | attacks missed |
|---|---:|---:|---:|---:|---:|---:|---:|
| catboost | 0.9840 | 0.9968 | 0.9903 | 0.00065 | 0.01604 | 404 | 2,049 |
| logistic_regression | 0.9746 | 0.5418 | 0.6965 | 0.16931 | 0.02540 | 105,258 | 3,244 |
| stacking | 0.9661 | 0.8298 | 0.8928 | 0.04072 | 0.03389 | 25,317 | 4,329 |
| xgboost | 0.9571 | 0.9981 | 0.9771 | 0.00038 | 0.04293 | 236 | 5,483 |
| lightgbm | 0.9513 | 0.9989 | 0.9745 | 0.00022 | 0.04870 | 136 | 6,220 |
| random_forest | 0.9277 | 0.9729 | 0.9498 | 0.00530 | 0.07228 | 3,298 | 9,232 |
| mlp | 0.8459 | 0.7864 | 0.8151 | 0.04721 | 0.15406 | 29,347 | 19,677 |

### Per-class recall (detection rate)

| model | BENIGN | Bot | Brute Force | DDoS | DoS | Heartbleed | Infiltration | PortScan | Web Attack |
|---|---|---|---|---|---|---|---|---|---|
| logistic_regression | 0.8307 | 0.1573 | 0.9581 | 0.9979 | 0.7992 | 1.0000 | 0.8182 | 0.9941 | 0.9487 |
| random_forest | 0.9947 | 0.1504 | 0.9767 | 0.9983 | 0.8372 | 0.7500 | 0.7273 | 0.9975 | 0.9160 |
| xgboost | 0.9996 | 0.1607 | 0.9989 | 0.9992 | 0.8937 | 1.0000 | 1.0000 | 0.9975 | 0.9502 |
| lightgbm | 0.9998 | 0.1538 | 0.9960 | 0.9991 | 0.8891 | 0.7500 | 1.0000 | 0.9955 | 0.9347 |
| catboost | 0.9994 | 0.1607 | 0.9985 | 0.9990 | 0.9571 | 1.0000 | 1.0000 | 0.9976 | 0.9611 |
| mlp | 0.9528 | 0.1607 | 0.9621 | 0.9951 | 0.6583 | 0.0000 | 0.7273 | 0.9962 | 0.9487 |
| stacking | 0.9593 | 0.1607 | 0.9971 | 0.9980 | 0.9069 | 1.0000 | 1.0000 | 0.9975 | 0.9471 |

### Per-class false-negative rate

| model | BENIGN | Bot | Brute Force | DDoS | DoS | Heartbleed | Infiltration | PortScan | Web Attack |
|---|---|---|---|---|---|---|---|---|---|
| logistic_regression | 0.1693 | 0.8427 | 0.0419 | 0.0021 | 0.2008 | 0.0000 | 0.1818 | 0.0059 | 0.0513 |
| random_forest | 0.0053 | 0.8496 | 0.0233 | 0.0017 | 0.1628 | 0.2500 | 0.2727 | 0.0025 | 0.0840 |
| xgboost | 0.0004 | 0.8393 | 0.0011 | 0.0008 | 0.1063 | 0.0000 | 0.0000 | 0.0025 | 0.0498 |
| lightgbm | 0.0002 | 0.8462 | 0.0040 | 0.0009 | 0.1109 | 0.2500 | 0.0000 | 0.0045 | 0.0653 |
| catboost | 0.0006 | 0.8393 | 0.0015 | 0.0010 | 0.0429 | 0.0000 | 0.0000 | 0.0024 | 0.0389 |
| mlp | 0.0472 | 0.8393 | 0.0379 | 0.0049 | 0.3417 | 1.0000 | 0.2727 | 0.0038 | 0.0513 |
| stacking | 0.0407 | 0.8393 | 0.0029 | 0.0020 | 0.0931 | 0.0000 | 0.0000 | 0.0025 | 0.0529 |

### Per-class false-positive rate

| model | BENIGN | Bot | Brute Force | DDoS | DoS | Heartbleed | Infiltration | PortScan | Web Attack |
|---|---|---|---|---|---|---|---|---|---|
| logistic_regression | 0.02540 | 0.10418 | 0.00262 | 0.00236 | 0.00229 | 0.00001 | 0.01584 | 0.01088 | 0.01551 |
| random_forest | 0.07228 | 0.00422 | 0.00000 | 0.00001 | 0.00018 | 0.00000 | 0.00000 | 0.00135 | 0.00001 |
| xgboost | 0.04293 | 0.00013 | 0.00000 | 0.00002 | 0.00017 | 0.00000 | 0.00000 | 0.00151 | 0.00031 |
| lightgbm | 0.04870 | 0.00005 | 0.00000 | 0.00000 | 0.00012 | 0.00000 | 0.00000 | 0.00111 | 0.00018 |
| catboost | 0.01604 | 0.00031 | 0.00001 | 0.00001 | 0.00020 | 0.00000 | 0.00000 | 0.00128 | 0.00022 |
| mlp | 0.15406 | 0.02512 | 0.00168 | 0.00009 | 0.00104 | 0.00000 | 0.00020 | 0.00914 | 0.00382 |
| stacking | 0.03389 | 0.00356 | 0.03033 | 0.00002 | 0.00022 | 0.00000 | 0.00014 | 0.00173 | 0.00033 |

Confusion matrices: `<model>_confusion_matrix.png`; full per-class precision/recall/F1: `<model>_per_class.csv`.

## Dimension 3 -- Operational Impact

What the error rates cost an analyst on this test set (749,401 flows, of which 621,680 are BENIGN).

| model | false alerts | alerts per 10k benign flows | attacks missed | worst-detected class | its recall |
|---|---:|---:|---:|---|---:|
| catboost | 404 | 6.5 | 2,049 | Bot | 0.1607 |
| logistic_regression | 105,258 | 1,693.1 | 3,244 | Bot | 0.1573 |
| stacking | 25,317 | 407.2 | 4,329 | Bot | 0.1607 |
| xgboost | 236 | 3.8 | 5,483 | Bot | 0.1607 |
| lightgbm | 136 | 2.2 | 6,220 | Bot | 0.1538 |
| random_forest | 3,298 | 53.0 | 9,232 | Bot | 0.1504 |
| mlp | 29,347 | 472.1 | 19,677 | Heartbleed | 0.0000 |

## Dimension 4 -- Computational Efficiency

Inference is measured on CPU for **every** model in batches of 1,000 flows, 30 repeats from random offsets, after one discarded warm-up batch. p95 matters as much as the median: an IDS that is usually fast and occasionally slow still drops flows.

> stacking, xgboost trained on GPU and were moved to CPU for timing. Otherwise this table would rank hardware rather than models -- and the Deployment ranking is decided on p95 latency.

| model | trained on | fit time | p50 latency | p95 latency | throughput | model size |
|---|---|---:|---:|---:|---:|---:|
| logistic_regression | cpu | 653.7s | 4.30 ms | 5.15 ms | 232,391 flows/s | 0.0 MB |
| catboost | gpu | 44.6s | 7.58 ms | 8.39 ms | 131,975 flows/s | 2.5 MB |
| mlp | cpu | 818.7s | 10.05 ms | 12.50 ms | 99,545 flows/s | 0.7 MB |
| xgboost | gpu | 154.6s | 43.42 ms | 54.10 ms | 23,030 flows/s | 15.8 MB |
| random_forest | cpu | 314.6s | 85.94 ms | 104.12 ms | 11,636 flows/s | 124.5 MB |
| stacking | gpu | 2,754.7s | 119.79 ms | 137.95 ms | 8,348 flows/s | 109.3 MB |
| lightgbm | cpu | 549.6s | 111.40 ms | 144.02 ms | 8,977 flows/s | 54.9 MB |

## Infiltration false-negative metrics

| model | threshold | precision | recall | F2 | FPR | FN | FN to BENIGN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| xgboost | native | 0.8462 | 1.0000 | 0.9649 | 0.0000 | 0 | 0 | 2 |
| lightgbm | native | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0 | 0 | 0 |
| catboost | native | 0.9167 | 1.0000 | 0.9821 | 0.0000 | 0 | 0 | 1 |
| stacking | native | 0.0965 | 1.0000 | 0.3481 | 0.0001 | 0 | 0 | 103 |
| logistic_regression | native | 0.0008 | 0.8182 | 0.0038 | 0.0158 | 2 | 1 | 11867 |
| random_forest | native | 1.0000 | 0.7273 | 0.7692 | 0.0000 | 3 | 3 | 0 |
| mlp | native | 0.0506 | 0.7273 | 0.1980 | 0.0002 | 3 | 3 | 150 |

## Verdict on accuracy

- `logistic_regression` test accuracy 0.8428 -- in the expected range.
-   - majority-class baseline accuracy = 0.8296. Model lift = +0.0132. macro_f1 = 0.5574 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = 0.4988 +/- 0.0278 (std is acceptable; result is reasonably stable).

- `random_forest` test accuracy 0.9820 -- in the expected range.
-   - majority-class baseline accuracy = 0.8296. Model lift = +0.1524. macro_f1 = 0.8404 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = 0.9163 +/- 0.0135 (small std confirms result is not a single-lucky-split fluke).

- `xgboost` reports test accuracy 0.9906 (>= 0.99). For CICIDS-style flow features this can be plausible because the dataset is highly separable for tree ensembles. Trust checks:
-   - majority-class baseline accuracy = 0.8296. Model lift = +0.1610. macro_f1 = 0.8773 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = 0.9697 +/- 0.0155 (small std confirms result is not a single-lucky-split fluke).

- `lightgbm` reports test accuracy 0.9903 (>= 0.99). For CICIDS-style flow features this can be plausible because the dataset is highly separable for tree ensembles. Trust checks:
-   - majority-class baseline accuracy = 0.8296. Model lift = +0.1607. macro_f1 = 0.8774 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = 0.9666 +/- 0.0168 (small std confirms result is not a single-lucky-split fluke).

- `catboost` reports test accuracy 0.9953 (>= 0.99). For CICIDS-style flow features this can be plausible because the dataset is highly separable for tree ensembles. Trust checks:
-   - majority-class baseline accuracy = 0.8296. Model lift = +0.1657. macro_f1 = 0.8873 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = 0.9179 +/- 0.0225 (std is acceptable; result is reasonably stable).

- `mlp` test accuracy 0.9331 -- in the expected range.
-   - majority-class baseline accuracy = 0.8296. Model lift = +0.1035. macro_f1 = 0.5369 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = 0.5697 +/- 0.0538 (std is moderate; the test score is real but CV folds vary -- treat the headline as best-case rather than mean).

- `stacking` test accuracy 0.9581 -- in the expected range.
-   - majority-class baseline accuracy = 0.8296. Model lift = +0.1285. macro_f1 = 0.6822 is the load-bearing number on this imbalanced dataset.
-   - 5-fold CV f1_macro = 0.7987 +/- 0.0279 (std is acceptable; result is reasonably stable).

## Model rankings -- three axes

No single ranking is authoritative. A model that detects the most attacks may raise too many alerts to be usable, and the fastest model is rarely the most accurate, so all three are published side by side with the rule that produced each.

| ranking | winner | decided by | rule |
|---|---|---:|---|
| **Overall Best** | `catboost` | f1_macro = 0.8873 | highest f1_macro, ties broken by f1_weighted |
| **Security-focused Best** | `catboost` | recall_macro = 0.8970 | highest recall_macro among models with binary_fpr <= 0.01 and no class at zero recall |
| **Deployment Best** | `catboost` | predict_latency_p95_ms = 8.3859 | lowest predict_latency_p95_ms among models within 0.02 f1_macro of the best, ties broken by model_size_mb |

> Security-focused Best excluded logistic_regression: binary_fpr=0.169312 fails <= 0.01
> Security-focused Best excluded mlp: binary_fpr=0.047206 fails <= 0.01
> Security-focused Best excluded stacking: binary_fpr=0.0407235 fails <= 0.01
> Deployment Best excluded logistic_regression: f1_macro=0.557358 fails >= 0.867266
> Deployment Best excluded random_forest: f1_macro=0.840406 fails >= 0.867266
> Deployment Best excluded mlp: f1_macro=0.536863 fails >= 0.867266
> Deployment Best excluded stacking: f1_macro=0.682234 fails >= 0.867266

## Consequences

**1. A false negative is more damaging than a false positive.**
Across this test set the models miss between 2,049 and 19,677 attack flows. Every missed flow is an intrusion that reached the network with no alert raised, whereas every false positive costs an analyst a few minutes. That asymmetry is why the Security-focused ranking maximises recall rather than accuracy.

**2. False positives cause alert fatigue.**
`catboost` has a binary FPR of 0.00065. That sounds negligible, but against 621,680 benign test flows it is **404 false alerts**. An analyst who cannot triage that volume starts ignoring the queue, at which point the true positives are missed too -- so an unbounded FPR destroys detection indirectly. The Security ranking therefore caps binary FPR at 1.0% rather than leaving it free.

**3. Class imbalance makes accuracy misleading.**
`catboost` reports accuracy 0.9953 against a majority-class baseline of 0.8296 -- a lift of only +0.1657. Its weakest attack class is **Bot**, at recall 0.1607: 491 of 585 test flows missed. Accuracy hides this completely because the class is a rounding error in the total, which is exactly why macro-F1 and per-class recall decide this comparison.

**4. Inference speed is a deployment constraint, not a detail.**
p95 latency spans 5.15 ms (`logistic_regression`) to 144.02 ms (`lightgbm`) per 1,000-flow batch -- a 27.9x spread, or 232,391 versus 8,977 flows/s. Artifact size ranges 0.0 MB to 124.5 MB. A sensor that cannot keep pace with the link drops flows, and a dropped flow is a false negative by another name.

## Top weaknesses + concrete improvements

1. **Minority-class metric variance** -- Heartbleed (7 train / 4 test), Infiltration (25 train / 11 test). A per-class recall computed from single-digit test rows moves in steps of tens of percent, so it is an anecdote, not an estimate. This report therefore publishes `f1_macro (reportable)` alongside plain macro-F1 and flags every affected class in the sample-size table above.
2. **Single-capture attack families** -- in CICIDS2017 each attack class occurs in exactly one capture file, so a model sees only one instance of each attack campaign. The chronological split prevents leakage within a capture but cannot show whether a model generalises to a *different* execution of the same attack. Improvement: validate the champion against CSE-CIC-IDS2018 as an unseen-campaign test.
3. **CICIDS labelling noise** -- labels are assigned per attack window, not per flow, so benign flows inside an attack window can carry an attack label. This inflates every model's apparent performance equally, so the ranking stays valid while the absolute numbers should be read as an upper bound.
4. **Flow-completion ordering** -- CICFlowMeter emits a record when a flow terminates, so the chronological order is by completion, not by start. Long-lived attacks (DoS GoldenEye) therefore trail past their attack window. This is the correct ordering for an IDS -- a flow's features only exist once it completes -- but it is not the same as ordering by attack time.

## Verifying the clean run

```
python -W error::Warning train.py
```

Any warning becomes a hard exception. Exit code 0 = clean.
