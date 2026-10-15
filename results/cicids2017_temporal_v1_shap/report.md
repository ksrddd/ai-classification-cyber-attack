# CICIDS2017 -- training run `cicids2017_temporal_v1_shap`

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
- CV: skipped

### Tuning fairness

Hyperparameter search was skipped for this run; every model uses its configured defaults.

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
| catboost | 0.9951 | 0.8969 | 0.8828 | 0.8969 | 0.8858 | 0.8593 | 0.9950 | 0.9950 | 0.9951 | 0.9838 |

> `f1_macro (reportable)` averages only the 7 classes with at least 30 test rows, excluding Heartbleed, Infiltration. Both numbers are shown because plain macro-F1 gives a class with a handful of test rows the same weight as one with hundreds of thousands; neither number alone tells the whole story.

### Trust checks

| model | CV f1_macro (mean +/- std) | majority baseline acc | shuffled-labels f1_macro |
|---|---|---:|---:|
| catboost | skipped | 0.8296 | skipped |

## Dimension 2 -- Attack Detection Ability

Attack-vs-BENIGN view. A false positive is a benign flow raised as an alert; a false negative is an attack that reached the network unnoticed.

| model | binary recall (DR) | binary precision | binary F1 | FPR | FNR | false alerts | attacks missed |
|---|---:|---:|---:|---:|---:|---:|---:|
| catboost | 0.9806 | 0.9969 | 0.9887 | 0.00063 | 0.01943 | 389 | 2,482 |

### Per-class recall (detection rate)

| model | BENIGN | Bot | Brute Force | DDoS | DoS | Heartbleed | Infiltration | PortScan | Web Attack |
|---|---|---|---|---|---|---|---|---|---|
| catboost | 0.9994 | 0.1607 | 0.9985 | 0.9990 | 0.9547 | 1.0000 | 1.0000 | 0.9975 | 0.9627 |

### Per-class false-negative rate

| model | BENIGN | Bot | Brute Force | DDoS | DoS | Heartbleed | Infiltration | PortScan | Web Attack |
|---|---|---|---|---|---|---|---|---|---|
| catboost | 0.0006 | 0.8393 | 0.0015 | 0.0010 | 0.0453 | 0.0000 | 0.0000 | 0.0025 | 0.0373 |

### Per-class false-positive rate

| model | BENIGN | Bot | Brute Force | DDoS | DoS | Heartbleed | Infiltration | PortScan | Web Attack |
|---|---|---|---|---|---|---|---|---|---|
| catboost | 0.01943 | 0.00029 | 0.00001 | 0.00001 | 0.00020 | 0.00000 | 0.00000 | 0.00082 | 0.00027 |

Confusion matrices: `<model>_confusion_matrix.png`; full per-class precision/recall/F1: `<model>_per_class.csv`.

## Dimension 3 -- Operational Impact

What the error rates cost an analyst on this test set (749,401 flows, of which 621,680 are BENIGN).

| model | false alerts | alerts per 10k benign flows | attacks missed | worst-detected class | its recall |
|---|---:|---:|---:|---|---:|
| catboost | 389 | 6.3 | 2,482 | Bot | 0.1607 |

## Dimension 4 -- Computational Efficiency

Inference is measured on CPU for **every** model in batches of 1,000 flows, 30 repeats from random offsets, after one discarded warm-up batch. p95 matters as much as the median: an IDS that is usually fast and occasionally slow still drops flows.

| model | trained on | fit time | p50 latency | p95 latency | throughput | model size |
|---|---|---:|---:|---:|---:|---:|
| catboost | gpu | 39.6s | 5.15 ms | 5.68 ms | 194,352 flows/s | 2.5 MB |

## Infiltration false-negative metrics

| model | threshold | precision | recall | F2 | FPR | FN | FN to BENIGN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | native | 0.9167 | 1.0000 | 0.9821 | 0.0000 | 0 | 0 | 1 |

## Verdict on accuracy

- `catboost` reports test accuracy 0.9951 (>= 0.99). For CICIDS-style flow features this can be plausible because the dataset is highly separable for tree ensembles. Trust checks:
-   - majority-class baseline accuracy = 0.8296. Model lift = +0.1656. macro_f1 = 0.8858 is the load-bearing number on this imbalanced dataset.
-   - CV trust check was skipped for this run to keep full-corpus training practical. Use a subsampled run with CV when you need fold-stability evidence.

## Model rankings -- three axes

No single ranking is authoritative. A model that detects the most attacks may raise too many alerts to be usable, and the fastest model is rarely the most accurate, so all three are published side by side with the rule that produced each.

| ranking | winner | decided by | rule |
|---|---|---:|---|
| **Overall Best** | `catboost` | f1_macro = 0.8858 | highest f1_macro, ties broken by f1_weighted |
| **Security-focused Best** | `catboost` | recall_macro = 0.8969 | highest recall_macro among models with binary_fpr <= 0.01 and no class at zero recall |
| **Deployment Best** | `catboost` | predict_latency_p95_ms = 5.6823 | lowest predict_latency_p95_ms among models within 0.02 f1_macro of the best, ties broken by model_size_mb |


## Consequences

**1. A false negative is more damaging than a false positive.**
Across this test set the models miss between 2,482 and 2,482 attack flows. Every missed flow is an intrusion that reached the network with no alert raised, whereas every false positive costs an analyst a few minutes. That asymmetry is why the Security-focused ranking maximises recall rather than accuracy.

**2. False positives cause alert fatigue.**
`catboost` has a binary FPR of 0.00063. That sounds negligible, but against 621,680 benign test flows it is **389 false alerts**. An analyst who cannot triage that volume starts ignoring the queue, at which point the true positives are missed too -- so an unbounded FPR destroys detection indirectly. The Security ranking therefore caps binary FPR at 1.0% rather than leaving it free.

**3. Class imbalance makes accuracy misleading.**
`catboost` reports accuracy 0.9951 against a majority-class baseline of 0.8296 -- a lift of only +0.1656. Its weakest attack class is **Bot**, at recall 0.1607: 491 of 585 test flows missed. Accuracy hides this completely because the class is a rounding error in the total, which is exactly why macro-F1 and per-class recall decide this comparison.

**4. Inference speed is a deployment constraint, not a detail.**
p95 latency spans 5.68 ms (`catboost`) to 5.68 ms (`catboost`) per 1,000-flow batch -- a 1.0x spread, or 194,352 versus 194,352 flows/s. Artifact size ranges 2.5 MB to 2.5 MB. A sensor that cannot keep pace with the link drops flows, and a dropped flow is a false negative by another name.

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
