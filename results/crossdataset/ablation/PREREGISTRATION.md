# Ablation pre-registration

Written before the runs, so the decision rule cannot be chosen after seeing the
numbers. With 60 features, five seeds and two split modes, something will always
look like it moved.

## Question

Does removing the features that adversarial validation finds most
dataset-identifying improve cross-dataset transfer?

## Design

* Drop the top-K features from `adversarial_validation/ranking.csv`, K in {5, 10, 20}.
  K=0 is the `protocol_v2` run already on disk.
* Each dropped feature is expanded to its **exact duplicate group**. Dropping
  `Total Backward Packets` while keeping `Subflow Bwd Packets` would remove a
  name rather than a measurement.
* `ECE Flag Count` is dropped alongside `RST Flag Count` at every K>=2. The two
  are identical on every 2017 row and differ on 1.1e-5 of 2018 rows, which fell
  just outside the rho>=0.9999 duplicate threshold. Leaving it in would let the
  dropped signal walk straight back in and make the ablation a strawman.
* Models: `lightgbm`, `random_forest`. Modes: chronological and random.
  Seeds 42-46. Everything else identical to `protocol_v2`.

## Decision rule

Ablation **helps** only if, at some K, the mean `recovered_pct` across both
models and both transfer directions rises by **at least 5 percentage points**
against K=0, and that rise exceeds **twice the seed-to-seed standard deviation**.

`recovered_pct`, not raw transfer F1. Transfer that rises because the
within-dataset ceiling fell toward it is not an improvement, and the raw number
cannot tell the two apart.

## Prediction

It will not help. Adversarial validation separates the corpora at AUC >= 0.9993
inside every shared class, and removing the top 40 of 60 features leaves that
AUC at 0.985-1.000 -- the fingerprint is carried redundantly by nearly the whole
feature space, so there is no small set whose removal can take it away.

Recorded before any ablation run: 2026-09-11.
