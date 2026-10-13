# SHAP round 4 pre-registration (class-level shift)

Written before running round 4. Date: 2026-09-30.

Rounds 1-3 stay as they are. Round 4 asks a new question and does not
change any earlier verdict.

## Why

Rounds 2 and 3 found no evidence that the fingerprint features push missed
attacks to Benign, in any of the four tree models. From that we wrote
"what is left is that the feature values shift as a whole". That sentence
came from ruling the other explanation out. It was never tested directly.

The fingerprint set F came from adversarial validation on the whole
dataset, so it says which features tell 2017 from 2018. It does not say
which features changed *for the same attack class*. A DDoS in 2018 can look
different from a DDoS in 2017 on features that are not in F.

Round 4 tests that: are the Benign pushes in missed attacks concentrated on
the features where that attack class moved the most between the datasets?

## Setup

Same models, rows and SHAP values as round 2 (LightGBM) and round 3 (RF,
XGBoost, CatBoost). The per-row positive Benign SHAP is read from the saved
`blocks*.joblib` files. No model is refit and no SHAP is recomputed.

**Shift per feature.** For each attack class c and each of the 60
representative features f:

- rows: class c in the *train* part of the chronological split (seed 0) of
  each dataset, at most 20,000 per dataset, drawn with seed 2026
- rows with NaN or infinity in f are dropped for that feature
- both samples are put through the pooled empirical CDF (average ranks
  divided by the pooled size), so every feature is on the same 0-1 scale
- shift(f, c) = 1-Wasserstein distance between the two ranked samples
- the KS statistic is also recorded and used only to break ties

Train rows are used so that the set is not chosen from the same test rows
the SHAP values come from.

**S (main).** S_c = the 13 features with the largest shift for class c.
13 is the size of the round 2 consensus set, so the null has the same size
as in round 2. Ties: larger KS first, then feature name.

S_c with 5 and 26 features is also reported with the same null, but not
used for the decision.

**Statistic.** G = mean over seeds 42-46 of s(missed) - s(caught), where s
is the share of the positive Benign SHAP that comes from S_c. Same as
round 2.

**Null.** 2,000 random feature sets of size 13 from the 60 features,
random seed 2026, each set used for all 5 seeds. One-sided p:

    p = (1 + number of G_null >= G) / (1 + 2000)

Main direction 2017 -> 2018. 2018 -> 2017 is reported but not judged.

Eligible classes: at least 100 missed rows and 30 caught rows in every
seed, per model, same as rounds 2 and 3.

## Decision

Per class and model:

- Supported: G > 0 and p < 0.05 / (number of eligible classes).
- Not supported: p >= 0.05.
- In between: unclear.

Per model: supported if most eligible classes are supported, not supported
if most are not supported, partial otherwise.

Overall (four models: LightGBM, RF, XGBoost, CatBoost):

- Class-level shift explains the misses: at least 3 of 4 models supported.
- Not supported: at least 3 of 4 models not supported.
- Partial: anything else.

## What each outcome means

- Supported: the missed attacks are pushed to Benign mostly through the
  features where that class moved. This is direct evidence for the shift
  explanation, and it names the features alignment should work on first.
- Not supported: the Benign push is not concentrated on the most-shifted
  features either. Then the shift explanation cannot be pinned to a small
  group of features, and rounds 1-4 together say the misses are spread over
  many features. We would report that as it is and not look for another
  feature set.
- Partial: report per class and per model, no overall claim.

## Also reported (not judged)

- Overlap between each S_c and the consensus F. If they share most
  features, round 4 partly repeats round 2 and we say so.
- Spearman correlation over the 60 features between shift(f, c) and
  d(f, c) = mean over seeds of (share of f in missed - share of f in caught).
- The shift table itself (60 features x 6 classes, W1 and KS).

## Expected result

Partial. Round 2 showed that random sets of 13 features already give wide
gaps (95th percentile 0.18 to 0.33), which means missed and caught rows
differ on many features. If the shift is broad, the 13 most-shifted
features may not stand out. We expect DoS to be the most likely class to
pass, because its G was positive in all three boosting models in round 3.

## Not included

- MLP, LR and stacking.
- Binary mode.
- Changing any threshold of rounds 1-3.
