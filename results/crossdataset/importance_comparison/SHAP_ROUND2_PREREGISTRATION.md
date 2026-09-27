# SHAP round 2 pre-registration (Part B with a null)

Written before running round 2. Date: 2026-09-26.

This round was designed after seeing round 1. Round 1 results stay as they
are and are reported as they are. Round 2 exists because round 1 Part B had
two weak points, not to replace its verdict.

## Why

Round 1 Part B measured s = share of the positive Benign SHAP that comes from
the fingerprint features F, and compared missed attacks against caught
attacks. Two problems:

1. The result depended on which table defined F. SHAP roles gave 26
   features, permutation roles gave 18, and the two verdicts differed
   (partial vs not supported).
2. There was no null. A gap of +0.14 could come from almost any group of
   features of that size, and we cannot tell from round 1.

## Setup

Same as round 1 Part B: protocol LightGBM, 77 features collapsed to the 60
representatives, 7 classes, chronological split, seeds 42-46, max 1,000 rows
per class per group per seed, SHAP `tree_path_dependent`, Benign output.
Main direction 2017 -> 2018. 2018 -> 2017 is reported but not judged.

Eligible classes: at least 100 missed rows and 30 caught rows in every seed
(same as round 1).

**F (main).** Consensus set: features that carry the dataset signal in both
the permutation table (`feature_roles.csv`) and the SHAP table
(`shap/shap_roles.csv`). That is 13 features. Using only features both
methods agree on removes the choice between the two tables.

F_shap (26) and F_perm (18) are also reported with the same null, but not
used for the decision.

**Statistic.** For each class, G = mean over the 5 seeds of
s(missed) - s(caught), as in round 1.

**Null.** 2,000 random feature sets of the same size as F, drawn from the 60
features without replacement. Each random set is used for all 5 seeds, so
G_null is computed the same way as G. One-sided p-value:

    p = (1 + number of G_null >= G) / (1 + 2000)

Random seed for drawing the sets: 2026.

## Decision

For each eligible class:

- Supported: G > 0 and p < 0.05 / (number of eligible classes) (Bonferroni).
- Not supported: p >= 0.05.
- In between: unclear.

Overall:

- Fingerprint causes the misses: most eligible classes supported.
- Not supported: most eligible classes not supported.
- Partial: anything else.

## Expected result

Partial. Round 1 showed the clearest gap for DDoS with the SHAP table, but
not with the permutation table. We expect DDoS to stay above the null and
Web Attack to be not supported. DoS is hard to call.

## Not included

- Other models (RF, XGBoost, CatBoost).
- Changing any round 1 threshold. Round 1 verdicts are final.
