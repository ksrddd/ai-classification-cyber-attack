# SHAP pre-registration

Written before running SHAP. Date: 2026-09-26.

Changes made before any real run (same day): attack side uses "either
corpus", adversarial model settings written out, and Part B sample cap
changed from 2,000 per group to 1,000 per class.

Change 2 (same day, after a code test on small samples, before any real
run): LightGBM now uses `TreeExplainer` in `tree_path_dependent` mode
instead of interventional. In the test, one interventional explanation of
2,000 rows took 9 minutes even with the background cut to 100 rows, so the
full plan would take close to 20 hours. Path-dependent needs no background.
Logistic regression keeps the interventional `LinearExplainer` with all 200
background rows (shap had been silently cutting it to 100). Kappa is still
computed per model against that model's own permutation roles, so the two
models using different SHAP modes does not affect the Part A decision.

Change 3 (same day, same code test): a class in Part B also needs at least
30 caught rows in every seed (the protocol's `MEASURABLE_MIN`). In the test,
Infiltration was judged with only 10 caught rows. 2017 has 36 Infiltration
rows in total, so Infiltration will not be judged in the real run.

## Why

`dataset_vs_attack_importance.py` gives each of the 60 features a role
(fingerprint only, attack only, entangled, neither) using permutation
importance. Permutation importance has a known problem with redundant
features: if two columns hold the same information, shuffling one of them
does not hurt the model, so both get a score near zero. Adversarial
validation showed the dataset signal is spread over many columns, so this
problem likely affects our role table.

SHAP shares the credit between redundant columns and can also explain single
rows. We want to use it for two things:

1. Check whether the role table still holds with SHAP.
2. Look at the attacks the model misses after transfer, and see which
   features push them toward Benign.

## Part A: check the role table

Setup is the same as the permutation run: the 60 features in
`adversarial_validation/findings.json`, seeds 42-46, 70/30 stratified split,
`Ids2018Preprocessor` fit on the train part, LightGBM and logistic regression.

- LightGBM: `TreeExplainer` with `feature_perturbation="tree_path_dependent"`
  (see change 2 above)
- Logistic regression: `LinearExplainer` (interventional), background of
  200 rows from the train part
- Explained rows: 2,000 per seed, stratified from the test part

Importance of a feature = mean |SHAP| summed over classes, divided by the
total over all 60 features (a share). The dataset side uses the same
measurement on the adversarial LightGBM, per shared class. A feature counts
for the dataset side if it passes in any class, same as the permutation run.

A feature "carries" a task if its share is at least 1/60 on the seed average
and in at least 4 of 5 seeds. (Permutation used a noise-floor rule, but SHAP
values are never negative, so that rule does not work here.) For the attack
side a feature counts if it carries in either corpus, same as the permutation
run.

The adversarial LightGBM is built exactly as in `adversarial_validation.py`
(same parameters, balanced draw of up to 20,000 rows per side, raw columns,
classes with at least 500 rows per side).

Comparison with `feature_roles.csv`: Cohen's kappa of the roles over the 60
features, one value per model. We also report Spearman correlation of the
two attack-importance rankings, but do not use it for the decision.

Decision:

- Confirmed: kappa >= 0.60 for both models.
- Not confirmed: kappa < 0.40 for either model. We then list the features
  that changed role and check if they are in the duplicate groups
  (`duplicate_representatives`) or have |correlation| >= 0.9 with another
  feature.
- Partial: anything else.

0.60 and 0.40 are the usual kappa levels for "substantial" and "moderate".

## Part B: missed attacks after transfer

Model: LightGBM set up the same way as `run_crossdataset.py` (77 features,
7 classes, seeds 42-46, chronological split). Main direction is 2017 -> 2018.
2018 -> 2017 is reported too but not used for the decision.

SHAP values can be added, so for duplicate columns we sum them into their
representative feature. This puts Part B on the same 60 features as Part A.

Row groups (max 1,000 rows per class per group per seed, so small classes
are not squeezed out of the sample):

- Missed: target test rows that are an attack but predicted Benign.
- Caught: source test rows of the same class that are predicted correctly.

For each row we take the SHAP values of the Benign output. F = features with
role "fingerprint only" or "entangled" for LightGBM from Part A. We compute

    s = sum of positive SHAP over features in F / sum of positive SHAP over all features

and average it over the rows in each group. Only positive values count,
because we want to know what pulls a row toward Benign.

If Part A is not confirmed, we compute s with both role tables and report both.

Decision (only classes with at least 100 missed rows and 30 caught rows in
every seed):

- Fingerprint causes the misses: s(missed) - s(caught) >= 0.10 in all 5
  seeds, in most of the classes.
- Not supported: the difference is <= 0.02 in most of the classes.
- Partial: anything else.

0.10 is the same gap the within-dataset control used.

## Expected result

Part A: partial. We expect some LightGBM features to move from "neither" to
"fingerprint only" or "entangled", because redundancy hides them from
permutation. Logistic regression should change less.

Part B: fingerprint causes the misses, weakest for Brute Force, since it is
the only attack class that partly transfers.

## Not included

- MLP and stacking: Kernel SHAP is slow and noisy, and they were not in the
  permutation run.
- Random forest, XGBoost, CatBoost: can be added later with the same rules.
- Binary mode and distribution alignment: separate work.

## Notes

- `protocol_v2` does not save its models, so Part B retrains LightGBM.
- `shap` is in `requirements.txt` but not installed in `.venv` yet.
