# SHAP round 3 pre-registration (three more tree models)

Written before running round 3. Date: 2026-09-27.

Rounds 1 and 2 used only LightGBM (and logistic regression in Part A).
Their results stay as they are. Round 3 asks whether the round 2 answer is
a LightGBM thing or holds for the other tree models in the protocol.

## Why

Round 2 found no evidence that the fingerprint features push missed attacks
toward Benign, but that was one model. Random forest, XGBoost and CatBoost
are also in the protocol and also fail to transfer. If one of them does lean
on the fingerprint features for its misses, the round 2 conclusion only holds
for LightGBM.

## Models and SHAP

Random forest, XGBoost, CatBoost, built and fitted exactly as in
`run_crossdataset.py` (same `build_model`, `seed_model`, `fit_model`,
`Ids2018Preprocessor`).

- Random forest and XGBoost: `TreeExplainer`, `tree_path_dependent`, same as
  LightGBM in rounds 1 and 2.
- CatBoost: CatBoost's own `get_feature_importance(type="ShapValues")`.
  `shap.TreeExplainer` crashes (access violation) when it reads a CatBoost
  model on this machine (shap 0.52.0, catboost 1.2.10). CatBoost's own
  values are also exact tree SHAP; in a test on 200 rows they added up to the
  raw model output within 2e-14.

Timing test (2017, seed 42, 200 rows): XGBoost 0.3 s, CatBoost 0.6 s,
random forest 15.6 s.

## Part B (the decision)

Same as round 2 in every setting: 60 representative features, 7 classes,
chronological split, seeds 42-46, max 1,000 rows per class per group per
seed, Benign output, positive SHAP only, missed = target attacks predicted
Benign, caught = source attacks predicted correctly. Main direction
2017 -> 2018; 2018 -> 2017 is reported but not judged.

F sets are the same three as round 2 (they come from the dataset side only,
so they do not depend on the attack model): consensus (13, main), SHAP (26),
permutation (18).

Null: 2,000 random feature sets of the same size, drawn with seed 2026. The
generator is restarted for each model, so every model is tested against the
same random sets, and the consensus draws are the same ones round 2 used.

Per model, the rule is the same as round 2:

- Class supported: G > 0 and p < 0.05 / (eligible classes for that model).
- Class not supported: p >= 0.05. In between: unclear.
- Model: "fingerprint causes the misses" if most eligible classes are
  supported, "not supported" if most are not supported, otherwise partial.

Eligible classes (at least 100 missed and 30 caught rows in every seed) are
worked out per model, because each model misses different rows.

Overall for round 3 (consensus set):

- Fingerprint causes the misses: at least 2 of the 3 models say so.
- Not supported: at least 2 of the 3 models say not supported.
- Partial: anything else.

## Part A (description only, no decision)

There are no permutation roles for these three models, so kappa against
permutation (the round 1 rule) cannot be computed. Instead, for each model:

- attack-side SHAP shares on the same setup as round 1 (60 features, seeds
  42-46, 70/30 stratified split, 2,000 explained rows per seed, both corpora);
- "carries" with the same rule as round 1 (share >= 1/60 on the seed mean
  and in 4 of 5 seeds, either corpus);
- roles using the round 1 SHAP dataset side (`shap/shap_roles.csv`);
- Cohen's kappa of these roles against the LightGBM SHAP roles, and Spearman
  of the attack shares against LightGBM's.

This only shows whether the tree models agree on which features carry the
attack task. It does not change any round 1 verdict.

## Expected result

Part B: not supported for all three models, as for LightGBM. Random forest
is the one most likely to differ, because bagging with `max_features="sqrt"`
spreads importance over more features.

Part A: kappa against LightGBM around 0.5 to 0.7 for XGBoost and CatBoost
(both boosting), lower for random forest.

## Not included

- Permutation importance for the three new models.
- MLP, logistic regression and stacking in Part B.
- Changing any round 1 or round 2 threshold or verdict.
