# SHAP within-target pre-registration

Written before running. Date: 2026-10-02.

## Why

The advisor asked for this. In rounds 1 to 3, "missed" rows came from the
target dataset and "caught" rows came from the source dataset. So any gap
between the two groups could come from the miss itself, or only from the
rows being from a different dataset. This check keeps the dataset fixed.

## Groups

Same models, split and rows as rounds 2 and 3. For each model, direction and
seed, inside the target test split and inside one attack class:

- missed: true class is this attack, model predicts Benign
- caught_tgt: true class is this attack, model predicts this attack
- caught_src: the old caught group (source test split, predicted correctly),
  kept only to split the old gap into two parts

Rows predicted as another attack class are not used.

missed and caught_src come from the round 2 (LightGBM) and round 3 (random
forest, XGBoost, CatBoost) caches. Only caught_tgt is new. The model is refit
with the same seed, and the run stops if the number of missed rows per class
does not match the cache, so the three groups come from the same model.

## Setup

Same as round 2: 60 representative features, chronological split, seeds
42-46, at most 1,000 rows per class per group per seed, Benign output,
positive SHAP only. Main feature set is consensus (13); SHAP (26) and
permutation (18) are reported too.

## Statistics

For one class and one seed, s(group) is the mean over rows of
(positive Benign SHAP on F) / (positive Benign SHAP on all features).

- G_within = s(missed) - s(caught_tgt), mean over seeds. This is the test.
- G_old = s(missed) - s(caught_src). Should match rounds 2 and 3.
- G_dataset = s(caught_tgt) - s(caught_src). Description only.

G_old = G_within + G_dataset, so this shows how much of the old gap is
"same class, different dataset" and how much is "missed vs caught".

Null: 2,000 random feature sets of the same size, seed 2026, restarted for
each model, same as round 3.

## Rules

A cell (model, direction, class) is eligible if every seed has at least 100
missed rows and at least 30 caught_tgt rows. Both directions are judged here,
because the 2017 -> 2018 direction alone has very few caught_tgt rows (most
classes are almost never caught there).

- Supported: G_within > 0 and p < 0.05 / (eligible cells for that model).
- Not supported: p >= 0.05. In between: unclear.

Per model: "miss-specific" if most eligible cells are supported, "not
supported" if most are not supported, else partial. Cells that are not
eligible are listed with their counts, because a class that is never caught
in the target cannot answer the question at all.
