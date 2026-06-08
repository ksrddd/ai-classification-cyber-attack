# SHAP Explanation

How and why we explain predictions.

## What SHAP gives you

SHAP (SHapley Additive exPlanations) assigns each input feature a value
that says: *"this much of the model's prediction came from this feature."*
The values are additive across features for any single prediction and
have a solid theoretical foundation (Shapley values from cooperative
game theory).

For our multiclass problem, each prediction has an *n-class*
explanation -- one SHAP value per (feature, class) pair.

## Tree models vs MLP

| Model | Explainer | Compute time | Notes |
|-------|-----------|--------------|-------|
| Random Forest | `TreeExplainer` | seconds | exact |
| XGBoost | `TreeExplainer` | seconds | exact |
| LightGBM | `TreeExplainer` | seconds | exact |
| CatBoost | `TreeExplainer` | seconds | exact |
| MLPClassifier | `KernelExplainer` | minutes | approximate, sampled |

`is_tree_based` in `src/features/pipeline.py` routes the analyzer to the
right backend automatically.

For MLP, `analysis_samples` and `background_samples` in
`config.yaml::shap` keep the runtime sane:

```yaml
shap:
  background_samples: 200
  analysis_samples: 1000
  top_k_features: 10
```

## Outputs

For each model under `results/shap/<model>/`:

| Artefact | What it shows |
|----------|---------------|
| `summary_<class>.png` | Beeswarm plot per class -- features ranked by mean abs SHAP, dot colour = feature value |
| `summary_bar.png` | Mean absolute SHAP per feature across all classes |
| `top_features.json` | Machine-readable top-K overall + per-class |

A cross-model narrative is written to `results/shap/shap_report.md` so
the dashboard has a single Markdown surface to render.

## How to read a Top-K table

Example (illustrative -- actual values come from training):

```
Class: DoS
Rank | Feature              | mean |SHAP| | Direction
-----+----------------------+-------------+--------------------------------
1    | Flow Bytes/s         | 0.42        | High flow rate -> DoS more likely
2    | Total Fwd Packets    | 0.31        | More packets -> DoS more likely
3    | Flow Duration        | 0.18        | Short bursts -> DoS more likely
4    | Fwd Packet Length Mean | 0.12      | Uniform sizes -> DoS more likely
5    | SYN Flag Count       | 0.09        | High SYN count -> DoS more likely
```

This is the kind of evidence the senior-project panel asks for. The
`shap_report.md` narrates the per-class top-3 in plain English.

## Sampling strategy

TreeExplainer is exact, but compute time still scales with
`n_samples x n_trees x n_features`. We compute on a stratified
`analysis_samples`-row subset of the test set -- 1000 rows by default,
adjustable in config. Large enough for stable summary plots, small
enough to finish in under a minute on a laptop CPU.
