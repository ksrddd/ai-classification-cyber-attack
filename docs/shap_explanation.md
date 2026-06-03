# SHAP Explanation

How and why we explain predictions.

## What SHAP gives you

SHAP (SHapley Additive exPlanations) assigns each input feature a value
that says: *"this much of the model's prediction came from this feature."*
The values are additive across features for any single prediction, and
they have a solid theoretical foundation (Shapley values from cooperative
game theory).

For our 4-class problem, each prediction has a 4-dimensional explanation
— one SHAP value per (feature, class) pair.

## Why TreeExplainer, not KernelExplainer

| Aspect | TreeExplainer | KernelExplainer |
|--------|---------------|-----------------|
| Applicable to | tree models (RF, XGBoost, …) | any model |
| Compute time | O(trees × leaves) — fast | O(2ⁿ) approximation — slow |
| Result | exact | approximate |
| Verdict for this project | use on Random Forest | skip; too slow on CPU |

So per ADR-007 we explain the Random Forest only. LR and MLP get
metrics but no SHAP; if the panel asks why, the answer is: "exact, fast,
defensible for the tree model; would be hours of CPU on the others."

## Outputs

| Artifact | What it shows |
|----------|---------------|
| `results/shap/summary_<class>.png` | Beeswarm plot per class — features ranked by mean abs SHAP, dot color = feature value |
| `results/shap/bar.png` | Mean absolute SHAP per feature across all classes |
| `results/shap/force_<idx>.png` | Per-instance force plot for top-K interesting samples |
| `reports/shap_report.md` | Narrative — Top-5 features per attack class with brief explanation |

## How to read a Top-5 table

Example (illustrative — actual values come from Phase 9)::

    Class: DoS Hulk
    Rank | Feature              | mean |SHAP| | Direction
    -----+----------------------+-------------+-----------
    1    | Flow Bytes/s         | 0.42        | High flow rate ↑ DoS probability
    2    | Total Fwd Packets    | 0.31        | More packets ↑ DoS probability
    3    | Flow Duration        | 0.18        | Short bursts ↑ DoS probability
    4    | Fwd Packet Length Mean | 0.12      | Uniform sizes ↑ DoS probability
    5    | SYN Flag Count       | 0.09        | High SYN count ↑ DoS probability

That table is the kind of evidence the panel will ask for. The
`shap_report.md` will narrate it sentence by sentence.

## Sampling strategy

TreeExplainer is exact but its time scales with `n_samples × n_trees ×
n_features`. We compute on a **stratified 500-sample subset** of the
test set — large enough for stable summary plots, small enough to finish
in under a minute on a laptop CPU.

If a deeper analysis is needed for the defence, the sample size is a
`config.yaml::shap.analysis_samples` knob.
