# Training Workflow

## End-to-end on the full corpus

```powershell
# 1. (one-time) Get the data into data/raw/.
#    See docs/dataset_preparation.md.

# 2. Exploratory data analysis -- writes plots + summary JSON.
python main.py --stage eda

# 3. Preprocess -- clean + label-map + split + persist to parquet.
python main.py --stage preprocess

# 4. Train every enabled model in config.yaml::models.
python main.py --stage train

# 5. Evaluate on the held-out test split + emit the comparison report.
python main.py --stage evaluate

# 6. SHAP explainability for every trained model.
python main.py --stage explain

# All five steps in one call:
python main.py --stage all
```

## One model at a time

```powershell
python main.py --stage train --model rf      # alias for random_forest
python main.py --stage train --model xgb     # alias for xgboost
python main.py --stage train --model lgbm    # alias for lightgbm
python main.py --stage train --model cat     # alias for catboost
python main.py --stage train --model nn      # alias for mlp

# Skip GridSearchCV / RandomizedSearchCV (fast iteration).
python main.py --stage train --model rf --skip-tuning
```

## Switching classification mode

In `config.yaml`:

```yaml
classification:
  mode: "binary"    # or "multiclass"
```

Then re-run preprocessing + training (the saved parquet files are
mode-specific because the encoded label column differs).

## Subsample for laptop-scale runs

The full CICIDS corpus is ~2.8M rows. To prototype on a 16 GB laptop:

```yaml
data:
  subsample_n: 300000   # stratified on the raw label column
```

Set to `null` to use the full corpus. Subsampling preserves class
proportions and guarantees at least one row per class even for rare
classes like Heartbleed.

## Hyperparameter tuning

Toggle in `config.yaml`:

```yaml
tuning:
  enabled: true
  strategy: "random"      # "grid" or "random"
  random_n_iter: 20       # ignored when strategy = "grid"
  verbose: 1
```

Per-model grids live under `models.<name>.grid` in `config.yaml`.

## Outputs

After a full run:

| Path | What |
|------|------|
| `data/processed/train.parquet`, `val.parquet`, `test.parquet` | the three splits |
| `data/processed/feature_names.json` | feature column order |
| `data/processed/label_classes.json` | label encoder classes |
| `models/label_encoder.joblib` | fitted `LabelEncoder` |
| `models/<name>.joblib` | fitted Pipeline per model |
| `results/metrics/<name>_val.json` | per-model val metrics (after tuning) |
| `results/metrics/<name>_test.json` | per-model test metrics |
| `results/metrics/<name>_cv_results.csv` | top-10 CV rows from the tuner |
| `results/metrics/model_comparison.csv` + `.png` | cross-model ranking |
| `reports/model_comparison.md` | narrative ranking + per-class tables |
| `results/figures/confusion_matrix_<name>.png` | normalized confusion matrix per model |
| `results/shap/<name>/*` | SHAP summary plots + top-features JSON |
| `results/shap/shap_report.md` | cross-model SHAP narrative |
| `logs/pipeline.log` | rolling log file |

## Reproducibility

Every run uses `RANDOM_STATE=42`. Stratified splits, stratified
subsampling, and stratified K-fold all consume the same seed. Re-running
on the same dataset + config should produce identical artefacts
(modulo floating-point non-determinism in CatBoost on Windows).
