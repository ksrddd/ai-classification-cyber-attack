# Inference Workflow

## CLI: batch prediction on a CSV

```powershell
python main.py --stage predict `
    --input  C:\path\to\my_flows.csv `
    --output C:\path\to\predictions.csv `
    --model  rf
```

`--model` accepts the same aliases as `--stage train`:
`rf`, `xgb`, `lgbm`, `cat`, `nn`, `lr`, or any canonical name. Default:
`random_forest`.

`--output` is optional -- if omitted, the result lands at
`data/processed/predictions_<input_stem>.csv`.

## Dashboard: interactive upload

```powershell
streamlit run dashboard/app.py
```

Open `Predict New CSV` from the sidebar, pick a model, drop a CSV. The
page:

1. Validates the schema against `data/processed/feature_names.json`.
   Mismatch -> friendly error with the missing-column list.
2. Runs `predict_dataframe(df, model_name=...)`.
3. Renders the head of the prediction table + a per-class bar chart.
4. Offers a one-click CSV download of the full result.

## What the predictions CSV contains

| Column | Type | Meaning |
|--------|------|---------|
| `predicted_label` | string | mapped class label (`BENIGN`, `DoS`, ... or `Normal`/`Attack`) |
| `true_label_raw` | string | the raw `Label` column from the input, if present (handy for QA) |
| `proba_<class>` | float | probability assigned to each class (one column per class) |
| `max_proba` | float | max of the per-class probabilities -- a confidence proxy |

If the underlying estimator doesn't expose `predict_proba` (rare --
only LinearSVC etc.) the probability columns are silently omitted.

## CSV schema requirements

The input CSV needs every feature column listed in
`data/processed/feature_names.json` (78 columns from CICIDS minus the
dropped duplicate). Extras (e.g. a raw `Label` column for QA) are
silently ignored. Column names are auto-stripped of CICIDS's leading
whitespace.

If a row has `Inf` or `NaN` in a feature column, the inference layer
replaces it with 0 before running the model. This mirrors the
training-time cleaning step rather than dropping the row -- at
inference, we never want to silently swallow a record.

## Programmatic API

```python
from pathlib import Path
from src.inference.predictor import predict_csv, predict_dataframe
import pandas as pd

# From a file:
result = predict_csv(
    input_csv=Path("my_flows.csv"),
    model_name="random_forest",
    output_csv=Path("predictions.csv"),
)
print(result.validation.message)
print(result.predictions.head())

# From an in-memory DataFrame:
df = pd.read_csv("my_flows.csv", encoding="latin-1")
result = predict_dataframe(df, model_name="xgboost")
print(result.predictions["predicted_label"].value_counts())
```

## Caching

The Streamlit pages share an in-process LRU cache for loaded models +
the label encoder. Toggle the model picker to switch models without
re-loading; reload the app to clear the cache.

`src.inference.predictor.clear_cache()` clears it programmatically.
