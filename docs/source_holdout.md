# Deterministic source-held-out evaluation

The research split is defined by `configs/splits/source_holdout_v2_70_30.json`. It
keeps capture files out of more than one role, so flows from one attack window
cannot land in both training and final test.

There is no calibration partition: the model-fit train partition is 70% and
the locked final test partition is 30%. The source-held protocol approximates
that ratio by capture file and deliberately prioritises capture separation over
an exact row count. Threshold calibration is disabled for this protocol.

Validate the manifest before a run:

```powershell
.\.venv\Scripts\python.exe main.py --stage audit
```

Train with provenance metadata and deterministic train-only quotas:

```powershell
.\.venv\Scripts\python.exe main.py --stage train `
  --split-manifest configs\splits\source_holdout_v2_70_30.json `
  --profile overnight --model rf --skip-tuning
```

The final test sources are never subsampled or used to choose a model. The
Heartbleed class has only 11 rows from one capture and PortScan has one source;
their results must therefore be marked exploratory rather than interpreted as
independent generalization evidence.
