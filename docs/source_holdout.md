# Deterministic source-held-out evaluation

The full-training research split is defined by
`configs/splits/source_holdout_v3_full_70_30.json`. It keeps capture files out
of more than one role, so flows from one attack window cannot land in both
training and final test.

V3 retains all 4,084,201 rows from the locked test sources and selects
9,529,802 training rows, giving an integer-exact 70/30 split over 13,614,003
rows. Every available attack row from a training source is kept; only BENIGN
is capped, using stable `_row_hash` ordering. There is no calibration
partition, and the locked test set never selects a threshold.

The older `source_holdout_v2_70_30.json` remains supported only so the checked-in
`local_300k_70_30` run can be reproduced and audited. Do not overwrite V2 or
attribute a new full-corpus run to it.

Validate the manifest before a run:

```powershell
.\.venv\Scripts\python.exe main.py --stage audit
```

Train with provenance metadata and deterministic train-only quotas:

```powershell
.\.venv\Scripts\python.exe main.py --stage train `
  --split-manifest configs\splits\source_holdout_v3_full_70_30.json `
  --profile overnight --model rf --skip-tuning
```

The preflight scans only `Label` and `source_file` from the cleaned cache and
rejects delivery if the selected counts differ from 9,529,802/4,084,201 or if
the train fraction differs from 70% by more than `1e-6`.

The final test sources are never subsampled or used to choose a model. The
Heartbleed class has only 11 training rows from one capture and PortScan has no
independent test source; their results must therefore be marked exploratory
rather than interpreted as independent generalization evidence.
