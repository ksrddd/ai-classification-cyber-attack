# CICIDS2017 main-run deliverable verification — 2026-11-07

## Scope and completed work

The latest report source supplied by the user was
`C:/Users/petgo/Downloads/Project_I_แก_20261015__1new/`.
The task brief was `C:/Users/petgo/Downloads/message.txt` (some text is duplicated/truncated).
The latest report already contained the requested chapter 3 scope correction,
chapter 4 CatBoost matrix and Bot discussion, six-model appendix, and appendix inclusion.
These edits were retained verbatim; no experimental values were changed.

**Case A applies.** All 32 files in the main-run bundle match the original manifest,
including the eight joblib files and seven PNGs. Verified files are already present at
`results/cicids2017_temporal_v1/`; an additional machine search or retraining is unnecessary.
The seven images in the latest report also match the manifest. No training or joblib loading was performed.
No source files were deleted or moved. Unrelated existing working-tree changes were left untouched.

## Required artifact checks

Candidate directory: `results/cicids2017_temporal_v1/`.
Expected digests: the existing `bundle_manifest.json` (unchanged).
Full expected/actual SHA-256 evidence: `artifact_verification.json`.

| File | SHA-256 result |
|---|---|
| catboost.joblib | MATCH |
| catboost_confusion_matrix.png | MATCH |
| label_encoder.joblib | MATCH |
| lightgbm.joblib | MATCH |
| lightgbm_confusion_matrix.png | MATCH |
| logistic_regression.joblib | MATCH |
| logistic_regression_confusion_matrix.png | MATCH |
| mlp.joblib | MATCH |
| mlp_confusion_matrix.png | MATCH |
| random_forest.joblib | MATCH |
| random_forest_confusion_matrix.png | MATCH |
| stacking.joblib | MATCH |
| stacking_confusion_matrix.png | MATCH |
| xgboost.joblib | MATCH |
| xgboost_confusion_matrix.png | MATCH |

## Report and slides

The four TeX files stored here are verbatim snapshots of the supplied latest report;
they are intended to be used with the remaining original report files, fonts, and images.
The six appendix images and the chapter 4 CatBoost image are committed in the canonical results directory.
The complete standalone report source and compiled deliverables are packaged under
`output/cicids2017-report/`.

The previously prepared `tmp/cicids2017-deliverables/AI_Attack_Classification_Defense_v4_Updated.pdf`
has 45 pages. Page 24 already shows the main-run matrix, Macro-F1 0.8873,
and the 491/585 Bot-to-BENIGN errors (94 correctly classified; recall 0.1607).
Its embedded image was compared with the canonical main-run PNG after RGB decoding:
all pixels match. The obsolete missing-artifacts statement is absent on that page.
This existing corrected slide PDF is included in the output package unchanged.

## Build and validation

Compiled successfully with Tectonic 0.17.0 (XeTeX), exit code 0: **79 pages**.
The list of figures contains all seven confusion matrices, and the PDF has no `??` references.
Each of the seven embedded matrix images matches the canonical PNG pixel for pixel.
CatBoost is on PDF page 49 (printed page 34); the six appendix matrices are on PDF pages 73–78
(printed pages 58–63). The scope statement is on PDF page 42 (printed page 27).
The matrix pages, scope statement, figure list, and slide 24 were rendered and visually checked.
The root bibliography completes successfully. Tectonic also attempts BibTeX on chapter-only
aux files, which report missing bibliography directives; these are harmless.
Existing fontspec/line-spacing warnings and an abstract line overfull by 2.66023pt remain.
No model training, application changes, or application test runs were needed.
See `build_verification.json` for the machine-readable checks.

## Git

Branch: `codex/finish-cicids2017-report`.
Only this task's evidence, TeX snapshots, and the seven exact PNGs are staged.
No joblib, unrelated code changes, or AI co-author trailer is included.
