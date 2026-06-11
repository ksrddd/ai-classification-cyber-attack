# Testing Guide · คู่มือทดสอบทั้งโปรเจกต์

> **TH:** คู่มือทดสอบครบทุกชั้น ตั้งแต่ environment, unit test, ไปจนถึง end-to-end + dashboard
> **EN:** Complete testing guide from environment setup through unit tests, end-to-end pipeline, and dashboard.

---

## 0. เปิด Terminal · Open Terminal

**TH:** เปิด PowerShell (หรือ cmd) แล้ว `cd` เข้าโปรเจกต์ + activate venv ทุกครั้งก่อนเริ่ม
**EN:** Open PowerShell (or cmd), `cd` into the project root, and activate the venv every time before working.

```powershell
cd C:\Users\ks\Documents\GitHub\cyber_attack_classification
.\.venv\Scripts\Activate.ps1
```

**ผลที่คาดหวัง · Expected:** เห็น `(.venv)` นำหน้า prompt · See `(.venv)` prefix on prompt.

ครั้งแรกถ้าเจอ execution policy error · First-time execution policy error:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

For macOS / Linux:

```bash
cd ~/cyber_attack_classification
source .venv/bin/activate
```

---

## 1. Environment Sanity Check · ตรวจ environment

**TH:** ตรวจว่า Python + dependencies ติดตั้งครบ
**EN:** Verify Python and all dependencies are installed correctly.

```powershell
# Check Python version (need 3.10+)
python --version

# Check all critical packages
python -c "import sklearn, xgboost, lightgbm, catboost, shap, streamlit, pandas; print('ALL OK')"

# Verify model registry loads
python -c "from src.models.registry import MODEL_CLASSES; print(list(MODEL_CLASSES))"
```

**คาดหวัง · Expected:**

```
Python 3.10.x or higher
ALL OK
['random_forest', 'xgboost', 'lightgbm', 'catboost', 'mlp', 'logistic_regression']
```

---

## 2. Unit Tests (pytest) · ทดสอบหน่วยย่อย

**TH:** รันชุดทดสอบทั้งหมด — ต้องผ่าน 61 ตัว
**EN:** Run the full test suite — should pass all 61 tests.

### 2.1 รันทั้งหมด · Run all

```powershell
pytest --no-cov
```

**คาดหวัง · Expected:** `61 passed in ~5-15s`

### 2.2 รันแบบสั้น (quiet mode) · Quiet mode

```powershell
pytest -q --no-cov
```

### 2.3 รันเฉพาะไฟล์ · Run specific file

```powershell
pytest tests/test_models.py --no-cov           # only models
pytest tests/test_preprocessing.py --no-cov    # only preprocessing
pytest tests/test_config.py --no-cov           # only config
```

### 2.4 หยุดที่ fail แรก · Stop at first failure

```powershell
pytest -x --no-cov
```

### 2.5 Verbose + แสดง local variables · Verbose with local vars

```powershell
pytest -vv -l
```

### 2.6 Coverage report

```powershell
pytest --cov=src --cov-report=term-missing --cov-report=html:results/coverage
# Open the HTML report (Windows)
start results/coverage/index.html
# macOS
open results/coverage/index.html
# Linux
xdg-open results/coverage/index.html
```

---

## 3. Code Quality · ตรวจคุณภาพโค้ด

**TH:** ตรวจ lint + format
**EN:** Lint + format check.

```powershell
# Lint
ruff check src tests

# Auto-fix safe issues
ruff check --fix src tests

# Format
black src tests

# Check format without modifying
black --check src tests
```

**คาดหวัง · Expected:** ไม่มี error · No errors.

---

## 4. Synthetic Data Smoke Test · ทดสอบด้วยข้อมูลสังเคราะห์

**TH:** ทดสอบ pipeline ได้โดยไม่ต้องมี CICIDS จริง — ใช้สำหรับ first-time setup
**EN:** Test the pipeline without needing the real CICIDS dataset — useful for first-time setup.

### 4.1 สร้างข้อมูลสังเคราะห์ · Generate sample data

```powershell
python scripts/generate_sample.py --rows 2000
```

**คาดหวัง · Expected:** `Wrote 2,020 rows x 79 cols to data\sample\synthetic_cicids.csv`

### 4.2 รัน EDA บนข้อมูลสังเคราะห์ · Run EDA on synthetic

```powershell
python main.py --stage eda --raw-dir data/sample
```

**คาดหวัง · Expected:** ลงท้ายด้วย `EDA summary written to ...eda_summary.json`
**Expected:** Ends with `EDA summary written to ...`

### 4.3 ทำ preprocess + train + evaluate · Preprocess + train + evaluate

```powershell
python main.py --stage preprocess --raw-dir data/sample
python main.py --stage train --model rf --skip-tuning
python main.py --stage evaluate --model rf
```

> **TH:** ข้อมูลสังเคราะห์ใช้สำหรับทดสอบไปป์ไลน์เท่านั้น **อย่านำผลไปรายงานในโปรเจกต์จริง**
> **EN:** Synthetic data is for pipeline testing only — **do NOT report metrics computed on synthetic data**.

---

## 5. Real Dataset Preparation · เตรียมข้อมูลจริง

**TH:** ตรวจว่ามี CICIDS2017 CSVs ครบ 8 ไฟล์ใน `data/raw/`
**EN:** Verify all 8 CICIDS2017 CSVs are present in `data/raw/`.

```powershell
# List files in data/raw/
ls data/raw/*.csv
```

**คาดหวัง · Expected:** 8 ไฟล์ · 8 files:

```
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
Friday-WorkingHours-Morning.pcap_ISCX.csv
Monday-WorkingHours.pcap_ISCX.csv
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv
```

**ถ้ายังไม่มี · If missing:** ดูวิธี extract ใน [`dataset_preparation.md`](dataset_preparation.md)

---

## 6. Quick Real-Data Smoke Test · ทดสอบเร็วกับข้อมูลจริง

**TH:** ทดสอบ pipeline กับข้อมูลจริงบน subsample เล็ก ๆ ใช้เวลาประมาณ 3-5 นาที
**EN:** Quick end-to-end smoke test on a small subsample (3-5 minutes).

### 6.1 ลดขนาด config ชั่วคราว · Temporarily shrink config

แก้ `src/config/config.yaml`:

```yaml
data:
  subsample_n: 50000          # ลดเป็น 50K ชั่วคราว · temp shrink to 50K
tuning:
  enabled: false              # ปิด tuning · skip tuning
```

### 6.2 รันทั้งหมด · Run all stages

```powershell
python main.py --stage all
```

**คาดหวัง · Expected:** ทั้งหมด 5 stages เสร็จ + เห็น metric ของทุกโมเดล
**Expected:** All 5 stages complete + metrics printed for every model.

### 6.3 อย่าลืม restore config · Don't forget to restore config

```yaml
data:
  subsample_n: 300000         # คืนค่าเดิม · restore default
tuning:
  enabled: true
```

---

## 7. Per-Stage Testing · ทดสอบทีละ stage

### 7.1 EDA

```powershell
python main.py --stage eda
```

**Output:**

- `results/figures/class_distribution.png`
- `results/figures/missing_value_audit.png`
- `results/figures/correlation_heatmap.png`
- `results/figures/feature_distributions.png`
- `results/metrics/eda_summary.json`

### 7.2 Preprocess

```powershell
python main.py --stage preprocess
```

**Output:**

- `data/processed/train.parquet`, `val.parquet`, `test.parquet`
- `data/processed/feature_names.json`
- `data/processed/label_classes.json`
- `models/label_encoder.joblib`

**Verify:**

```powershell
python -c "import pandas as pd; df = pd.read_parquet('data/processed/train.parquet'); print(df.shape); print(df['label_encoded'].value_counts())"
```

### 7.3 Train

```powershell
# Train one model (fast)
python main.py --stage train --model rf --skip-tuning

# Train all 5 models with tuning
python main.py --stage train
```

**Output:**

- `models/<name>.joblib` (one per enabled model)
- `results/metrics/<name>_val.json`
- `results/metrics/val_summary.csv`

**Aliases · ตัวย่อโมเดล:**

| Alias | Full name |
|---|---|
| `rf` | random_forest |
| `xgb` | xgboost |
| `lgbm` | lightgbm |
| `cat` | catboost |
| `nn` | mlp |
| `lr` | logistic_regression |
| `all` | every enabled model |

### 7.4 Evaluate

```powershell
python main.py --stage evaluate
# or single model:
python main.py --stage evaluate --model rf
```

**Output:**

- `results/metrics/<name>_test.json`
- `results/metrics/classification_report_<name>.csv`
- `results/figures/confusion_matrix_<name>.png`
- `results/metrics/model_comparison.csv` + `.png`
- `reports/model_comparison.md`

### 7.5 Explain (SHAP)

```powershell
python main.py --stage explain
# or single model:
python main.py --stage explain --model rf
```

**Output (per model):**

- `results/shap/<name>/summary_bar.png`
- `results/shap/<name>/summary_<class>.png` (one per class)
- `results/shap/<name>/top_features.json`
- `results/shap/shap_report.md`

### 7.6 Predict

```powershell
python main.py --stage predict --input my_traffic.csv --output preds.csv --model lgbm
```

**Output:** CSV with columns:

- `predicted_label`
- `true_label_raw` (if input CSV has a `Label` column)
- `proba_<class>` (one column per class)
- `max_proba` (highest probability — confidence proxy)

---

## 8. Dashboard Testing · ทดสอบ Dashboard

### 8.1 Launch

```powershell
streamlit run dashboard/app.py
```

**TH:** Browser จะเปิดอัตโนมัติที่ `http://localhost:8501`
**EN:** Browser opens automatically at `http://localhost:8501`.

ปิด server · Stop server: **Ctrl+C**

### 8.2 Checklist ทุกหน้า · Per-page checklist

| หน้า · Page | ตรวจ · Check |
|---|---|
| **Landing (app)** | Hero gradient + 4 KPI cards + status pills + 6 nav cards |
| **1. Dataset Overview** | KPI row + Plotly bar chart (per-class colors) + class count table |
| **2. EDA** | 4 tabs (class / missing / correlation / distributions) + warning pills if missing/Inf |
| **3. Model Performance** | Model selector + radar chart + per-class bar chart + confusion matrix |
| **4. Model Comparison** | Rank selector + leaderboard with medals + grouped bar + download CSV |
| **5. SHAP** | Model + class selector + overall bar + per-class bar + beeswarm expander |
| **6. Predict New CSV** | Model selector + file upload + validation pill + bar chart + table + download |

### 8.3 ทดสอบหน้า 6 (Predict) · Test Page 6

**TH:** ใช้ test CSV ที่ generate จาก synthetic data หรือ subset จาก CICIDS จริง
**EN:** Upload either the synthetic CSV or a subset from a real CICIDS file.

```powershell
# Generate sample CSV first
python scripts/generate_sample.py --rows 100 --out data/sample/test_upload.csv
```

จากนั้นใน dashboard · Then in dashboard:

1. เลือก model · Pick model
2. Upload `data/sample/test_upload.csv`
3. ดู predictions + download · View predictions + download

### 8.4 Headless launch (สำหรับ CI) · For CI

```powershell
streamlit run dashboard/app.py --server.headless true --server.port 8501
```

---

## 9. Programmatic API Testing · เทส API จากโค้ด

**TH:** ทดสอบ inference จาก Python script
**EN:** Test inference programmatically.

```python
# test_inference.py
from pathlib import Path
from src.inference.predictor import predict_csv

result = predict_csv(
    input_csv=Path("data/sample/synthetic_cicids.csv"),
    model_name="random_forest",
    output_csv=Path("test_predictions.csv"),
)
print(f"Validation: {result.validation.message}")
print(f"Predicted {len(result.predictions)} rows")
print(result.predictions["predicted_label"].value_counts())
```

```powershell
python test_inference.py
```

---

## 10. Full Test Suite Summary · สรุปการทดสอบทั้งหมด

| # | ทดสอบ · Test | คำสั่ง · Command | คาดหวัง · Expected |
|---|---|---|---|
| 1 | Environment | `python -c "import xgboost, lightgbm, catboost; print('OK')"` | `OK` |
| 2 | Unit tests | `pytest --no-cov` | `61 passed` |
| 3 | Lint | `ruff check src tests` | no errors |
| 4 | Format | `black --check src tests` | all files formatted |
| 5 | Synthetic E2E | `python main.py --stage all --raw-dir data/sample` | all stages pass |
| 6 | Real EDA | `python main.py --stage eda` | summary JSON written |
| 7 | Real preprocess | `python main.py --stage preprocess` | 3 parquet files |
| 8 | Train 1 model | `python main.py --stage train --model rf --skip-tuning` | model joblib saved |
| 9 | Train all | `python main.py --stage train` | 5 model files |
| 10 | Evaluate | `python main.py --stage evaluate` | comparison CSV + MD |
| 11 | SHAP | `python main.py --stage explain` | shap PNGs + JSON |
| 12 | Inference | `python main.py --stage predict --input ... --output ...` | predictions CSV |
| 13 | Dashboard | `streamlit run dashboard/app.py` | all 7 pages load |

---

## 11. Troubleshooting · การแก้ปัญหา

| ปัญหา · Problem | สาเหตุ · Cause | แก้ · Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'pandas'` | ไม่ได้ activate venv | `.\.venv\Scripts\Activate.ps1` |
| `ModuleNotFoundError: 'xgboost'/'lightgbm'/'catboost'` | dep ไม่ครบ | `pip install -r requirements.txt` |
| `No module named 'tabulate'` | dep หายไป | `pip install tabulate` |
| `No CSV files found in data/raw/` | ยัง extract CICIDS ไม่เสร็จ | ดู `dataset_preparation.md` |
| `LabelEncoder is missing N expected classes` (warning) | subsample เล็กไม่ครบทุกคลาส | เพิ่ม `subsample_n` ใน config |
| Memory error ตอน train | RAM น้อยเกินไป | ลด `subsample_n` ใน config |
| `Glyph 150 (\x96) missing from font` (warning) | font Arial ไม่มี glyph | warning เฉย ๆ ไม่กระทบ |
| Streamlit port already in use | port 8501 ถูกใช้แล้ว | `streamlit run dashboard/app.py --server.port 8502` |
| `Repository not found` ตอน push | GitHub repo ยังไม่ได้สร้าง | สร้างที่ <https://github.com/new> |
| Password rejected ตอน push | GitHub ไม่รับ password ปกติ | ใช้ Personal Access Token แทน |

---

## 12. CI-Style One-Liner · เทสครบในคำสั่งเดียว

**TH:** สำหรับ CI หรือเช็คก่อน push
**EN:** For CI or pre-push check.

PowerShell:

```powershell
ruff check src tests; if ($?) { black --check src tests; if ($?) { pytest --no-cov -q } }
```

bash:

```bash
ruff check src tests && black --check src tests && pytest --no-cov -q
```

**คาดหวัง · Expected:** ผ่านทั้ง 3 ขั้น · all 3 checks pass.

---

## 13. Pre-Push Checklist · เช็คก่อน push GitHub

| ขั้น · Step | คำสั่ง · Command | ผ่านเมื่อ · Pass when |
|---|---|---|
| 1 | `git status` | working tree clean |
| 2 | `pytest --no-cov` | 61 passed |
| 3 | `ruff check src tests` | 0 errors |
| 4 | `black --check src tests` | 0 reformat needed |
| 5 | `du -sh data/raw models results` | only gitignored dirs are big |
| 6 | `git log --oneline -3` | commit message ตรงจริง · matches reality |
| 7 | `git remote -v` | URL ชี้ไป GitHub repo ที่สร้างไว้แล้ว · points to existing repo |
| 8 | `git push -u origin main` | success |

---

> See also: [`dataset_preparation.md`](dataset_preparation.md), [`training_workflow.md`](training_workflow.md), [`inference_workflow.md`](inference_workflow.md), [`evaluation.md`](evaluation.md), [`shap_explanation.md`](shap_explanation.md), [`architecture.md`](architecture.md), [`ml_pipeline.md`](ml_pipeline.md), [`feature_mapping.md`](feature_mapping.md).
