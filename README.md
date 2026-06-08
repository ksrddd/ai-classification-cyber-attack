# AI-Based Cyber Attack Classification

**ระบบจำแนกประเภทการโจมตีทางไซเบอร์จาก Network Logs ด้วย Machine Learning + Explainable AI**

> Senior Project — Faculty of Information Technology, KMITL

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-61%20passed-brightgreen)](tests/)
[![Dataset](https://img.shields.io/badge/dataset-CICIDS2017-orange)](https://www.unb.ca/cic/datasets/ids-2017.html)

---

## Table of Contents · สารบัญ

1. [Overview · ภาพรวม](#1-overview--ภาพรวม)
2. [Features · ฟีเจอร์](#2-features--ฟีเจอร์)
3. [Project Status · สถานะโครงการ](#3-project-status--สถานะโครงการ)
4. [Tech Stack · เทคโนโลยีที่ใช้](#4-tech-stack--เทคโนโลยีที่ใช้)
5. [Project Structure · โครงสร้างโปรเจกต์](#5-project-structure--โครงสร้างโปรเจกต์)
6. [Installation · การติดตั้ง](#6-installation--การติดตั้ง)
7. [Dataset Preparation · การเตรียมข้อมูล](#7-dataset-preparation--การเตรียมข้อมูล)
8. [Configuration · การตั้งค่า](#8-configuration--การตั้งค่า)
9. [Quick Start · เริ่มต้นใช้งานเร็ว](#9-quick-start--เริ่มต้นใช้งานเร็ว)
10. [Pipeline Stages · ขั้นตอนของไปป์ไลน์](#10-pipeline-stages--ขั้นตอนของไปป์ไลน์)
11. [Dashboard · แดชบอร์ด](#11-dashboard--แดชบอร์ด)
12. [Inference · การพยากรณ์ผล](#12-inference--การพยากรณ์ผล)
13. [Testing · การทดสอบ](#13-testing--การทดสอบ)
14. [Results · ผลลัพธ์](#14-results--ผลลัพธ์)
15. [Architecture · สถาปัตยกรรม](#15-architecture--สถาปัตยกรรม)
16. [Troubleshooting · การแก้ปัญหา](#16-troubleshooting--การแก้ปัญหา)
17. [Development · การพัฒนา](#17-development--การพัฒนา)
18. [Out of Scope · นอกเหนือขอบเขต](#18-out-of-scope--นอกเหนือขอบเขต)
19. [License & Authors · ใบอนุญาตและผู้พัฒนา](#19-license--authors--ใบอนุญาตและผู้พัฒนา)

---

## 1. Overview · ภาพรวม

### English

This project reads CICIDS2017 network-flow records and automatically
classifies each flow as one of several attack types — like a doctor
diagnosing illness from symptoms. Existing IDS/IPS tools detect *that*
an attack happened; this system tells the operator *what kind*, so the
response can be precise.

The whole pipeline — data engineering, feature engineering, training,
evaluation, explainability, dashboard, inference — is implemented as a
real ML project: config-driven, tested, reproducible, leakage-free.

### ภาษาไทย

โปรเจกต์นี้รับข้อมูล Network Flow จากชุดข้อมูล CICIDS2017 มาจำแนกประเภท
การโจมตีทางไซเบอร์โดยอัตโนมัติ — เปรียบเหมือนหมอที่วินิจฉัยโรคจากอาการ
ระบบ IDS/IPS ทั่วไปบอกได้แค่ว่า "มีการโจมตีเกิดขึ้น" แต่ไม่บอกว่าเป็น
ประเภทไหน ระบบนี้ระบุประเภทให้ผู้ดูแลตอบสนองได้ตรงจุด

ไปป์ไลน์ทั้งหมด — Data engineering, Feature engineering, Training,
Evaluation, Explainable AI, Dashboard, Inference — ถูกพัฒนาด้วยมาตรฐาน
ML จริง: ปรับด้วย config, มี test, รันซ้ำได้, ป้องกัน data leakage

---

## 2. Features · ฟีเจอร์

| Feature | English | ภาษาไทย |
|---------|---------|---------|
| Dual classification | Binary (`Normal`/`Attack`) **and** 10-class multi-class | รองรับทั้งแบบ 2 คลาส และ 10 คลาส |
| Five ML models | Random Forest · XGBoost · LightGBM · CatBoost · MLP | โมเดล 5 แบบ ครอบคลุม tree ensemble และ neural net |
| Label normalization | Maps 15 raw CICIDS labels into a clean scheme | แปลง 15 label ดิบของ CICIDS ให้สะอาด |
| Stratified 3-way split | 60% train / 20% val / 20% test, leakage-proof | แบ่งข้อมูล 60/20/20 แบบ stratified ป้องกัน data leakage |
| Hyperparameter tuning | GridSearchCV / RandomizedSearchCV configurable | ปรับ hyperparameter ด้วย Grid/Random Search ผ่าน config |
| Full metric set | Accuracy · P/R/F1 (weighted+macro+per-class) · ROC-AUC · MCC | ครบทุก metric ที่ใช้ใน security ML |
| Explainability | SHAP TreeExplainer + KernelExplainer fallback | อธิบาย model ด้วย SHAP รองรับทั้ง tree และ MLP |
| Streamlit dashboard | 6 pages: data, EDA, performance, comparison, SHAP, predict | แดชบอร์ด Streamlit 6 หน้า |
| Schema-safe inference | Upload CSV → validate → predict + probabilities | อัปโหลด CSV ใหม่แล้วทำนายได้ พร้อมเช็ค schema |
| Test suite | 61 pytest cases + real-data smoke test | มี test 61 รายการ + smoke test กับข้อมูลจริง |
| Reproducible | Single `RANDOM_STATE=42` + stratified everything | รันซ้ำได้ผลเดิมเสมอ |
| Config-driven | All knobs in `config.yaml` — no magic numbers | ค่าทั้งหมดอยู่ใน config ไม่มีฮาร์ดโค้ดในตัว module |

### Classification Schemes · ระบบจำแนกคลาส

**Binary mode** — สำหรับการตรวจจับว่ามีการโจมตีหรือไม่
- `Normal` — BENIGN traffic (การจราจรปกติ)
- `Attack` — ทุกประเภทที่ไม่ใช่ BENIGN

**Multi-class mode** — สำหรับระบุประเภทการโจมตี (10 คลาส)

| Class | Description (EN) | คำอธิบาย (TH) | CICIDS sources |
|---|---|---|---|
| `BENIGN` | Normal background traffic | การจราจรปกติ | ทุกไฟล์ |
| `DoS` | Denial of Service attacks | การโจมตีแบบ DoS | DoS Hulk, GoldenEye, slowloris, Slowhttptest |
| `DDoS` | Distributed DoS | การโจมตีแบบกระจายจากหลายแหล่ง | DDoS |
| `PortScan` | Network port scanning | สแกนพอร์ตเพื่อหาช่องโหว่ | PortScan |
| `Bot` | Botnet C&C traffic | การสื่อสารของบอตเน็ต | Bot |
| `Web Attack` | XSS, SQL Injection, web brute force | การโจมตีเว็บแอป | Web Attack (3 sub-types) |
| `Brute Force` | Credential brute-forcing | เดารหัสผ่านบริการเครือข่าย | FTP-Patator, SSH-Patator |
| `Infiltration` | Lateral movement / infiltration | การเจาะระบบเข้าไปใน network | Infiltration |
| `Heartbleed` | OpenSSL Heartbleed exploit | ช่องโหว่ Heartbleed | Heartbleed |
| `Other` | Catch-all for unknown variants | คลาสรอง สำหรับ label ที่ไม่รู้จัก | (fallback) |

---

## 3. Project Status · สถานะโครงการ

| Phase | Description (EN) | คำอธิบาย (TH) | Status |
|-------|------------------|---------------|--------|
| 1 | Requirement analysis | วิเคราะห์ความต้องการ | done |
| 2 | Architecture design | ออกแบบสถาปัตยกรรม | done |
| 3 | Project scaffold | วาง scaffold โปรเจกต์ | done |
| 4 | Data engineering | จัดการข้อมูล + label mapping | done |
| 5 | Feature engineering | คัดเลือก + เตรียม feature | done |
| 6 | Model development | 5 model wrappers + registry | done |
| 7 | Hyperparameter tuning | GridSearch / RandomizedSearch | done |
| 8 | Evaluation | metric / confusion matrix / report | done |
| 9 | SHAP / XAI | อธิบายโมเดลด้วย SHAP | done |
| 10 | Streamlit dashboard | แดชบอร์ด 6 หน้า | done |
| 11 | Testing | pytest 61 tests + smoke test | done |
| 12 | Inference / MLOps prep | ระบบทำนายผลข้อมูลใหม่ | done |
| 13 | Documentation | เอกสาร + README นี้ | done |

---

## 4. Tech Stack · เทคโนโลยีที่ใช้

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Data | `pandas`, `numpy`, `pyarrow` |
| Classical ML | `scikit-learn`, `imbalanced-learn` |
| Gradient boosting | `xgboost`, `lightgbm`, `catboost` |
| Explainability | `shap` |
| Visualization | `matplotlib`, `seaborn` |
| Dashboard | `streamlit` |
| Persistence | `joblib`, `pyarrow` (parquet) |
| Config | `pyyaml` |
| Logging | stdlib `logging` |
| Testing | `pytest`, `pytest-cov` |
| Lint / format | `ruff`, `black` |

---

## 5. Project Structure · โครงสร้างโปรเจกต์

```
cyber_attack_classification/
├── data/
│   ├── raw/                      # CICIDS2017 CSVs (download separately)
│   ├── interim/                  # intermediate parquet files
│   ├── processed/                # train/val/test parquet + feature_names.json
│   └── sample/                   # synthetic CICIDS-shaped fixture for tests
├── docs/
│   ├── architecture.md           # module map + ADRs
│   ├── ml_pipeline.md            # end-to-end ML flow
│   ├── evaluation.md             # metric definitions + rationale
│   ├── shap_explanation.md       # SHAP setup + interpretation guide
│   ├── dataset_preparation.md    # how to acquire & extract CICIDS
│   ├── feature_mapping.md        # raw → normalized label table
│   ├── training_workflow.md      # how to train models
│   └── inference_workflow.md     # how to run prediction
├── notebooks/                    # 01_EDA → 05_SHAP (interactive)
├── src/
│   ├── config/                   # constants + YAML loader
│   ├── data/                     # CSV loader, schema, EDA, label_mapping, splitter
│   ├── features/                 # cleaning, encoding, selection, pipeline, validator
│   ├── models/                   # BaseModel + 5 wrappers + tuner + registry
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── catboost_model.py
│   │   ├── mlp.py
│   │   ├── logistic_regression.py
│   │   └── registry.py
│   ├── evaluation/               # metrics, confusion matrix, comparison
│   ├── explainability/           # SHAP analyzer
│   ├── inference/                # batch prediction on user CSVs
│   ├── visualization/            # plot helpers
│   ├── utils/                    # logging, I/O, seeds
│   └── pipelines/                # eda, preprocess, train, evaluate, explain, predict
├── models/                       # *.joblib (gitignored — generated at training)
├── reports/                      # generated markdown reports
├── results/
│   ├── metrics/                  # CSV + JSON
│   ├── figures/                  # PNG (EDA + confusion matrices)
│   └── shap/<model>/             # SHAP plots + top-features JSON
├── dashboard/
│   ├── app.py                    # landing page
│   ├── _shared.py                # shared cached loaders
│   └── pages/
│       ├── 01_Dataset_Overview.py
│       ├── 02_EDA.py
│       ├── 03_Model_Performance.py
│       ├── 04_Model_Comparison.py
│       ├── 05_SHAP.py
│       └── 06_Predict_New_CSV.py
├── tests/                        # pytest (61 tests)
├── scripts/                      # generate_sample.py, etc.
├── logs/                         # pipeline.log
├── main.py                       # CLI entry point
├── pyproject.toml
├── requirements.txt
└── README.md                     # ← this file
```

---

## 6. Installation · การติดตั้ง

### Prerequisites · สิ่งที่ต้องมีก่อน

- Python 3.10 หรือสูงกว่า (Python 3.10+ required)
- Git
- RAM 8 GB ขึ้นไปสำหรับ subsample (16 GB+ สำหรับ full dataset)
- พื้นที่ดิสก์ ~3 GB สำหรับชุดข้อมูล + artefacts

### Windows (PowerShell)

```powershell
# 1. Clone the repository
git clone https://github.com/<user>/cyber_attack_classification.git
cd cyber_attack_classification

# 2. Create + activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .[dev]   # editable install + dev tools (pytest, ruff, black)
```

### macOS / Linux (bash)

```bash
# 1. Clone
git clone https://github.com/<user>/cyber_attack_classification.git
cd cyber_attack_classification

# 2. venv
python3 -m venv .venv
source .venv/bin/activate

# 3. Install
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .[dev]
```

### Verify installation · ตรวจสอบการติดตั้ง

```bash
python -c "from src.models.registry import MODEL_CLASSES; print(list(MODEL_CLASSES))"
# Expected output:
# ['random_forest', 'xgboost', 'lightgbm', 'catboost', 'mlp', 'logistic_regression']
```

---

## 7. Dataset Preparation · การเตรียมข้อมูล

### Source · แหล่งข้อมูล

CICIDS2017 จาก Canadian Institute for Cybersecurity — ต้องขอ access:
<https://www.unb.ca/cic/datasets/ids-2017.html>

ไฟล์ที่ต้องการคือ `MachineLearningCSV.zip` (~225 MB compressed, ~884 MB
extracted, 8 CSVs, ~2.8 million flow records).

### Extract into `data/raw/`

**Windows (PowerShell)**

```powershell
# Replace the source path with where you put the zip
$Zip = 'C:\path\to\MachineLearningCSV.zip'
Expand-Archive -Path $Zip -DestinationPath 'data\raw\.tmp'
Move-Item data\raw\.tmp\MachineLearningCVE\*.csv data\raw\
Remove-Item -Recurse data\raw\.tmp
Get-ChildItem data\raw\*.csv | Select-Object Name, Length
```

**bash / macOS / Linux / Git Bash**

```bash
cd data/raw
unzip /path/to/MachineLearningCSV.zip
mv MachineLearningCVE/*.csv .
rmdir MachineLearningCVE
ls -lh *.csv
```

After extraction your `data/raw/` should contain:

```
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv         # 77 MB · DDoS
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv     # 77 MB · PortScan
Friday-WorkingHours-Morning.pcap_ISCX.csv                # 58 MB · Bot
Monday-WorkingHours.pcap_ISCX.csv                        # 177 MB · BENIGN only
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv  # 83 MB · Infiltration
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv   # 52 MB · Web Attack
Tuesday-WorkingHours.pcap_ISCX.csv                       # 135 MB · FTP/SSH-Patator
Wednesday-workingHours.pcap_ISCX.csv                     # 225 MB · DoS + Heartbleed
```

ดูรายละเอียดเพิ่มเติมที่ [`docs/dataset_preparation.md`](docs/dataset_preparation.md)

### Without real data · ถ้ายังไม่มีข้อมูลจริง

สำหรับทดสอบ / สาธิต สามารถสร้างข้อมูลสังเคราะห์ได้:

```bash
python scripts/generate_sample.py --rows 2000
# Wrote 2,020 rows x 79 cols to data/sample/synthetic_cicids.csv
```

ใช้แทน `data/raw/` ได้ผ่าน `--raw-dir`:

```bash
python main.py --stage eda --raw-dir data/sample
```

> NOTE: ห้ามนำผลจากข้อมูลสังเคราะห์ไปรายงานในโปรเจกต์จริง — มีไว้สำหรับ
> ทดสอบ pipeline เท่านั้น

---

## 8. Configuration · การตั้งค่า

ทุกค่าอยู่ในไฟล์เดียว: [`src/config/config.yaml`](src/config/config.yaml)

ค่าสำคัญที่อาจปรับ:

```yaml
classification:
  mode: "multiclass"          # หรือ "binary"

data:
  raw_dir: "data/raw"
  subsample_n: 300000         # null = ใช้ข้อมูลทั้งหมด (~2.8M rows)
  drop_other_class: false

preprocessing:
  test_size: 0.2
  val_size: 0.2
  scaler: "standard"          # "standard" | "minmax" | "robust"

models:
  random_forest: { enabled: true,  ... }
  xgboost:       { enabled: true,  ... }
  lightgbm:      { enabled: true,  ... }
  catboost:      { enabled: true,  ... }
  mlp:           { enabled: true,  ... }
  # disable specific models by setting enabled: false

tuning:
  enabled: true
  strategy: "random"          # "grid" | "random"
  random_n_iter: 20
```

---

## 9. Quick Start · เริ่มต้นใช้งานเร็ว

### One-liner: end-to-end pipeline (ทั้งระบบในคำสั่งเดียว)

```bash
python main.py --stage all
```

จะรัน: EDA → preprocess → train (5 โมเดล) → evaluate → SHAP → ครบทุกอย่าง

### Three-line minimum (สั้นที่สุดเพื่อให้ได้ผลลัพธ์)

```bash
python main.py --stage preprocess   # เตรียมข้อมูล
python main.py --stage train         # train โมเดลทั้งหมด
python main.py --stage evaluate      # ประเมินผล + report
```

### Launch dashboard (เปิดแดชบอร์ดดูผล)

```bash
streamlit run dashboard/app.py
```

---

## 10. Pipeline Stages · ขั้นตอนของไปป์ไลน์

ทุก stage ใช้ผ่าน `python main.py --stage <name>` ครับ

### Stage: `eda` — Exploratory Data Analysis

```bash
python main.py --stage eda
```

**Output:**
- `results/figures/class_distribution.png`
- `results/figures/missing_value_audit.png`
- `results/figures/correlation_heatmap.png`
- `results/figures/feature_distributions.png`
- `results/metrics/eda_summary.json`

### Stage: `preprocess` — Clean + Label-map + Split

```bash
python main.py --stage preprocess
```

**Output:**
- `data/processed/train.parquet` · `val.parquet` · `test.parquet`
- `data/processed/feature_names.json`
- `data/processed/label_classes.json`
- `models/label_encoder.joblib`

### Stage: `train` — Train Models

```bash
# train all 5 enabled models
python main.py --stage train

# train one model only (alias)
python main.py --stage train --model rf       # random_forest
python main.py --stage train --model xgb      # xgboost
python main.py --stage train --model lgbm     # lightgbm
python main.py --stage train --model cat      # catboost
python main.py --stage train --model nn       # mlp

# skip hyperparameter tuning (faster iteration)
python main.py --stage train --model rf --skip-tuning
```

**Output:**
- `models/<name>.joblib` (fitted Pipeline per model)
- `results/metrics/<name>_val.json` (validation metrics)
- `results/metrics/<name>_cv_results.csv` (top-10 CV rows)
- `results/metrics/val_summary.csv` (cross-model val ranking)

### Stage: `evaluate` — Test-set Metrics + Confusion Matrix + Comparison

```bash
python main.py --stage evaluate
# or just one model:
python main.py --stage evaluate --model rf
```

**Output:**
- `results/metrics/<name>_test.json`
- `results/metrics/classification_report_<name>.csv`
- `results/figures/confusion_matrix_<name>.png`
- `results/metrics/model_comparison.csv`
- `results/metrics/model_comparison.png`
- `reports/model_comparison.md`

### Stage: `explain` — SHAP Explainability

```bash
python main.py --stage explain          # all trained models
python main.py --stage explain --model rf
```

**Output (per model):**
- `results/shap/<name>/summary_bar.png`
- `results/shap/<name>/summary_<class>.png` (one per class)
- `results/shap/<name>/top_features.json`
- `results/shap/shap_report.md`

### Stage: `predict` — Batch Inference on a New CSV

```bash
python main.py --stage predict \
    --input  path/to/my_traffic.csv \
    --output path/to/predictions.csv \
    --model  rf
```

**Output:** CSV with columns
- `predicted_label`
- `true_label_raw` (if input CSV has a `Label` column)
- `proba_<class>` (one column per class)
- `max_proba` (highest probability — confidence proxy)

### Stage: `all`

```bash
python main.py --stage all
```

Runs `eda → preprocess → train → evaluate → explain` in order.

### CLI flags reference · พารามิเตอร์ทั้งหมด

| Flag | Values | Default | Purpose |
|---|---|---|---|
| `--stage` | `eda`, `preprocess`, `train`, `evaluate`, `explain`, `predict`, `all` | (required) | Which stage to run |
| `--model` | `all`, `rf`/`random_forest`, `xgb`/`xgboost`, `lgbm`/`lightgbm`, `cat`/`catboost`, `nn`/`mlp`, `lr`/`logistic_regression` | `all` | Which model(s) |
| `--config` | path | `src/config/config.yaml` | Override config path |
| `--input` | path | — | Input CSV for `--stage predict` |
| `--output` | path | (auto) | Output CSV for `--stage predict` |
| `--raw-dir` | path | (from config) | Override raw data dir (e.g. `data/sample`) |
| `--skip-tuning` | flag | `false` | Skip GridSearch/RandomSearch in train |
| `--log-level` | `DEBUG`/`INFO`/`WARNING`/`ERROR` | `INFO` | Verbosity |

---

## 11. Dashboard · แดชบอร์ด

```bash
streamlit run dashboard/app.py
```

จะเปิด browser ไปที่ `http://localhost:8501` พร้อมเมนู 6 หน้า:

| Page | English | ภาษาไทย |
|---|---|---|
| 1. Dataset Overview | Source, row counts, class balance | ภาพรวม + การกระจายของคลาส |
| 2. EDA | Distribution plots, correlations, missing values | กราฟ EDA |
| 3. Model Performance | Per-model metrics + confusion matrix + report | metric รายโมเดล + confusion matrix |
| 4. Model Comparison | Cross-model ranking + bar chart | เปรียบเทียบโมเดลทั้งหมด |
| 5. SHAP | Feature importance + per-class explanations | อธิบาย model ด้วย SHAP |
| 6. Predict New CSV | Upload → validate → predict + download | อัปโหลด CSV ใหม่เพื่อพยากรณ์ |

> **Tip:** หน้า 6 ต้อง preprocess + train โมเดลก่อนถึงจะใช้ได้
> เพราะต้องโหลด `feature_names.json` และไฟล์ `models/<name>.joblib`

---

## 12. Inference · การพยากรณ์ผล

### CLI

```bash
python main.py --stage predict \
    --input  data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv \
    --output predictions.csv \
    --model  lgbm
```

### Programmatic API · เรียกใช้จากโค้ด Python

```python
from pathlib import Path
from src.inference.predictor import predict_csv, predict_dataframe
import pandas as pd

# จากไฟล์
result = predict_csv(
    input_csv=Path("my_flows.csv"),
    model_name="random_forest",
    output_csv=Path("predictions.csv"),
)
print(result.validation.message)        # OK -- 10000 rows, 78 columns
print(result.predictions.head())

# จาก DataFrame ที่โหลดอยู่แล้ว
df = pd.read_csv("my_flows.csv", encoding="latin-1")
result = predict_dataframe(df, model_name="xgboost")
print(result.predictions["predicted_label"].value_counts())
```

### Schema requirements · ข้อกำหนดของ CSV ที่อัปโหลด

- ต้องมีคอลัมน์ฟีเจอร์ครบตามที่อยู่ใน `data/processed/feature_names.json`
  (77 คอลัมน์ หลังตัด CICIDS duplicate column)
- คอลัมน์เกินอนุญาตได้ (เช่น `Label` สำหรับ QA)
- Inf / NaN จะถูกแทนด้วย 0 ก่อนพยากรณ์
- ระบบจะ strip whitespace นำหน้าชื่อคอลัมน์ให้อัตโนมัติ

ดูตัวอย่างเพิ่มเติม: [`docs/inference_workflow.md`](docs/inference_workflow.md)

---

## 13. Testing · การทดสอบ

### 13.1 Unit tests (pytest) · ทดสอบหน่วยย่อย

```bash
# Run all 61 tests
pytest

# Quiet mode (less output)
pytest -q

# Without coverage report
pytest --no-cov

# Run one file
pytest tests/test_models.py

# Run one test
pytest tests/test_models.py::test_aliases_resolve_to_canonical_names

# Run by marker
pytest -m "not slow"

# Stop on first failure
pytest -x

# Verbose + show local variables on failure
pytest -vv -l
```

**Expected output:**

```
============================= test session starts =============================
collected 61 items

tests/test_config.py ........                                            [ 13%]
tests/test_data_loader.py .........                                      [ 27%]
tests/test_eda.py ....                                                   [ 34%]
tests/test_evaluation.py .....                                           [ 42%]
tests/test_features.py .........                                         [ 57%]
tests/test_models.py ............                                        [ 77%]
tests/test_preprocessing.py ..........                                   [ 93%]
tests/test_utils.py ....                                                 [100%]

======================= 61 passed, 3 warnings in ~8s ==========================
```

### 13.2 Coverage report · รายงาน code coverage

```bash
# Terminal + HTML
pytest --cov=src --cov-report=term-missing --cov-report=html:results/coverage

# Then open in browser
# Windows
start results/coverage/index.html
# macOS
open results/coverage/index.html
# Linux
xdg-open results/coverage/index.html
```

### 13.3 Lint & format · ตรวจคุณภาพโค้ด

```bash
# Lint
ruff check src tests

# Format
black src tests

# Lint + auto-fix safe issues
ruff check --fix src tests
```

### 13.4 Smoke test with synthetic data · ทดสอบไปป์ไลน์ด้วยข้อมูลสังเคราะห์

ไม่ต้องมี CICIDS จริงก็รันได้:

```bash
# 1. Generate synthetic data
python scripts/generate_sample.py --rows 2000

# 2. Run pipeline against sample dir
python main.py --stage eda      --raw-dir data/sample
python main.py --stage preprocess --raw-dir data/sample

# 3. Train one model (synthetic data is small, no need to tune)
python main.py --stage train --model rf --skip-tuning

# 4. Evaluate + SHAP
python main.py --stage evaluate --model rf
python main.py --stage explain  --model rf
```

### 13.5 Real-data smoke test · ทดสอบกับข้อมูลจริง

สร้างไฟล์ `scripts/smoke_test.py` หรือใช้คำสั่งต่อไปนี้ใน Python REPL:

```python
import sys
sys.path.insert(0, ".")
from pathlib import Path
from src.pipelines.preprocess import run as run_preprocess
from src.pipelines.train      import run as run_train
from src.pipelines.evaluate   import run as run_evaluate
from src.pipelines.explain    import run as run_explain

cfg = Path("src/config/config.yaml")

# Make sure data/raw/ has at least one CICIDS CSV first.
summary = run_preprocess(cfg)
print(summary["label_distribution"])

train  = run_train(cfg,    model="rf",  skip_tuning=True)
ev     = run_evaluate(cfg, model="rf")
sh     = run_explain(cfg,  model="rf")
print("Best:", ev["best_model"])
```

ทางลัด: รัน end-to-end ด้วย CSV เดียว (Friday DDos = เล็กที่สุดมี BENIGN + DDoS):

```yaml
# แก้ src/config/config.yaml ชั่วคราว
data:
  subsample_n: 50000
  required_files:
    - "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
tuning:
  enabled: false
```

```bash
python main.py --stage all
```

### 13.6 Dashboard smoke test · ทดสอบแดชบอร์ด

```bash
# Start dashboard in background
streamlit run dashboard/app.py

# Open http://localhost:8501
# Check that all 6 sidebar pages load without error.
# Page 6 needs models/*.joblib to exist (run --stage train first).
```

### 13.7 Continuous testing · ทดสอบขณะพัฒนา

```bash
# Watch mode (install pytest-watch separately if you want this)
pip install pytest-watch
ptw -- --no-cov
```

---

## 14. Results · ผลลัพธ์

### Smoke-test benchmark (3 CSVs, 50K-row stratified subsample)

ทดสอบบน Windows, Python 3.13, CPU only · ผลจาก Friday-DDos + PortScan + Bot:

| Model | Accuracy | F1 (weighted) | F1 (macro) |
|---|---|---|---|
| **LightGBM** | 0.9981 | **0.9981** | 0.9611 |
| **CatBoost** | 0.9976 | 0.9977 | 0.9561 |
| **XGBoost** | 0.9975 | 0.9973 | 0.9155 |
| **Random Forest** | 0.9971 | 0.9969 | 0.9025 |
| **MLP** | 0.9953 | 0.9944 | 0.8225 |

**Key observations:**
- All 5 models > 99.4% accuracy (CICIDS is well-separable on this subset)
- F1-macro spread (0.82–0.96) → minority classes still matter
- LightGBM wins both speed and F1-weighted
- MLP underperforms slightly — expected without SMOTE on this imbalance

> Full-corpus results will land here once `data.subsample_n: null` runs
> overnight on a workstation. (Note: results on synthetic data should
> NOT be reported — that's fixture data, not science.)

---

## 15. Architecture · สถาปัตยกรรม

### Module dependency · การอ้างถึงกันของ module

```
config + utils       ← imported by everyone
   |
data + features      ← depends on config + utils
   |
models               ← depends on data + features + config + utils
   |
evaluation + xai     ← depends on models + data
   |
inference            ← depends on models + features (validator)
   |
pipelines            ← composes everything; called by main.py + dashboard
```

A lower layer never imports from a higher one — cycles impossible by construction.

### Data flow · การไหลของข้อมูล

```
data/raw/*.csv
   → loader.load_raw + schema.validate_schema
   → cleaning.clean (NaN/Inf/dup-column/dup-row)
   → label_mapping.add_mapped_column
   → splitter.train_val_test_split (60/20/20 stratified)
   → data/processed/{train,val,test}.parquet
   → Pipeline(StandardScaler + Classifier).fit
   → tuner.tune_model (CV=5, RandomizedSearchCV)
   → models/<name>.joblib
   → evaluation.compute_metrics + confusion_matrix + comparison
   → explainability.shap_analyzer
   → results/{metrics,figures,shap}/...
   → inference.predictor (for new CSVs)
   → dashboard pages
```

### Key architectural rules · กติกาสถาปัตยกรรม

1. **Single `RANDOM_STATE = 42`** ทุก seed (numpy, sklearn, xgboost, ...) ใช้ค่าเดียวกัน
2. **Scaler อยู่ใน sklearn Pipeline** → fit ใหม่ทุก fold ของ CV → ไม่มี data leakage
3. **Config-driven** — paths, hyperparameters, class lists อยู่ใน `config.yaml` หมด
4. **Stratified everything** — train_test_split, KFold, subsample
5. **Schema validation** at both ingestion (loader) and inference (validator)

อ่านต่อ: [`docs/architecture.md`](docs/architecture.md), [`docs/ml_pipeline.md`](docs/ml_pipeline.md)

---

## 16. Troubleshooting · การแก้ปัญหา

### `ModuleNotFoundError: No module named 'tabulate'`

```bash
pip install tabulate
```

ลงไว้แล้วใน `requirements.txt` ตั้งแต่ v0.2.0 — แต่ถ้า install ก่อนหน้านั้น ต้อง install ใหม่

### `ModuleNotFoundError: No module named 'xgboost' / 'lightgbm' / 'catboost'`

```bash
pip install -r requirements.txt
# หรือเฉพาะที่ขาด
pip install xgboost lightgbm catboost
```

### `UnicodeDecodeError` หรือ Web Attack labels แสดงเป็นอักขระแปลก

ปกติแล้ว — CICIDS Web Attack labels มีไบต์ `0x96` (Windows-1252 en-dash)
Loader อ่านเป็น `encoding="latin-1"` และ `label_mapping.normalize_label`
ลบ control chars ออกก่อน lookup → จัดการให้อัตโนมัติแล้ว

### `FileNotFoundError: No CSV files found in data/raw/`

ยังไม่ extract CICIDS — ดูข้อ 7. Dataset Preparation

### `LabelEncoder is missing N expected classes` (warning)

เกิดเมื่อ subsample เล็กไม่ครอบคลุมทุกคลาส (เช่น Heartbleed มีแค่ 11 แถวในทั้งระบบ)
ไม่ใช่ error — เพิ่ม `subsample_n` ถ้าต้องการครอบคลุมครบ

### Memory error เมื่อโหลด full dataset

ลด subsample ใน config:

```yaml
data:
  subsample_n: 200000   # หรือเล็กกว่า
```

### `Glyph 150 (\x96) missing from font(s) Arial` (warning)

ไม่ใช่ error — matplotlib เตือนว่าฟอนต์ Arial ไม่มี glyph สำหรับ U+0096
(ใน label `Web Attack \x96 ...`) — ส่งผลแค่ข้อความใน plot บางตัวเท่านั้น

### Streamlit port already in use

```bash
streamlit run dashboard/app.py --server.port 8502
```

---

## 17. Development · การพัฒนา

### Workflow ที่แนะนำ

```bash
# 1. Branch
git checkout -b feature/my-change

# 2. Make changes; run tests during dev
pytest -x --no-cov

# 3. Lint + format
ruff check --fix src tests
black src tests

# 4. Full test + coverage
pytest

# 5. Commit + push
git add ...
git commit -m "feat: ..."
git push origin feature/my-change
```

### Adding a new model · เพิ่มโมเดลใหม่

1. สร้าง `src/models/<name>_model.py` extends `BaseModel`, implement `_build_estimator()`
2. ลงทะเบียนใน `src/models/registry.py` — เพิ่มเข้า `MODEL_CLASSES` + `ALIASES`
3. เพิ่ม block ใน `config.yaml::models.<name>` พร้อม `baseline` + `grid`
4. เพิ่ม test ใน `tests/test_models.py`
5. รัน `python main.py --stage train --model <name>` เพื่อตรวจสอบ

### Adding a new feature-selection method

แก้ `src/features/feature_selector.py` — เพิ่ม `select_by_<name>()`,
ลงทะเบียนใน `run_all_methods()`, เพิ่มเข้า `config.yaml::feature_selection.methods`

### Conventions · ข้อตกลง

- Type hints ทุก public function
- Docstring เป็นภาษาอังกฤษ (มาตรฐานการเขียน open-source)
- `RANDOM_STATE` อ่านจาก `src/config/constants.py` เสมอ
- ห้าม `print()` ใน production code — ใช้ `logger`
- ห้าม magic number — ทุกค่าใน `config.yaml`

---

## 18. Out of Scope · นอกเหนือขอบเขต

โครงการนี้ **ไม่** ครอบคลุม:

- Deep learning beyond MLP (no CNN/LSTM/Transformer; no PyTorch/TensorFlow)
- Real-time / streaming inference
- Cloud deployment / Kubernetes
- GPU training
- PCAP parsing — เราใช้ flow CSVs ที่ extract เรียบร้อยจาก UNB CIC
- Adversarial robustness testing

รายการเหล่านี้ถือเป็น **Future Work**

---

## 19. License & Authors · ใบอนุญาตและผู้พัฒนา

### License

MIT License — ดูไฟล์ [`LICENSE`](LICENSE)

### Authors · ผู้พัฒนา

| Name | ID | Faculty |
|---|---|---|
| Sirachet Chotthakunanan (ศิรเชษฐ์ โชติฐากุลอนันต์) | 66070191 | Information Technology |
| Sukhum Rudeemaetakul (สุขุม ฤดีเมธากุล) | 66070315 | Information Technology |

### Advisor · อาจารย์ที่ปรึกษา

Asst. Prof. Dr. Prapan Pavarangkoon (ผศ.ดร.ประพันธ์ ปวรางกูร)
Department of Information Technology, KMITL

### Citation · การอ้างอิง

ถ้าคุณนำโปรเจกต์นี้ไปใช้ในงานวิชาการ กรุณาอ้างอิงดังนี้:

```bibtex
@misc{cyberml_kmitl_2569,
  title  = {AI-Based Cyber Attack Classification from Network Logs using CICIDS2017},
  author = {Chotthakunanan, Sirachet and Rudeemaetakul, Sukhum},
  year   = {2569 (2026)},
  note   = {Senior Project, Faculty of Information Technology, KMITL.
            Advisor: Asst. Prof. Dr. Prapan Pavarangkoon},
}
```

### Dataset citation · การอ้างอิงชุดข้อมูล

```bibtex
@inproceedings{sharafaldin2018toward,
  title     = {Toward Generating a New Intrusion Detection Dataset and
               Intrusion Traffic Characterization},
  author    = {Sharafaldin, Iman and Lashkari, Arash Habibi and
               Ghorbani, Ali A.},
  booktitle = {ICISSP},
  year      = {2018},
}
```

---

## Acknowledgements · กิตติกรรมประกาศ

- ขอขอบคุณ **Canadian Institute for Cybersecurity (UNB)** สำหรับชุดข้อมูล CICIDS2017
- ขอขอบคุณ open-source community: scikit-learn, XGBoost, LightGBM, CatBoost, SHAP, Streamlit
- ขอขอบคุณ **ผศ.ดร.ประพันธ์ ปวรางกูร** สำหรับคำแนะนำตลอดโครงการ

---

<div align="center">

**Built with care for the KMITL IT senior project · 2569**

[Documentation](docs/) · [Issues](../../issues) · [Discussions](../../discussions)

</div>
