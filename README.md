# AI-Based Cyber Attack Classification

**ระบบจำแนกประเภทการโจมตีทางไซเบอร์จาก Network Logs ด้วย Machine Learning + Explainable AI**

> Senior Project — Faculty of Information Technology, KMITL

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-152%20passed-brightgreen)](tests/)
[![Delivery preflight](https://github.com/ksrddd/ai-classification-cyber-attack/actions/workflows/delivery-preflight.yml/badge.svg)](https://github.com/ksrddd/ai-classification-cyber-attack/actions/workflows/delivery-preflight.yml)
[![Dataset](https://img.shields.io/badge/dataset-CICIDS2017-orange)](https://www.unb.ca/cic/datasets/ids-2017.html)
[![Dataset](https://img.shields.io/badge/dataset-CICIDS2018-blue)](https://www.unb.ca/cic/datasets/ids-2018.html)
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

This project reads combined CICIDS2017 and CSE-CIC-IDS2018 network-flow records and automatically
classifies each flow as one of several attack types — like a doctor
diagnosing illness from symptoms. Existing IDS/IPS tools detect *that*
an attack happened; this system tells the operator *what kind*, so the
response can be precise.

The whole pipeline — data engineering, feature engineering, training,
evaluation, explainability, dashboard, inference — is implemented as a
real ML project: config-driven, tested, reproducible, leakage-free.

### ภาษาไทย

โปรเจกต์นี้รับข้อมูล Network Flow จากชุดข้อมูล CICIDS2017 และ CSE-CIC-IDS2018 มาจำแนกประเภท
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
| Seven ML models | Random Forest · XGBoost · LightGBM · CatBoost · MLP · Logistic Regression · Stacking | โมเดล 7 แบบ ครอบคลุม baseline, tree ensemble, neural net และ stacking |
| Label normalization | Maps CICIDS2017 and CSE-CIC-IDS2018 labels into one clean scheme | แปลง label ดิบของ CICIDS/CSE-CIC ให้สะอาด |
| Source-held split | Deterministic 70% train / 30% locked test by source manifest | แยกแหล่งข้อมูล 70/30 แบบกำหนดซ้ำได้ ป้องกัน source leakage |
| Hyperparameter tuning | GridSearchCV / RandomizedSearchCV configurable | ปรับ hyperparameter ด้วย Grid/Random Search ผ่าน config |
| Full metric set | Accuracy · P/R/F1 (weighted+macro+per-class) · ROC-AUC · MCC | ครบทุก metric ที่ใช้ใน security ML |
| Explainability | SHAP TreeExplainer + KernelExplainer fallback | อธิบาย model ด้วย SHAP รองรับทั้ง tree และ MLP |
| Dashboard + API | Streamlit 6 pages plus Next.js UI backed by FastAPI | มีทั้ง Streamlit และ Next.js/FastAPI สำหรับดูผลและทำนาย |
| Schema-safe inference | Upload CSV → validate → predict + probabilities | อัปโหลด CSV ใหม่แล้วทำนายได้ พร้อมเช็ค schema |
| Artifact lifecycle | Checksummed bundle manifest · atomic checkpoints · resume · champion promotion | ตรวจความครบถ้วนของ artifact, resume งาน และ promote champion อย่างปลอดภัย |
| Offline red-team | Copy-only label/shift/OOD/perturbation checks | ตรวจ label conflict, distribution shift, OOD และ robustness โดยไม่แก้ locked test |
| Test suite | 152 pytest cases + smoke/integration tests | มี test 152 รายการ ครอบคลุม pipeline, API, security และ artifact lifecycle |
| Reproducible | `RANDOM_STATE=42` + hash quotas + source-grouped CV | สุ่มซ้ำได้และรักษา source isolation ระหว่าง train/CV/test |
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
| 6 | Model development | 7 model wrappers + registry + stacking | done |
| 7 | Hyperparameter tuning | GridSearch / RandomizedSearch | done |
| 8 | Evaluation | metric / confusion matrix / report | done |
| 9 | SHAP / XAI | อธิบายโมเดลด้วย SHAP | done |
| 10 | Dashboard + REST API | Streamlit 6 หน้า + Next.js/FastAPI | done |
| 11 | Testing | pytest 152 tests + integration/security smoke tests | done |
| 12 | Inference / MLOps | checksum manifest, checkpoints, resume, promotion | done |
| 13 | Offline red-team | label/shift/OOD/perturbation checks | done |
| 14 | Documentation | เอกสารส่งมอบ + README นี้ | done |

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
| Dashboard / Web | `streamlit`, `next.js`, `react`, `recharts` |
| REST API | `fastapi`, `uvicorn` |
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
│   ├── raw/                      # CICIDS2017 + CSE-CIC-IDS2018 CSVs (download separately)
│   ├── interim/                  # intermediate parquet files
│   ├── processed/                # cleaned cache + legacy split artifacts
│   └── sample/                   # synthetic CICIDS-shaped fixture for tests
├── configs/splits/               # versioned source-held split manifests (70/30)
├── docs/
│   ├── architecture.md           # module map + ADRs
│   ├── ml_pipeline.md            # end-to-end ML flow
│   ├── evaluation.md             # metric definitions + rationale
│   ├── shap_explanation.md       # SHAP setup + interpretation guide
│   ├── dataset_preparation.md    # how to acquire & extract CICIDS
│   ├── feature_mapping.md        # raw → normalized label table
│   ├── training_workflow.md      # how to train models
│   ├── inference_workflow.md     # how to run prediction
│   └── source_holdout_protocol.md# source isolation + locked-test protocol
├── notebooks/                    # 01_EDA → 05_SHAP (interactive)
├── src/
│   ├── config/                   # constants + YAML loader
│   ├── data/                     # loader, schema, deterministic split, provenance
│   ├── features/                 # cleaning, encoding, selection, pipeline, validator
│   ├── models/                   # 7 models + tuner + stacking + registry
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
│   ├── artifacts/                # bundle manifest, checksum, promotion
│   ├── security/                 # offline red-team checks
│   ├── training/                 # atomic checkpoint + resume state
│   ├── visualization/            # plot helpers
│   ├── utils/                    # logging, I/O, seeds
│   └── pipelines/                # train/evaluate/audit/red-team/promote/etc.
├── models/                       # *.joblib (gitignored — generated at training)
├── reports/                      # generated markdown reports
├── results/
│   ├── <run-id>/                 # immutable run bundle + checkpoints + reports
│   └── champion.json             # published champion pointer + policy evidence
├── api/main.py                   # FastAPI backend + bounded CSV upload
├── web/                          # Next.js dashboard frontend
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
├── tests/                        # pytest (152 tests)
├── scripts/                      # sample generator + utility scripts
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

เวอร์ชันนี้รองรับไฟล์ flow CSV ทั้งชุดข้อมูล CICIDS2017 และ CSE-CIC-IDS2018 จาก Canadian Institute for Cybersecurity
ให้วางไฟล์ CSV ทั้งหมดไว้ใน `data/raw/` โดย local training cache ปัจจุบันถูกสร้างขึ้นจากไฟล์ CSV ดิบจำนวน 18 ไฟล์ และสร้างระเบียนข้อมูล flow ที่สะอาดแล้วประมาณ 13.9 ล้านแถว

ลิงก์หน้าชุดข้อมูลที่มีประโยชน์:
- CICIDS2017: <https://www.unb.ca/cic/datasets/ids-2017.html>
- CSE-CIC-IDS2018: <https://www.unb.ca/cic/datasets/ids-2018.html>

### รูปแบบโครงสร้างโฟลเดอร์ `data/raw/` ที่คาดหวัง

```text
data/raw/
  02-14-2018.csv
  02-15-2018.csv
  02-16-2018.csv
  ...
  Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
  Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
  ...
```

`train.py` จะปรับประเภทคอลัมน์และ label ของทั้ง CICIDS2017 และ CSE-CIC-IDS2018 ให้เป็นโครงสร้างข้อมูลเดียวกันก่อนทำการฝึกฝน (training) แนะนำให้สร้างแคชข้อมูลสะอาดใหม่เฉพาะเมื่อมีการเปลี่ยนแปลงของไฟล์ CSV ดิบเท่านั้น:

```bash
python main.py --stage train --refresh-cache --force
```


### Source · แหล่งข้อมูล

CICIDS2017 จาก Canadian Institute for Cybersecurity — ต้องขอ access:
<https://www.unb.ca/cic/datasets/ids-2017.html>

ไฟล์ที่ต้องการคือ `MachineLearningCSV.zip` (~225 MB compressed, ~884 MB
extracted, 8 CSVs, ~2.8 million flow records).

### Extract into `data/raw/` · การแตกไฟล์ลงใน `data/raw/`

**Windows (PowerShell)**

```powershell
# แทนที่พาธต้นทางด้วยตำแหน่งที่คุณเก็บไฟล์ zip
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

หลังจากการแตกไฟล์ โฟลเดอร์ `data/raw/` ควรประกอบด้วยไฟล์ต่อไปนี้:

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
```

> หมายเหตุ: ข้อมูลสังเคราะห์ (synthetic data) มีไว้เพื่อทดสอบการทำงานของไปป์ไลน์เท่านั้น ห้ามนำผลการวัดประสิทธิภาพ (metrics) ที่ได้จากข้อมูลสังเคราะห์ไปรายงานเป็นผลลัพธ์ของโครงการ
## 8. Configuration · การตั้งค่า

ทุกค่าอยู่ในไฟล์เดียว: [`src/config/config.yaml`](src/config/config.yaml)

ค่าสำคัญที่อาจปรับ:

```yaml
classification:
  mode: "multiclass"          # หรือ "binary"

data:
  raw_dir: "data/raw"
  subsample_n: 300000         # null = ใช้ข้อมูลทั้งหมด (~13.9M cleaned rows)
  drop_other_class: false

preprocessing:
  test_size: 0.3
  val_size: 0.0
  scaler: "standard"          # "standard" | "minmax" | "robust"

models:
  random_forest: { enabled: true,  ... }
  xgboost:       { enabled: true,  ... }
  lightgbm:      { enabled: true,  ... }
  catboost:      { enabled: true,  ... }
  mlp:           { enabled: true,  ... }
  logistic_regression: { enabled: true, ... }
  stacking:      { enabled: true,  ... }
  # ปิดการใช้งานโมเดลเฉพาะเจาะจงโดยตั้งค่าเป็น enabled: false

tuning:
  enabled: true
  strategy: "random"          # "grid" | "random"
  random_n_iter: 20
```

---

## 9. Quick Start · เริ่มต้นใช้งานเร็ว

ใช้ `main.py` เป็น entry point หลัก แต่ละการรันควรกำหนด `--run-name` เพื่อเขียน bundle แยกที่
`results/<run-name>/` จากนั้นตรวจ integrity และ promote เฉพาะ run ที่ผ่านเกณฑ์เป็น champion

```bash
# 1. ตรวจโครงสร้างและ version ของ source-held manifest
python main.py --stage audit --split-manifest configs/splits/source_holdout_v3_full_70_30.json

# 2. Full Train แบบ source-held 70/30 (เครื่อง RAM สูงหรือ DGX)
python main.py --stage train --run-name thesis_local \
  --preset full --profile overnight \
  --split-manifest configs/splits/source_holdout_v3_full_70_30.json \
  --model all --skip-tuning

# 3. อ่าน metrics จาก run เดิม และ promote ตาม selection policy
python main.py --stage evaluate --run-name thesis_local
python main.py --stage promote --run-name thesis_local

# เปิดแดชบอร์ด Streamlit
python main.py --stage dashboard

# พยากรณ์ข้อมูลจากไฟล์ CSV ใหม่
python main.py --stage predict --input path/to/my_traffic.csv --output predictions.csv --model rf
```

ตัวเลือกเสริมทั่วไปสำหรับการเทรนโมเดล:

```bash
# บังคับเทรนใหม่แม้ว่าจะมีไฟล์ผลลัพธ์อยู่แล้ว
python main.py --stage train --force

# สร้างไฟล์ parquet สะอาดใหม่จากไฟล์ CSV ดิบ
python main.py --stage train --refresh-cache --force

# เทรนเฉพาะโมเดลเดี่ยวที่กำหนด
python main.py --stage train --model rf --force
python main.py --stage train --model xgb --force
python main.py --stage train --model lgbm --force

# เทรนแบบรวดเร็วโดยข้ามขั้นตอนการค้นหา Hyperparameter
python main.py --stage train --skip-tuning

# RAM 16 GB: train with the larger safe preset
python main.py --stage train --run-name latest --preset 16gb --force

# Full clean-cache training: uses every row in data/processed/cicids_clean.parquet.
# Recommended RAM: 64 GB minimum, 96-128 GB preferred for all enabled models.
python main.py --stage train --run-name latest --preset full --force --skip-cv --skip-label-shuffle

# Full training, faster but with no hyperparameter search.
python main.py --stage train --run-name latest --preset full --force --skip-hp --skip-cv --skip-label-shuffle
```
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
# train all latest enabled models
python main.py --stage train

# train one model only (alias)
python main.py --stage train --model rf       # random_forest
python main.py --stage train --model xgb      # xgboost
python main.py --stage train --model lgbm     # lightgbm
python main.py --stage train --model cat      # catboost
python main.py --stage train --model nn       # mlp
python main.py --stage train --model lr       # logistic_regression
python main.py --stage train --model stacking # stacking ensemble

# skip hyperparameter tuning (faster iteration)
python main.py --stage train --model rf --skip-tuning

# RAM presets for a named run bundle
python main.py --stage train --run-name latest --preset 8gb --force
python main.py --stage train --run-name latest --preset 16gb --force
python main.py --stage train --run-name latest --preset 32gb --force

# full clean-cache training (needs high RAM; skip CV checks for practicality)
python main.py --stage train --run-name latest --preset full --force --skip-cv --skip-label-shuffle

# safer first full run on a 16 GB machine: try LightGBM only
python main.py --stage train --model lgbm --run-name latest --preset full --force --skip-hp --skip-cv --skip-label-shuffle
```

**Output:**
- `results/<run-name>/<model>.joblib` (fitted Pipeline per model)
- `results/<run-name>/<model>_metrics.json` (locked-test metrics)
- `results/<run-name>/<model>_per_class.csv` (per-class report)
- `results/<run-name>/metrics.json` and `report.md` (cross-model summary)
- `results/<run-name>/bundle_manifest.json` (SHA-256 integrity manifest)
- `results/<run-name>/checkpoints/` (atomic state for resume)

### Stage: `evaluate` — Test-set Metrics + Confusion Matrix + Comparison

```bash
python main.py --stage evaluate
# or just one model:
python main.py --stage evaluate --model rf
```

**Output:**
- `results/<run-name>/<model>_metrics.json`
- `results/<run-name>/<model>_per_class.csv`
- `results/<run-name>/<model>_confusion_matrix.png`
- `results/<run-name>/metrics.json`
- `results/<run-name>/report.md`

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

Runs the training path and writes artifacts under `results/<run-name>/` (`latest` remains the compatibility default).

### Stages: `audit`, `red-team`, `promote`

```bash
# Validate source manifest structure and version
python main.py --stage audit --split-manifest configs/splits/source_holdout_v3_full_70_30.json

# Run copy-only security checks against a trained candidate bundle
python main.py --stage red-team --run-name thesis_local \
  --reference path/to/reference_clean.csv \
  --candidate path/to/candidate_labelled.csv \
  --model-path results/thesis_local/random_forest.joblib

# Verify the bundle and publish results/champion.json according to the selection policy
python main.py --stage promote --run-name thesis_local
```

`red-team` จะเขียนรายงานใน candidate bundle โดยไม่แก้ locked-test data ส่วน `promote` จะตรวจ checksum
ก่อนอัปเดต champion pointer หากไม่มีโมเดลผ่าน FPR threshold สถานะจะเป็น conditional และบันทึกเหตุผลไว้

### CLI flags reference · พารามิเตอร์ทั้งหมด

| Flag | Values | Default | Purpose |
|---|---|---|---|
| `--stage` | `train`, `evaluate`, `dashboard`, `predict`, `eda`, `preprocess`, `explain`, `all`, `audit`, `red-team`, `promote` | (required) | Which stage to run |
| `--model` | `all`, `rf`, `xgb`, `lgbm`, `cat`, `nn`, `lr`, `stacking` (or canonical names) | `all` | Which model(s) |
| `--config` | path | `src/config/config.yaml` | Override config path |
| `--run-name` | string | `latest` | Immutable output bundle name under `results/` |
| `--split-manifest` | path | — | Versioned source-held split manifest |
| `--profile` | `dev`, `overnight` | `dev` | Resource/tuning profile |
| `--accelerator` | `cpu`, `gpu` | `cpu` | Select CPU or supported NVIDIA GPU path |
| `--gpu-devices` | string | `0` | GPU device IDs forwarded to supported models |
| `--reference` | CSV path | — | Clean reference data for `red-team` |
| `--candidate` | labelled CSV path | — | Candidate data for `red-team` |
| `--model-path` | trusted joblib path | — | Optional model for evasion checks |
| `--input` | path | — | Input CSV for `--stage predict` |
| `--output` | path | (auto) | Output CSV for `--stage predict` |
| `--raw-dir` | path | (from config) | Override raw data dir (e.g. `data/sample`) |
| `--preset` | `8gb`, `16gb`, `32gb`, `full` | none | Training RAM/data-size preset |
| `--skip-tuning` | flag | `false` | Skip hyperparameter search in train |
| `--skip-cv` | flag | `false` | Skip cross-validation trust check during training |
| `--skip-label-shuffle` | flag | `false` | Skip shuffled-label sanity check during training |
| `--force` | flag | `false` | Retrain existing model artifacts |
| `--refresh-cache` | flag | `false` | Rebuild `data/processed/cicids_clean.parquet` from raw CSVs |
| `--port` | int | `8501` | Streamlit port for `--stage dashboard` |
| `--log-level` | `DEBUG`/`INFO`/`WARNING`/`ERROR` | `INFO` | Verbosity |

---

## 11. Dashboard · แดชบอร์ด

```bash
python main.py --stage dashboard
```

Open `http://localhost:8501`. If that port is already used:

```bash
python main.py --stage dashboard --port 8502
```

The Streamlit dashboard now reads `results/latest/` directly and falls back to those artifacts when legacy `models/` or `results/metrics/` are empty.

| Page | English | ภาษาไทย |
|---|---|---|
| 1. Dataset Overview | Combined CICIDS/CSE-CIC row counts and class balance | ภาพรวม + การกระจายของคลาส |
| 2. EDA | Distribution plots, correlations, missing values | กราฟ EDA |
| 3. Model Performance | Per-model metrics + confusion matrix + report | metric รายโมเดล + confusion matrix |
| 4. Model Comparison | Cross-model ranking + bar chart | เปรียบเทียบโมเดลทั้งหมด |
| 5. SHAP | Feature importance + per-class explanations | อธิบาย model ด้วย SHAP |
| 6. Predict New CSV | Upload → validate → predict + download | อัปโหลด CSV ใหม่เพื่อพยากรณ์ |

> Page 6 expects the combined CICIDS/CSE-CIC 80-feature schema and uses model artifacts from `results/latest/`.

มี Next.js dashboard เป็นอีก frontend หนึ่ง โดยเปิด backend และ frontend คนละ terminal:

```bash
# Terminal 1: FastAPI (http://localhost:8000, OpenAPI docs at /docs)
uvicorn api.main:app --reload --port 8000

# Terminal 2: Next.js (http://localhost:3000)
cd web
npm install
npm run dev
```

FastAPI อ่านไฟล์อัปโหลดเป็น chunk และจำกัดขนาด CSV ที่ 50 MB, ตรวจ path component และคืน
`model_run_id`/`contract_version` ใน prediction response เพื่อ trace กลับไปยัง artifact ได้

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
print(result.validation.message)        # OK -- 10000 rows, 80 columns
print(result.predictions.head())

# จาก DataFrame ที่โหลดอยู่แล้ว
df = pd.read_csv("my_flows.csv", encoding="latin-1")
result = predict_dataframe(df, model_name="xgboost")
print(result.predictions["predicted_label"].value_counts())
```

### Schema requirements · ข้อกำหนดของ CSV ที่อัปโหลด

- ต้องมีคอลัมน์ฟีเจอร์ครบตามที่อยู่ใน `data/processed/feature_names.json`
  (80 คอลัมน์ของ combined CICIDS/CSE-CIC schema)
- คอลัมน์เกินอนุญาตได้ (เช่น `Label` สำหรับ QA)
- ค่า Inf จะถูกแปลงเป็น NaN แล้วส่งให้ fitted imputer ใน model pipeline จัดการ โดยรักษา missingness semantics เดิม
- ระบบจะ strip whitespace นำหน้าชื่อคอลัมน์ให้อัตโนมัติ
- เมื่อใช้ published champion ระบบจะตรวจ bundle checksum ก่อนโหลด และส่ง `model_run_id`/`contract_version` กลับเพื่อ audit ได้

ดูตัวอย่างเพิ่มเติม: [`docs/inference_workflow.md`](docs/inference_workflow.md)

---

## 13. Testing · การทดสอบ

### 13.1 Unit tests (pytest) · ทดสอบหน่วยย่อย

```bash
# Run all 151 tests
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
152 passed, 4 warnings in ~94s
```

ผลล่าสุดที่ยืนยันใน repository นี้คือ `152 passed, 4 warnings` และ coverage รวม 57%
เวลาในการรันขึ้นอยู่กับเครื่อง (บน environment ที่ใช้ตรวจ README รอบนี้ประมาณ 17 วินาที)

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
ruff check .

# Format
black src tests scripts main.py

# Lint + auto-fix safe issues
ruff check --fix .
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
python main.py --stage dashboard

# Open http://localhost:8501
# Check that all 6 sidebar pages load without error.
# Page 6 needs results/latest/*.joblib to exist (run --stage train first).
```

### 13.7 Continuous testing · ทดสอบขณะพัฒนา

```bash
# Watch mode (install pytest-watch separately if you want this)
pip install pytest-watch
ptw -- --no-cov
```

---

## 14. Results · ผลลัพธ์

### Latest verified local run: `local_300k_70_30`

ผลที่ commit ไว้ล่าสุดเป็น **CPU smoke/reference run** บน combined CICIDS2017 + CSE-CIC-IDS2018 จำนวน
300,000 แถว ใช้ manifest `source_holdout_v2_70_30`: train 210,000 / locked test 90,000,
80 features, 9 observed classes, ไม่มี calibration split และข้าม hyperparameter search

| Model | Accuracy | Macro-F1 | Balanced accuracy | Infiltration F2 | FPR |
|---|---:|---:|---:|---:|---:|
| Stacking | **0.9529** | 0.8438 | 0.8491 | 0.8285 | 0.0346 |
| **Random Forest** | 0.9507 | **0.9160** | 0.8987 | 0.7711 | **0.0269** |
| XGBoost | 0.9523 | 0.8448 | 0.8478 | 0.8249 | 0.0347 |
| LightGBM | 0.9507 | 0.8407 | 0.8442 | 0.7973 | 0.0316 |
| CatBoost | 0.9467 | 0.9070 | **0.9561** | **0.8285** | 0.0416 |
| MLP | 0.9284 | 0.8822 | 0.9440 | 0.8269 | 0.0627 |
| Logistic Regression | 0.8994 | 0.8181 | 0.8602 | 0.7229 | 0.0720 |

Selection policy กำหนด `target_max_fpr = 0.02` แต่ไม่มีโมเดลใดผ่านเกณฑ์นี้ จึง promote
Random Forest แบบ `conditional_no_model_meets_fpr` โดยเลือก FPR ต่ำสุดก่อน แล้วจึงพิจารณา Macro-F1
สถานะดังกล่าวถูกบันทึกใน `results/champion.json` และต้องรายงานข้อจำกัดนี้ ไม่ควรอ้างว่าโมเดลผ่าน FPR target แล้ว

ไฟล์หลักที่ตรวจสอบได้:

- `results/local_300k_70_30/metrics.json`, `report.md` และ per-model artifacts
- `results/local_300k_70_30/bundle_manifest.json` — 32 core artifacts พร้อม checksums
- `results/local_300k_70_30/red_team.json` — actual copy-only smoke กับ Random Forest
- `results/champion.json` — champion pointer และ selection evidence

Red-team ที่ 5% perturbation ผ่าน แต่ 10% ไม่ผ่าน; รายงานนี้ใช้ processed-split smoke แบบ legacy
จึงไม่ใช่ผล robustness สุดท้ายของ locked source-held test ส่วน Heartbleed มีเพียง 5 ตัวอย่าง
(train 3 / test 2) และคลาส `Other` ไม่ปรากฏใน 9 observed classes ของ run นี้ การสรุปประสิทธิภาพ
ต้องดู Macro-F1, per-class recall, FPR และ confusion matrix
ควบคู่กับ Accuracy เสมอ

> ตารางข้างบนเป็น local reference run เท่านั้น

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
   → schema validation + cleaned parquet cache + provenance
   → versioned source manifest + SHA-256 verification
   → deterministic source-held split (70% train / 30% locked test)
   → deterministic hash quotas within each source
   → GroupKFold by source for train-only CV/tuning
   → fitted preprocessing Pipeline + 7 candidate models
   → results/<run-id>/ checkpoints + metrics + explainability
   → bundle_manifest.json (SHA-256 integrity verification)
   → offline red-team checks on a copy of candidate data
   → policy-based promotion → results/champion.json
   → inference.predictor / FastAPI / dashboard
```

### Key architectural rules · กติกาสถาปัตยกรรม

1. **Single `RANDOM_STATE = 42`** ทุก seed (numpy, sklearn, xgboost, ...) ใช้ค่าเดียวกัน
2. **Scaler อยู่ใน sklearn Pipeline** → fit ใหม่ทุก fold ของ CV → ไม่มี data leakage
3. **Config-driven** — paths, hyperparameters, class lists อยู่ใน `config.yaml` หมด
4. **Source isolation first** — locked test sources ไม่เข้า train/CV; CV group ตาม source
5. **Deterministic quotas** — hash-based sampling ทำให้ rerun ได้ชุดเดิมและตรวจ provenance ได้
6. **Immutable run bundles** — checkpoint เขียนแบบ atomic, resume ได้ และ checksum ก่อน promote/load
7. **Schema validation** at both ingestion (loader) and inference (validator)

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
python main.py --stage dashboard --port 8502
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
- Multi-node / distributed GPU training (single-node NVIDIA GPU path รองรับ XGBoost/CatBoost และ stacking บางส่วน)
- PCAP parsing — เราใช้ flow CSVs ที่ extract เรียบร้อยจาก UNB CIC
- Online adversarial attacks หรือการสร้าง traffic โจมตีระบบจริง (มีเฉพาะ offline copy-only red-team)

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
  title  = {AI-Based Cyber Attack Classification from Network Logs using CICIDS2017 and CSE-CIC-IDS2018},
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

- ขอขอบคุณ **Canadian Institute for Cybersecurity (UNB)** สำหรับชุดข้อมูล CICIDS2017 และ CSE-CIC-IDS2018
- ขอขอบคุณ open-source community: scikit-learn, XGBoost, LightGBM, CatBoost, SHAP, Streamlit
- ขอขอบคุณ **ผศ.ดร.ประพันธ์ ปวรางกูร** สำหรับคำแนะนำตลอดโครงการ

---

<div align="center">

**Built with care for the KMITL IT senior project · 2569**

[Documentation](docs/) · [Issues](../../issues) · [Discussions](../../discussions)

</div>
