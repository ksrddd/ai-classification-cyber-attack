# AI-Based Cyber Attack Classification

**ระบบจำแนกประเภทการโจมตีทางไซเบอร์จาก Network Logs ด้วย Machine Learning + Explainable AI**

> Senior Project — Faculty of Information Technology, KMITL

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-152%20passed-brightgreen)](tests/)
[![Delivery preflight](https://github.com/ksrddd/ai-classification-cyber-attack/actions/workflows/delivery-preflight.yml/badge.svg)](https://github.com/ksrddd/ai-classification-cyber-attack/actions/workflows/delivery-preflight.yml)
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

**Evaluation protocol.** Models are judged on four dimensions — classification
performance, attack detection ability, operational impact, and computational
efficiency — with **macro-F1 and per-class recall as the deciding metrics, not
accuracy**. Three rankings are published side by side (Overall / Security-focused /
Deployment) instead of a single "best" model. The train/test split is
chronological, so no model is tested on traffic from the same moment it trained on.

### ภาษาไทย

โปรเจกต์นี้รับข้อมูล Network Flow จากชุดข้อมูล CICIDS2017 มาจำแนกประเภท
การโจมตีทางไซเบอร์โดยอัตโนมัติ — เปรียบเหมือนหมอที่วินิจฉัยโรคจากอาการ
ระบบ IDS/IPS ทั่วไปบอกได้แค่ว่า "มีการโจมตีเกิดขึ้น" แต่ไม่บอกว่าเป็น
ประเภทไหน ระบบนี้ระบุประเภทให้ผู้ดูแลตอบสนองได้ตรงจุด

ไปป์ไลน์ทั้งหมด — Data engineering, Feature engineering, Training,
Evaluation, Explainable AI, Dashboard, Inference — ถูกพัฒนาด้วยมาตรฐาน
ML จริง: ปรับด้วย config, มี test, รันซ้ำได้, ป้องกัน data leakage

**มาตรฐานการประเมินผล** วัดผล 4 มิติ ได้แก่ Classification Performance,
Attack Detection Ability, Operational Impact และ Computational Efficiency
โดยใช้ **Macro-F1 และ per-class Recall เป็นตัวตัดสินหลัก ไม่ใช่ Accuracy**
และจัดอันดับ 3 มุม (Overall / Security-focused / Deployment) แทนการเลือก
"โมเดลที่ดีที่สุด" เพียงตัวเดียว การแบ่ง train/test เป็นแบบเรียงตามเวลา
โมเดลจึงไม่ถูกทดสอบด้วยทราฟฟิกช่วงเวลาเดียวกับที่ใช้เทรน

---

## 2. Features · ฟีเจอร์

| Feature | English | ภาษาไทย |
|---------|---------|---------|
| Dual classification | Binary (`Normal`/`Attack`) **and** multi-class (9 classes on CICIDS2017) | รองรับทั้งแบบ 2 คลาส และหลายคลาส |
| Seven ML models | Random Forest · XGBoost · LightGBM · CatBoost · MLP · Logistic Regression · Stacking | โมเดล 7 แบบ ครอบคลุม baseline, tree ensemble, neural net และ stacking |
| Label normalization | Maps the 15 raw CICIDS2017 labels into 9 clean attack families | แปลง label ดิบ 15 แบบให้เหลือ 9 คลาส |
| Temporal split | Chronological 70% train / 30% locked test inside every (capture, class) group, no RNG | แบ่งตามเวลาแบบเรียงลำดับ ป้องกัน leakage โดยไม่ต้องสุ่ม |
| Hyperparameter tuning | GridSearchCV / RandomizedSearchCV configurable | ปรับ hyperparameter ด้วย Grid/Random Search ผ่าน config |
| Full metric set | Accuracy · P/R/F1 (weighted+macro+per-class) · FPR/FNR (per-class+binary) · MCC | ครบทุก metric ที่ใช้ใน security ML |
| Explainability | SHAP TreeExplainer + KernelExplainer fallback | อธิบาย model ด้วย SHAP รองรับทั้ง tree และ MLP |
| Dashboard + API | Streamlit 6 pages plus Next.js UI backed by FastAPI | มีทั้ง Streamlit และ Next.js/FastAPI สำหรับดูผลและทำนาย |
| Schema-safe inference | Upload CSV → validate → predict + probabilities | อัปโหลด CSV ใหม่แล้วทำนายได้ พร้อมเช็ค schema |
| Artifact lifecycle | Checksummed bundle manifest · atomic checkpoints · resume · champion promotion | ตรวจความครบถ้วนของ artifact, resume งาน และ promote champion อย่างปลอดภัย |
| Offline red-team | Copy-only label/shift/OOD/perturbation checks | ตรวจ label conflict, distribution shift, OOD และ robustness โดยไม่แก้ locked test |
| Test suite | 228 pytest cases + smoke/integration tests | มี test 228 รายการ ครอบคลุม pipeline, API, security และ artifact lifecycle |
| Reproducible | `RANDOM_STATE=42` + hash quotas + source-grouped CV | สุ่มซ้ำได้และรักษา source isolation ระหว่าง train/CV/test |
| Config-driven | Training knobs live in `train.py`'s own `CONFIG` dict; `config.yaml` drives only `--stage eda` and dashboard labels | ค่าฝั่งเทรนอยู่ใน `CONFIG` dict ใน `train.py`; `config.yaml` ใช้เฉพาะ `--stage eda` และ label ของ dashboard |

### Classification Schemes · ระบบจำแนกคลาส

**Binary mode** — สำหรับการตรวจจับว่ามีการโจมตีหรือไม่
- `Normal` — BENIGN traffic (การจราจรปกติ)
- `Attack` — ทุกประเภทที่ไม่ใช่ BENIGN

**Multi-class mode** — สำหรับระบุประเภทการโจมตี (CICIDS2017 มี 9 คลาส)

`Other` เป็นคลาสสำรองสำหรับ label ที่ map ไม่ได้ แถวที่ตกลง `Other` จะถูก drop ตอน
cleaning จึงไม่ปรากฏใน 2017 — เหลือ 9 คลาสจริง จำนวนแถวด้านล่างมาจาก
`data/processed/cicids2017_clean.parquet`

| Class | Description (EN) | คำอธิบาย (TH) | Capture file | n_train | n_test |
|---|---|---|---|---:|---:|
| `BENIGN` | Normal background traffic | การจราจรปกติ | ทุกไฟล์ | 1,450,574 | 621,680 |
| `DoS` | Denial of Service attacks | การโจมตีแบบ DoS | Wednesday | 135,611 | 58,119 |
| `DDoS` | Distributed DoS | การโจมตีแบบกระจายจากหลายแหล่ง | Friday-Afternoon-DDos | 89,609 | 38,405 |
| `PortScan` | Network port scanning | สแกนพอร์ตเพื่อหาช่องโหว่ | Friday-Afternoon-PortScan | 63,485 | 27,209 |
| `Brute Force` | Credential brute-forcing | เดารหัสผ่านบริการเครือข่าย | Tuesday | 6,405 | 2,745 |
| `Web Attack` | XSS, SQL Injection, web brute force | การโจมตีเว็บแอป | Thursday-Morning | 1,500 | 643 |
| `Bot` | Botnet C&C traffic | การสื่อสารของบอตเน็ต | Friday-Morning | 1,363 | 585 |
| `Infiltration` | Lateral movement / infiltration | การเจาะระบบเข้าไปใน network | Thursday-Afternoon | 25 | 11 |
| `Heartbleed` | OpenSSL Heartbleed exploit | ช่องโหว่ Heartbleed | Wednesday | 7 | 4 |

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
| 11 | Testing | pytest 228 tests + integration/security smoke tests | done |
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
│   ├── raw/                      # 8 CICIDS2017 CSVs (download separately)
│   ├── interim/                  # intermediate parquet files
│   ├── processed/                # cleaned cache + legacy split artifacts
│   └── sample/                   # synthetic CICIDS-shaped fixture for tests
├── configs/splits/               # temporal split manifest (70/30, chronological)
├── docs/
│   ├── architecture.md           # module map + ADRs
│   ├── ml_pipeline.md            # end-to-end ML flow
│   ├── evaluation.md             # metric definitions + rationale
│   ├── shap_explanation.md       # SHAP setup + interpretation guide
│   ├── dataset_preparation.md    # how to acquire & extract CICIDS
│   ├── feature_mapping.md        # raw → normalized label table
│   ├── training_workflow.md      # how to train models
│   ├── inference_workflow.md     # how to run prediction
├── notebooks/                    # exploratory notebooks (not part of the pipeline)
├── src/
│   ├── config/                   # constants + YAML loader
│   ├── data/                     # loader, schema, temporal split, provenance
│   ├── features/                 # cleaning, encoding, selection, pipeline, validator
│   ├── models/                   # 7 models + tuner + stacking + registry
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── catboost_model.py
│   │   ├── mlp.py
│   │   ├── logistic_regression.py
│   │   └── registry.py
│   ├── explainability/           # SHAP analyzer
│   ├── inference/                # batch prediction on user CSVs
│   ├── artifacts/                # bundle manifest, checksum, promotion
│   ├── security/                 # offline red-team checks
│   ├── training/                 # checkpoint/resume + GPU acceptance gate
│   ├── visualization/            # plot helpers
│   ├── utils/                    # logging, I/O, seeds
│   └── pipelines/                # eda + SHAP explain stages
├── results/
│   ├── <run-id>/                 # run bundle: models, metrics, report, SHAP
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
├── tests/                        # pytest (228 tests)
├── scripts/                      # sample/mock data generators
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
# ['random_forest', 'xgboost', 'lightgbm', 'catboost', 'mlp', 'logistic_regression', 'stacking']
```

---

## 7. Dataset Preparation · การเตรียมข้อมูล

เวอร์ชันนี้รองรับไฟล์ flow CSV ทั้งชุดข้อมูล CICIDS2017 จาก Canadian Institute for Cybersecurity (2,497,980 แถวสะอาด จาก 8 capture files)
ต้องมีครบทั้ง 8 ไฟล์ เพราะแต่ละคลาสโจมตีอยู่ในไฟล์เดียว ถ้าขาดไฟล์ใดไฟล์หนึ่ง
จะหายไปทั้งคลาส — ตัว loader จะหยุดและแจ้ง error แทนที่จะสร้าง cache ที่ไม่ครบ

ลิงก์หน้าชุดข้อมูลที่มีประโยชน์:
- CICIDS2017: <https://www.unb.ca/cic/datasets/ids-2017.html>

### รูปแบบโครงสร้างโฟลเดอร์ `data/raw/` ที่คาดหวัง

```text
data/raw/
  # ต้องมีครบทั้ง 8 ไฟล์
  Monday-WorkingHours.pcap_ISCX.csv                            # BENIGN ล้วน
  Tuesday-WorkingHours.pcap_ISCX.csv                           # Brute Force
  Wednesday-workingHours.pcap_ISCX.csv                         # DoS, Heartbleed
  Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv       # Web Attack
  Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv  # Infiltration
  Friday-WorkingHours-Morning.pcap_ISCX.csv                    # Bot
  Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv         # PortScan
  Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv             # DDoS
```

> **ต้องใช้ CICIDS2017 เวอร์ชัน *MachineLearningCVE*** (79 คอลัมน์) ซึ่งเป็นเวอร์ชันที่
> **ไม่มีคอลัมน์ `Timestamp`** โปรเจกต์นี้จึงใช้ลำดับแถวใน CSV เป็นลำดับเวลาแทน
> และมีการพิสูจน์ความถูกต้องอัตโนมัติทุกครั้งที่รัน
>
> ห้ามสลับ/เรียงลำดับแถวในไฟล์ CSV ดิบ เพราะลำดับแถวคือข้อมูลเวลาเพียงอย่างเดียวที่มี

`train.py` จะทำความสะอาดคอลัมน์และ normalize label ก่อนเทรน แนะนำให้สร้างแคชใหม่เฉพาะเมื่อไฟล์ CSV ดิบเปลี่ยนเท่านั้น:

```bash
python main.py --stage preprocess --refresh-cache
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

**ค่าที่คุมการเทรนจริงอยู่ใน `train.py` เอง** ไม่ใช่ `config.yaml` — `train.py` เป็น
training path เดียวของโปรเจกต์ (self-contained `CONFIG` dict, ไม่ import
`config.yaml` เลย) ปรับค่าอย่าง `hp_search_n_iter`, `rf_class_weight`,
`target_max_fpr`, RAM presets ฯลฯ ได้ 2 ทาง:

- ผ่าน CLI flag (ดูตาราง §10 CLI flags reference)
- แก้ `CONFIG` dict ที่ต้นไฟล์ `train.py` โดยตรง สำหรับค่าที่ไม่มี flag

[`src/config/config.yaml`](src/config/config.yaml) ยังมีอยู่ แต่ใช้เฉพาะ
`--stage eda` และ label/mode ที่ dashboard กับ FastAPI ใช้แสดงผล — **แก้ไฟล์นี้
ไม่มีผลกับ `--stage train` เลย**

```yaml
classification:
  mode: "multiclass"          # หรือ "binary" — ใช้โดย dashboard/API เท่านั้น

data:
  raw_dir: "data/raw"         # ใช้โดย --stage eda เท่านั้น
```

---

## 9. Quick Start · เริ่มต้นใช้งานเร็ว

ใช้ `main.py` เป็น entry point หลัก แต่ละการรันควรกำหนด `--run-name` เพื่อเขียน bundle แยกที่
`results/<run-name>/` จากนั้นตรวจ integrity และ promote เฉพาะ run ที่ผ่านเกณฑ์เป็น champion

```bash
# 1. สร้าง cache เฉพาะ CICIDS2017 (~2.5M แถว ใช้เวลาไม่กี่นาที)

python main.py --stage preprocess --refresh-cache

# 2. ตรวจ manifest และเทียบกับ split ที่ข้อมูลจริงให้ออกมา
python main.py --stage audit \
  --split-manifest configs/splits/cicids2017_temporal_70_30.json --verify-data

# 3. เทรนเต็มรูปแบบ 70/30 แบบเรียงตามเวลา
#    --accelerator gpu เร่งเฉพาะ xgboost/catboost/stacking
#    อีก 4 โมเดลใช้ CPU และ "reuse" artifact เดิมโดยไม่เทรนซ้ำ
python main.py --stage train --run-name thesis_local --model all \
  --split-manifest configs/splits/cicids2017_temporal_70_30.json \
  --preset 32gb --accelerator gpu

# 4. อ่าน metrics และ promote (เขียน champion + ranking 3 มุม)
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

# --preset only affects the HP search budget now -- the trainer always uses
# every row of data/processed/cicids2017_clean.parquet (2.5M rows) regardless
# of preset, via the temporal split manifest. Measured on a 31 GB machine
# training all 7 models with --preset 32gb; 16 GB should be enough.
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

### Stage: `preprocess` — Build the canonical clean cache

```bash
# Rebuild from data/raw (a few minutes for 2.5M rows).
python main.py --stage preprocess --refresh-cache

# Reuse and validate an existing cache.
python main.py --stage preprocess
```

**Output:** `data/processed/cicids2017_clean.parquet` (2,497,980 rows), plus a
console summary of row count, feature count, source files, and label
distribution.

The cache strips whitespace from column names, drops leaky and schema-only
columns, normalizes the 15 raw labels into 9 families, rejects remaining
missing/Inf feature values, and preserves metadata columns (`dataset_id`,
`source_file`, `capture_window`, `_row_index`) that are excluded from the 77
model features.

`_row_index` is the row's position in its **raw** CSV, recorded before any row is
dropped. It is the ordering key for the chronological split — see §15.

### Stage: `audit` — Validate the split manifest

```bash
# Structural validation only.
python main.py --stage audit --split-manifest configs/splits/cicids2017_temporal_70_30.json

# Also rebuild the split from the cache and compare against the manifest's
# expected per-class counts, and re-verify the capture chronology.
python main.py --stage audit \
  --split-manifest configs/splits/cicids2017_temporal_70_30.json --verify-data
```

`--verify-data` is what turns the manifest from a claim into a checked fact: it
confirms the split the data actually yields still matches the counts the
manifest advertises.

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

# Full GPU run. Only xgboost / catboost / stacking consult --accelerator.
python main.py --stage train --run-name thesis_local --model all \
  --split-manifest configs/splits/cicids2017_temporal_70_30.json \
  --preset 32gb --accelerator gpu
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
# All models of a run.
python main.py --stage explain --run-name thesis_local

# One model only.
python main.py --stage explain --run-name thesis_local --model cat
```

อ่านโมเดลจาก run bundle แล้วสร้าง test set ขึ้นใหม่ด้วย temporal split ชุดเดิม
จาก manifest เดียวกัน แถวที่นำมาอธิบายจึงเป็นแถวเดียวกับที่ใช้วัดผลเสมอ
ไม่มีทางหลุดออกจากกันได้

**Output (per model), inside the run bundle:**
- `results/<run-name>/shap/<model>/summary_bar.png`
- `results/<run-name>/shap/<model>/summary_<class>.png` (one per class — 9 classes)
- `results/<run-name>/shap/<model>/top_features.json`
- `results/<run-name>/shap/shap_report.md` (cross-model summary)

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
python main.py --stage audit --split-manifest configs/splits/cicids2017_temporal_70_30.json --verify-data

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
| `--split-manifest` | path | canonical temporal manifest | Chronological split manifest; `train.py` rejects any manifest whose `version` isn't a temporal one |
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
| `--refresh-cache` | flag | `false` | Rebuild the dataset's parquet cache from raw CSVs |
| `--verify-data` | flag | `false` | `--stage audit` only: rebuild the split and check it against the manifest's expected counts |
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

The Streamlit dashboard, the FastAPI backend, SHAP and batch inference all read the **published run** — the one named in `results/champion.json`, resolved by `published_run_dir()`. There is no second source of metrics to drift out of sync.

| Page | English | ภาษาไทย |
|---|---|---|
| 1. Dataset Overview | CICIDS2017 row counts and class balance | ภาพรวม + การกระจายของคลาส |
| 2. EDA | Distribution plots, correlations, missing values | กราฟ EDA |
| 3. Model Performance | Per-model metrics + confusion matrix + report | metric รายโมเดล + confusion matrix |
| 4. Model Comparison | Cross-model ranking + bar chart | เปรียบเทียบโมเดลทั้งหมด |
| 5. SHAP | Feature importance + per-class explanations | อธิบาย model ด้วย SHAP |
| 6. Predict New CSV | Upload → validate → predict + download | อัปโหลด CSV ใหม่เพื่อพยากรณ์ |

> Page 6 expects the 77-feature CICIDS2017 schema and uses model artifacts from the published run (see above), not a hardcoded `results/latest/`.

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
print(result.validation.message)        # OK -- 10000 rows, 77 columns
print(result.predictions.head())

# จาก DataFrame ที่โหลดอยู่แล้ว
df = pd.read_csv("my_flows.csv", encoding="latin-1")
result = predict_dataframe(df, model_name="xgboost")
print(result.predictions["predicted_label"].value_counts())
```

### Schema requirements · ข้อกำหนดของ CSV ที่อัปโหลด

- ต้องมีคอลัมน์ฟีเจอร์ครบตามที่อยู่ใน `feature_columns.json` ของ run bundle ที่ promote แล้ว
  (เช่น `results/cicids2017_temporal_v1/feature_columns.json` — 77 คอลัมน์)
- คอลัมน์เกินอนุญาตได้ (เช่น `Label` สำหรับ QA)
- ค่า Inf จะถูกแปลงเป็น NaN แล้วส่งให้ fitted imputer ใน model pipeline จัดการ โดยรักษา missingness semantics เดิม
- ระบบจะ strip whitespace นำหน้าชื่อคอลัมน์ให้อัตโนมัติ
- เมื่อใช้ published champion ระบบจะตรวจ bundle checksum ก่อนโหลด และส่ง `model_run_id`/`contract_version` กลับเพื่อ audit ได้

ดูตัวอย่างเพิ่มเติม: [`docs/inference_workflow.md`](docs/inference_workflow.md)

---

## 13. Testing · การทดสอบ

### 13.1 Unit tests (pytest) · ทดสอบหน่วยย่อย

```bash
# Run all 228 tests
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

### 13.4 Smoke test without the real dataset · ทดสอบโดยไม่มีข้อมูลจริง

การเทรนต้องใช้ CICIDS2017 ครบทั้ง 8 capture files เสมอ เพราะแต่ละคลาสโจมตี
อยู่ในไฟล์เดียว ขาดไฟล์ใดไฟล์หนึ่งจะหายไปทั้งคลาส `--stage train` จึงหยุด
พร้อมแจ้ง error แทนที่จะเทรนบนคอร์ปัสที่ไม่ครบ — ไม่มีเส้นทางเทรนด้วยข้อมูลสังเคราะห์

สิ่งที่รันได้โดยไม่ต้องมีข้อมูลจริง:

```bash
# ชุดทดสอบทั้งหมด ใช้ fixture ในตัว ไม่แตะ data/raw
pytest -q --no-cov

# EDA บนข้อมูลสังเคราะห์
python scripts/generate_sample.py --rows 2000
python main.py --stage eda --raw-dir data/sample

# ตรวจ manifest แบบ static (ไม่ต้องมี cache)
python main.py --stage audit --split-manifest configs/splits/cicids2017_temporal_70_30.json
```

### 13.5 Real-data smoke test · ทดสอบกับข้อมูลจริง

รันผ่าน CLI ทีละ stage ได้เลย ไม่ต้องเขียนสคริปต์เพิ่ม:

```bash
# 1. สร้าง cache (ต้องมี data/raw ครบ 8 ไฟล์)
python main.py --stage preprocess --refresh-cache

# 2. ตรวจว่า manifest ตรงกับ split ที่ข้อมูลให้ออกมาจริง
python main.py --stage audit   --split-manifest configs/splits/cicids2017_temporal_70_30.json --verify-data

# 3. เทรนโมเดลเดียวแบบเร็ว (ข้าม HP search และ trust checks)
python main.py --stage train --run-name smoke --model rf   --skip-tuning --skip-cv --skip-label-shuffle

# 4. อ่านผล + promote + SHAP
python main.py --stage evaluate --run-name smoke
python main.py --stage promote  --run-name smoke
python main.py --stage explain  --run-name smoke --model rf
```

> `--stage evaluate` และ `--stage explain` ถ้าไม่ใส่ `--run-name` จะอ่าน run ที่
> `results/champion.json` publish ไว้ ซึ่งเป็นตัวเดียวกับที่ dashboard, API
> และ batch inference ใช้

### 13.6 Dashboard smoke test · ทดสอบแดชบอร์ด

```bash
# Start dashboard in background
python main.py --stage dashboard

# Open http://localhost:8501
# Check that all 6 sidebar pages load without error.
# Page 6 needs a promoted run (run --stage train then --stage promote first).
```

### 13.7 Continuous testing · ทดสอบขณะพัฒนา

```bash
# Watch mode (install pytest-watch separately if you want this)
pip install pytest-watch
ptw -- --no-cov
```

---

## 14. Results · ผลลัพธ์

### Latest verified run: `cicids2017_temporal_v1`

รันเต็มบน CICIDS2017 ทั้งชุด ด้วย manifest `cicids2017_temporal_v1`:
train 1,748,579 / locked test 749,401 แถว (70.00 / 30.00), 77 features, 9 คลาส,
hyperparameter search ครบทุกโมเดล (RandomizedSearchCV, n_iter=20, scoring=`f1_macro`),
XGBoost/CatBoost/Stacking เทรนบน GPU ที่เหลือบน CPU

#### Dimension 1–2 · Classification & Detection

| Model | Accuracy | **Macro-F1** | Macro-F1 (reportable) | Macro P | **Macro R** | Binary FPR | Binary FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| **CatBoost** | 0.9953 | **0.8873** | 0.8613 | 0.8839 | **0.8970** | 0.00065 | 0.0160 |
| LightGBM | 0.9903 | 0.8774 | 0.8627 | **0.9403** | 0.8576 | **0.00022** | 0.0487 |
| XGBoost | 0.9906 | 0.8773 | 0.8542 | 0.8894 | 0.8889 | 0.00038 | 0.0429 |
| Random Forest | 0.9820 | 0.8404 | 0.8378 | 0.8846 | 0.8165 | 0.00530 | 0.0723 |
| Stacking | 0.9581 | 0.6822 | 0.7092 | 0.6551 | 0.8852 | 0.04072 | 0.0339 |
| Logistic Regression | 0.8428 | 0.5574 | 0.6348 | 0.5244 | 0.8338 | 0.16931 | 0.0254 |
| MLP | 0.9331 | 0.5369 | 0.6767 | 0.5180 | 0.7112 | 0.04721 | 0.1541 |

#### Per-class recall — ทำไม Accuracy ถึงหลอกตา

| Model | BENIGN | **Bot** | Brute Force | DDoS | DoS | Heartbleed | Infiltration | PortScan | Web Attack |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *n_test* | *621,680* | ***585*** | *2,745* | *38,405* | *58,119* | *4* | *11* | *27,209* | *643* |
| CatBoost | .9994 | **.1607** | .9985 | .9990 | .9571 | 1.000 | 1.000 | .9976 | .9611 |
| LightGBM | .9998 | **.1538** | .9960 | .9991 | .8891 | .7500 | 1.000 | .9955 | .9347 |
| XGBoost | .9996 | **.1607** | .9989 | .9992 | .8937 | 1.000 | 1.000 | .9975 | .9502 |
| Random Forest | .9947 | **.1504** | .9767 | .9983 | .8372 | .7500 | .7273 | .9975 | .9160 |
| Stacking | .9593 | **.1607** | .9971 | .9980 | .9069 | 1.000 | 1.000 | .9975 | .9471 |
| MLP | .9528 | **.1607** | .9621 | .9951 | .6583 | **0.000** | .7273 | .9962 | .9487 |

> **ผลสำคัญที่สุดของงานนี้:** ทุกโมเดลจับ `Bot` ได้เพียง ~15–16% เท่ากันหมด
> ความสม่ำเสมอนี้บ่งชี้ว่าเป็นคุณสมบัติของข้อมูล ไม่ใช่จุดอ่อนของโมเดล — Bot ทั้ง 1,948 flow
> อยู่ใน capture เดียว การแบ่งตามเวลาจึงเทรนด้วย flow ช่วงต้นและทดสอบด้วยช่วงท้าย
> ซึ่งโมเดลทั้งหมด generalize ข้ามไม่ได้
>
> **การแบ่งข้อมูลแบบสุ่มจะซ่อนผลนี้ทั้งหมด** เพราะ flow จาก burst เดียวกันจะกระจายอยู่ทั้งสองฝั่ง
> (นี่คือสิ่งที่เปเปอร์ CICIDS2017 ส่วนใหญ่รายงาน) จึงเป็นหลักฐานสนับสนุนการเลือกใช้ temporal split
>
> พูดอีกแบบ: **CatBoost ได้ accuracy 99.53% ขณะที่พลาด Bot ไป 84%** — ตรงกับที่อาจารย์เตือนเรื่อง
> class imbalance ทำให้ accuracy สูงโดยจับ attack บางชนิดไม่ได้

#### Dimension 4 · Computational Efficiency

วัด inference บน CPU ทุกโมเดล batch ละ 1,000 flows (30 รอบ) เพื่อให้เทียบกันได้

| Model | Trained on | Fit time | p50 | **p95** | Throughput | Size |
|---|---|---:|---:|---:|---:|---:|
| CatBoost | gpu | 45 s | 7.58 ms | **8.39 ms** | 131,975 /s | 2.5 MB |
| MLP | cpu | 819 s | 10.05 ms | 12.50 ms | 99,545 /s | 0.7 MB |
| XGBoost | gpu | 155 s | 43.42 ms | 54.10 ms | 23,030 /s | 15.8 MB |
| Random Forest | cpu | 315 s | — | 104.12 ms | — | 124.5 MB |
| Stacking | gpu | 2,755 s | 119.79 ms | 137.95 ms | 8,348 /s | 109.3 MB |
| LightGBM | cpu | 550 s | — | 144.02 ms | — | 54.9 MB |
| Logistic Regression | cpu | 654 s | 4.76 ms | 5.15 ms | 232,391 /s | 0.04 MB |

#### Rankings · การจัดอันดับ 3 มุม

| Ranking | Winner | Decided by | Excluded |
|---|---|---|---|
| **Overall Best** | `catboost` | f1_macro = 0.8873 | — |
| **Security-focused Best** | `catboost` | recall_macro = 0.8970 | MLP (FPR 4.7%), Stacking (4.1%), LogReg (16.9%) — เกินเพดาน 1% |
| **Deployment Best** | `catboost` | p95 = 8.39 ms | RF / Stacking / MLP / LogReg — ต่ำกว่าเพดานคุณภาพ f1_macro ≥ 0.867 |

CatBoost ชนะทั้งสามมุม ควรรายงานตามจริงว่า **ในข้อมูลชุดนี้ไม่มี trade-off ให้ต้องเลือก**
ไม่ใช่แกล้งสร้างข้อขัดแย้ง กลไกการคัดออกทำงานจริง (เห็นได้จากคอลัมน์ Excluded)
เกณฑ์ทั้งหมดประกาศไว้ล่วงหน้าใน `configs/ranking_policy.json` ก่อนรัน

ไฟล์หลักที่ตรวจสอบได้:

- `results/cicids2017_temporal_v1/metrics.json`, `report.md` และ per-model artifacts
- `results/cicids2017_temporal_v1/bundle_manifest.json` — checksums ของทุกไฟล์ใน bundle
- `results/champion.json` — champion pointer, selection evidence และ ranking ทั้ง 3 มุม

ข้อจำกัดที่ต้องรายงานคู่กันเสมอ: `Heartbleed` มี test เพียง 4 แถว และ `Infiltration` 11 แถว
ค่า recall ของสองคลาสนี้จึงเป็น anecdote ไม่ใช่ค่าประมาณที่เสถียร (ดู `f1_macro_reportable`)
และ MLP จับ Heartbleed ไม่ได้เลย (recall 0.000)

## 15. Architecture · สถาปัตยกรรม

### Module dependency · การอ้างถึงกันของ module

```
config + utils       ← imported by everyone
   |
data + features      ← depends on config + utils
   |
models               ← depends on data + features + config + utils
   |
explainability (xai) ← depends on models + data
   |
inference            ← depends on models + features (validator)
   |
pipelines            ← composes everything; called by main.py + dashboard
```

A lower layer never imports from a higher one — cycles impossible by construction.

### Data flow · การไหลของข้อมูล

```
data/raw/*.csv
   → schema validation + per-dataset parquet cache + provenance
     (เก็บ _row_index = ตำแหน่งแถวใน CSV ดิบ → ใช้เป็น ordering key)
   → split manifest (configs/splits/cicids2017_temporal_70_30.json)
   → ตรวจ chronology กับตารางเวลาโจมตีที่ CIC ประกาศ
   → chronological 70/30 ภายในทุกกลุ่ม (capture file, class)
   → เทียบ split ที่ได้กับ expected per-class counts ใน manifest
   → StratifiedKFold สำหรับ CV/tuning เฉพาะฝั่ง train
   → fitted preprocessing Pipeline + 7 candidate models
   → results/<run-id>/ checkpoints + metrics + explainability
   → bundle_manifest.json (SHA-256 integrity verification)
   → offline red-team checks on a copy of candidate data
   → policy-based promotion → results/champion.json (champion + 3 rankings)
   → inference.predictor / FastAPI / dashboard
```

### ทำไมถึงเปลี่ยนวิธีแบ่งข้อมูล · Why the split protocol changed

Source holdout กันข้อมูลรั่วด้วยการยกทั้ง capture file ไปฝั่ง train หรือ test ซึ่งทำได้
เพราะต้องมี attack family เดียวกันกระจายอยู่หลายไฟล์ แต่**ใน CICIDS2017
แต่ละคลาสโจมตีอยู่ในไฟล์เดียวเท่านั้น** (ดูตารางใน §2) การยกไฟล์ใดออกไปเป็น test
จึงทำให้คลาสนั้นไม่มีข้อมูลเทรนเลย — source holdout เป็นไปไม่ได้บนคอร์ปัส 2017

วิธีที่ใช้แทนคือแบ่ง **ตามเวลาภายในแต่ละ capture แยกทีละคลาส**: flow ช่วงต้น 70%
ไปเทรน ช่วงท้าย 30% ไปทดสอบ ทุกคลาสจึงมีอยู่ทั้งสองฝั่ง และไม่มี test flow ใด
เกิดก่อน train flow ของคลาสตัวเอง — เป็นโปรโตคอลที่เข้มกว่า random split ที่งานวิจัย
CICIDS2017 ส่วนใหญ่ใช้

ต้องแบ่งแยกทีละ `(ไฟล์, คลาส)` ไม่ใช่แค่ทีละไฟล์ เพราะ Heartbleed ทั้ง 11 flow
เกิดใน burst เดียวยาว ~20 นาที ที่ตำแหน่ง ~86% ของ capture วันพุธ ถ้าตัดที่ระดับไฟล์
มันจะไปอยู่ฝั่ง test ทั้งหมดและไม่เหลือให้เทรนเลย

**ที่มาของลำดับเวลา:** ไฟล์ CICIDS2017 แบบ *MachineLearningCVE* (79 คอลัมน์)
**ไม่มีคอลัมน์ `Timestamp`** จึงใช้ `_row_index` แทน และลำดับแถวนั้นเรียงตามเวลาจริง
ซึ่ง `temporal_split.validate_raw_label_chronology` **พิสูจน์** ด้วยการเทียบกับตารางเวลา
โจมตีที่ CIC ประกาศ ไม่ใช่แค่สมมติเอา — ทั้ง 7 คู่ที่เรียงลำดับได้ตรงกันหมด

| capture | ลำดับที่สังเกตได้ (median position) | ตารางที่ CIC ประกาศ |
|---|---|---|
| Wednesday | slowloris .073 < Slowhttptest .104 < Hulk .284 < GoldenEye .856 < Heartbleed .863 | 09:47 < 10:14 < 10:43 < 11:10 < 15:12 ✓ |
| Thursday-Web | Brute Force .241 < XSS .464 < Sql Injection .538 | 09:20 < 10:15 < 10:40 ✓ |
| Tuesday | FTP-Patator .089 < SSH-Patator .468 | 09:20 < 14:00 ✓ |

ข้อควรระวังที่ต้องเขียนในเล่ม: CICFlowMeter เขียน record ตอน flow **จบ** ลำดับนี้จึงเป็น
ลำดับการสิ้นสุดของ flow ไม่ใช่เวลาเริ่ม (เห็นชัดที่ DoS GoldenEye ซึ่งถือ connection ไว้นาน
record จึงลากยาวเลยหน้าต่างโจมตี 11:10–11:23 ไปมาก) ซึ่งเป็นลำดับที่ถูกต้องสำหรับ IDS อยู่แล้ว
เพราะ feature ของ flow จะมีก็ต่อเมื่อ flow จบแล้ว

### Key architectural rules · กติกาสถาปัตยกรรม

1. **Single `RANDOM_STATE = 42`** ทุก seed (numpy, sklearn, xgboost, ...) ใช้ค่าเดียวกัน
2. **Scaler อยู่ใน sklearn Pipeline** → fit ใหม่ทุก fold ของ CV → ไม่มี data leakage
3. **Config-driven** — paths, hyperparameters, class lists อยู่ใน `config.yaml` หมด
4. **Locked test ไม่เข้า train/CV** เด็ดขาด — แต่**บนโปรโตคอล 2017 ไม่ใช้ GroupKFold ตาม source**
   เพราะแต่ละคลาสอยู่ capture เดียว การ group ตาม source จะทำให้ทั้งคลาสตกอยู่ fold เดียว
   จึงใช้ `StratifiedKFold` แทน และ CV ใช้เพื่อ HP search + ตรวจความเสถียรเท่านั้น
   ตัวเลขหลักมาจาก locked test set เสมอ
5. **Deterministic split** — ไม่มี RNG เลย ผลขึ้นกับเนื้อข้อมูลอย่างเดียว สลับลำดับ input แล้วได้ split เดิม
6. **Immutable run bundles** — checkpoint เขียนแบบ atomic, resume ได้ และ checksum ก่อน promote/load
7. **Schema validation** at both ingestion (loader) and inference (validator)
8. **Fair tuning** — ทุกโมเดลที่ tune ได้ ใช้ method เดียวกัน, `n_iter` เท่ากัน, objective เดียวกัน
   (`f1_macro`) และ search space ขนาดเท่ากันพอดี (144 combinations) จึงสำรวจพื้นที่ในสัดส่วนเท่ากัน
   ห้ามใส่ `max_iter` (งบของ optimizer) หรือ `class_weight` (เป็นของ `--imbalance-strategy`) ลงใน grid
9. **Metric key เพิ่มได้ ห้าม rename** — `f1_macro` / `target_fpr` / `target_false_negatives`
   ถูกอ่านโดย promotion, `main.py`, dashboard และ scripts ด้วย `.get(key, 0.0)`
   การ rename จึงทำให้ promote ผิดตัวแบบเงียบๆ แทนที่จะ error

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

### `LabelEncoder is missing N expected classes`

ไม่ควรเกิดกับ manifest มาตรฐาน (`cicids2017_temporal_70_30.json`) เพราะ temporal
split รับประกันว่าทุกคลาสมีทั้งฝั่ง train และ test เสมอ (Heartbleed ก็ยังได้ 7
train / 4 test) ถ้าเจอ แปลว่า `--split-manifest` ที่ใช้อยู่ไม่ใช่ manifest ที่ถูกต้อง
ให้รัน `python main.py --stage audit --split-manifest <path> --verify-data` เพื่อตรวจ

### Memory error ระหว่างเทรน

`--preset` คุมเฉพาะขนาด HP search subset (`hp_search_subsample`,
`hp_search_n_iter`) ไม่ได้ลดขนาด training set — corpus เต็มที่ใช้เทรนคงที่ที่
1,748,579 แถวเสมอ (manifest-driven) ถ้า RAM ไม่พอ:

```bash
# ลด parallel HP search jobs ก่อน (ตัวที่กิน RAM มากสุด)
python main.py --stage train --preset 8gb ...

# หรือข้าม HP search ไปเลยระหว่างทดลอง
python main.py --stage train --skip-tuning --skip-cv ...
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
