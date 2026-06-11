# Implementation Log · บันทึกการพัฒนาโครงงาน

> **TH:** บันทึกขั้นตอนการพัฒนาตาม 5 phase ของ proposal โดยอธิบายว่าทำอะไร เพราะอะไร ที่ไฟล์ไหน และผลลัพธ์ออกมาเป็นแบบไหน
>
> **EN:** Build log structured around the proposal's 5 phases — explaining what was done, why, in which files, and what output was produced.

**ผู้พัฒนา:** นายศิรเชษฐ์ โชติฐากุลอนันต์ (66070191) · นายสุขุม ฤดีเมธากุล (66070315)
**อาจารย์ที่ปรึกษา:** ผศ.ดร.ประพันธ์ ปวรางกูร
**Proposal อนุมัติ:** 25 เมษายน 2569
**วันที่บันทึก:** 10 มิถุนายน 2569

---

## สารบัญ · Table of Contents

| Section | หัวข้อ · Topic |
|---|---|
| 0 | [ภาพรวมโครงงานตาม Proposal](#section-0-ภาพรวมโครงงานตาม-proposal) |
| 1 | [Proposal Phase 1: ศึกษาและเตรียมข้อมูล](#section-1-proposal-phase-1-ศึกษาและเตรียมข้อมูล) |
| 2 | [Proposal Phase 2: พัฒนาโมเดล](#section-2-proposal-phase-2-พัฒนาโมเดล) |
| 3 | [Proposal Phase 3: ประเมินผลและเปรียบเทียบ](#section-3-proposal-phase-3-ประเมินผลและเปรียบเทียบ) |
| 4 | [Proposal Phase 4: วิเคราะห์โมเดลด้วย SHAP](#section-4-proposal-phase-4-วิเคราะห์โมเดลด้วย-shap) |
| 5 | [Proposal Phase 5: แสดงผลและสรุป](#section-5-proposal-phase-5-แสดงผลและสรุป) |
| 6 | [Bonus Features ที่ทำเกิน Proposal](#section-6-bonus-features-ที่ทำเกิน-proposal) |
| 7 | [วิธีรันโครงงานทีละขั้น](#section-7-วิธีรันโครงงานทีละขั้น) |
| 8 | [ผลลัพธ์สำคัญที่ต้องอธิบายอาจารย์](#section-8-ผลลัพธ์สำคัญที่ต้องอธิบายอาจารย์) |

---

## Section 0: ภาพรวมโครงงานตาม Proposal

### วัตถุประสงค์ (ตาม proposal)

> สร้างโมเดล Machine Learning สำหรับจำแนกประเภทการโจมตี (attack type classification) จากข้อมูล network traffic
> เน้นการทดลองและเปรียบเทียบประสิทธิภาพของโมเดลต่าง ๆ
> วิเคราะห์ feature ที่มีผลต่อการตัดสินใจของโมเดล (Explainable AI)

### ตาราง Mapping: Proposal → Implementation

| Proposal Phase | สิ่งที่ proposal บอกให้ทำ | ทำที่ไฟล์ไหน | ผลลัพธ์อยู่ที่ไหน |
|---|---|---|---|
| Phase 1 | ศึกษา + เตรียมข้อมูล | `src/data/`, `src/features/cleaning.py`, `src/pipelines/eda.py`, `src/pipelines/preprocess.py` | `results/figures/`, `results/metrics/eda_summary.json`, `data/processed/` |
| Phase 2 | พัฒนาโมเดล LR + RF + MLP | `src/models/*.py`, `src/pipelines/train.py` | `models/*.joblib` |
| Phase 3 | ประเมินผลด้วย Acc/P/R/F1 + Confusion matrix + เปรียบเทียบ | `src/evaluation/`, `src/pipelines/evaluate.py` | `results/metrics/*.json`, `results/figures/confusion_matrix_*.png`, `reports/model_comparison.md` |
| Phase 4 | วิเคราะห์ด้วย Feature importance / SHAP | `src/explainability/shap_analyzer.py`, `src/pipelines/explain.py` | `results/shap/<model>/`, `results/shap/shap_report.md` |
| Phase 5 | แสดงผลในรูปแบบที่เข้าใจง่าย + อาจมี dashboard | `dashboard/`, `reports/` | Streamlit app + Markdown reports |

---

## Section 1: Proposal Phase 1 — ศึกษาและเตรียมข้อมูล

### 1.1 ดาวน์โหลดและตรวจสอบ Dataset

**Dataset:** CICIDS2017 (Canadian Institute for Cybersecurity)
**ที่มา:** https://www.unb.ca/cic/datasets/ids-2017.html
**ขนาด:** 8 ไฟล์ CSV รวม ~880 MB extracted, ~2.83 ล้าน flow records

ไฟล์ที่ใช้:

```
data/raw/
├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv       (77 MB · DDoS)
├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv   (77 MB · PortScan)
├── Friday-WorkingHours-Morning.pcap_ISCX.csv              (58 MB · Bot)
├── Monday-WorkingHours.pcap_ISCX.csv                      (177 MB · BENIGN เท่านั้น)
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv (83 MB · Infiltration)
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv (52 MB · Web Attack)
├── Tuesday-WorkingHours.pcap_ISCX.csv                     (135 MB · FTP/SSH-Patator)
└── Wednesday-workingHours.pcap_ISCX.csv                   (225 MB · DoS + Heartbleed)
```

### 1.2 EDA (Exploratory Data Analysis)

**คำสั่ง:**
```powershell
python main.py --stage eda
```

**ทำอะไร:**
1. โหลด CSV ทั้ง 8 ไฟล์ผ่าน `src/data/loader.py::load_raw()`
2. Concat รวมเป็น DataFrame เดียว (2.83M rows × 79 columns)
3. Strip whitespace นำหน้าชื่อคอลัมน์ (CICIDS มี bug ติดมา)
4. Subsample แบบ stratified เหลือ 300,000 rows
5. คำนวณสถิติ: missing values, infinite values, duplicate rows, class distribution
6. สร้างกราฟ 4 ภาพ

**ผลลัพธ์:**

`results/metrics/eda_summary.json`:
```json
{
  "shape": [284786, 79],
  "n_features": 78,
  "missing_total": 0,
  "infinite_total": 0,
  "duplicate_row_count": 0,
  "label_distribution": {
    "BENIGN": 233327,
    "DoS": 21020,
    "PortScan": 15328,
    "DDoS": 13567,
    "Brute Force": 1101,
    "Web Attack": 230,
    "Bot": 208,
    "Infiltration": 4,
    "Heartbleed": 1
  }
}
```

**กราฟที่สร้าง:**
- `results/figures/class_distribution.png` — bar chart แสดงจำนวนแต่ละ class
- `results/figures/missing_value_audit.png` — bar chart แสดงคอลัมน์ที่มี missing
- `results/figures/correlation_heatmap.png` — heatmap 78×78 features
- `results/figures/feature_distributions.png` — histogram ของ 12 feature สำคัญ

**ข้อสังเกตจาก EDA:**
- ข้อมูล Imbalanced มาก — BENIGN คิดเป็น 82% ของทั้งหมด
- Heartbleed มีแค่ 1 row, Infiltration มี 4 rows — น้อยเกินไปสำหรับ train
- Web Attack มี 3 sub-type (XSS, SQL Injection, Brute Force) ใช้ byte `\x96` เป็น separator (Windows-1252 en-dash)

### 1.3 Preprocessing

**คำสั่ง:**
```powershell
python main.py --stage preprocess
```

**ทำอะไร (ตามลำดับ):**

1. **Load** — โหลด 8 CSVs, stratified subsample ที่ 300K
2. **Clean** — `src/features/cleaning.py`
   - Strip whitespace ชื่อคอลัมน์
   - ลบ duplicate column `Fwd Header Length.1` (CICIDS bug)
   - แทน Inf ด้วย NaN
   - ลบ rows ที่มี NaN (488 rows)
   - ลบ exact duplicate rows (14,898 rows)
3. **Label mapping** — `src/data/label_mapping.py`
   - แปลง raw labels 15 รูปแบบ → mapped 10 classes
   - เช่น `"DoS Hulk"`, `"DoS GoldenEye"`, `"DoS slowloris"`, `"DoS Slowhttptest"` → `"DoS"`
   - เช่น `"FTP-Patator"`, `"SSH-Patator"` → `"Brute Force"`
   - เช่น `"Web Attack \x96 XSS"`, `"\x96 Sql Injection"`, `"\x96 Brute Force"` → `"Web Attack"`
4. **Filter rare classes** — drop คลาสที่มี < 10 samples (Heartbleed, Infiltration, Other)
   - เหตุผล: stratified split + 5-fold CV ต้องการอย่างน้อย 10 samples/class
5. **Label encoding** — `LabelEncoder` แปลง string → integer
6. **Train/Val/Test Split** — 60/20/20 แบบ stratified
   - test_size=0.2 จากทั้งหมด
   - val_size=0.2 จาก (train+val) → 16% ของทั้งหมด
7. **Save artefacts**
   - `data/processed/train.parquet` (182,259 rows)
   - `data/processed/val.parquet` (45,565 rows)
   - `data/processed/test.parquet` (56,957 rows)
   - `data/processed/feature_names.json` (77 features)
   - `data/processed/label_classes.json` (7 classes)
   - `models/label_encoder.joblib`

**คลาสสุดท้ายที่ใช้ train (7 คลาส):**

```
BENIGN · Bot · Brute Force · DDoS · DoS · PortScan · Web Attack
```

> **หมายเหตุ:** Heartbleed (1 row) และ Infiltration (4 rows) ถูกตัดออกตาม proposal ที่อนุญาตให้ "ศึกษาเฉพาะบางประเภทของการโจมตี"

---

## Section 2: Proposal Phase 2 — พัฒนาโมเดล

### 2.1 โมเดลตาม Proposal

Proposal กำหนดให้พัฒนา 3 โมเดลพื้นฐาน:

| โมเดล | ประเภท | ไฟล์ |
|---|---|---|
| **Logistic Regression** | Linear baseline | `src/models/logistic_regression.py` |
| **Random Forest** | Tree ensemble | `src/models/random_forest.py` |
| **Neural Network (MLP)** | Multi-layer perceptron | `src/models/mlp.py` |

### 2.2 Architecture

ทุกโมเดล inherit จาก `BaseModel` (`src/models/base.py`) — interface เดียวกัน:

```
class BaseModel:
    def _build_estimator()       # subclass returns sklearn-compatible classifier
    def build()                   # wraps in sklearn Pipeline(scaler + clf)
    def fit(X, y)
    def predict(X)
    def predict_proba(X)
    def save() / load()
```

**ทำไม wrap ด้วย Pipeline:** เพื่อให้ scaler **fit ใหม่ทุก CV fold** ป้องกัน data leakage (ADR-006)

### 2.3 รัน Training

**คำสั่ง:**
```powershell
# Train ทุกโมเดล (LR + RF + MLP + bonus 3 ตัว)
python main.py --stage train

# หรือ train เฉพาะตัวเดียว
python main.py --stage train --model lr --skip-tuning
python main.py --stage train --model rf --skip-tuning
python main.py --stage train --model nn --skip-tuning  # MLP
```

**Hyperparameters (baseline ใน config.yaml):**

```yaml
logistic_regression:
  C: 1.0                       # regularization strength
  solver: "lbfgs"
  class_weight: "balanced"     # ชดเชย imbalanced classes
  max_iter: 1000

random_forest:
  n_estimators: 200
  max_depth: null              # ไม่จำกัด depth
  class_weight: "balanced"
  n_jobs: -1                   # ใช้ทุก CPU core

mlp:
  hidden_layer_sizes: [128, 64]
  activation: "relu"
  solver: "adam"
  max_iter: 200
  early_stopping: true         # ป้องกัน overfitting
```

### 2.4 ผลลัพธ์การ Train

ไฟล์ที่สร้าง (ต่อโมเดล):
- `models/<name>.joblib` — fitted Pipeline (scaler + classifier)
- `results/metrics/<name>_val.json` — validation metrics
- `results/metrics/val_summary.csv` — ranking ของทุกโมเดลรวม

---

## Section 3: Proposal Phase 3 — ประเมินผลและเปรียบเทียบ

### 3.1 รัน Evaluation

**คำสั่ง:**
```powershell
python main.py --stage evaluate
```

**ทำอะไร:**
1. โหลด `test.parquet` (56,957 rows ที่โมเดลไม่เคยเห็น)
2. โหลดแต่ละ `<model>.joblib`
3. คำนวณ metric ทุกตัว:
   - Accuracy
   - Precision (weighted, macro, per-class)
   - Recall (weighted, macro, per-class)
   - F1-score (weighted, macro, per-class)
   - ROC AUC (one-vs-rest, macro-averaged)
   - Matthews Correlation Coefficient
4. สร้าง confusion matrix สำหรับแต่ละโมเดล
5. รวมผลเป็นตารางเปรียบเทียบ

### 3.2 Metric Definitions (สำหรับอธิบายอาจารย์)

| Metric | สูตร | ความหมาย |
|---|---|---|
| Accuracy | (TP + TN) / Total | สัดส่วน predict ถูกทั้งหมด |
| Precision | TP / (TP + FP) | จากที่ predict ว่าเป็น attack ถูกจริงกี่ % |
| Recall | TP / (TP + FN) | จาก attack จริง ถูก detect ได้กี่ % |
| F1 | 2 × P × R / (P + R) | harmonic mean ของ P+R |
| F1-macro | mean(F1 per class) | เฉลี่ยเท่ากันทุก class — ดีสำหรับ imbalanced |
| F1-weighted | Σ(F1_c × n_c) / N | weighted ตามขนาด class |

**ทำไมต้องดู F1-macro สำหรับ Imbalanced data:**
F1-weighted สูงเพราะ BENIGN เยอะ — โมเดลที่ทาย BENIGN หมดได้ accuracy 82% ทันที **ต้องดู F1-macro เพื่อเห็นประสิทธิภาพกับ class น้อย**

### 3.3 ผลการเปรียบเทียบ (Test Set)

ดูไฟล์เต็มที่ `reports/model_comparison.md`

#### โมเดลตาม Proposal

| Rank | Model | Accuracy | F1-weighted | F1-macro | สรุป |
|---|---|---|---|---|---|
| 1 | random_forest | 0.9983 | 0.9984 | 0.9456 | ✅ ดีมาก |
| 2 | mlp | 0.9820 | 0.9817 | 0.8065 | ⚠️ พอใช้ (Web Attack F1=0.23) |
| 3 | logistic_regression | 0.8509 | 0.8927 | **0.5652** | ❌ ไม่เพียงพอ |

#### โมเดล Bonus (ทำเพิ่มเพื่อเปรียบเทียบ)

| Rank | Model | Accuracy | F1-weighted | F1-macro |
|---|---|---|---|---|
| 1 (overall) | **lightgbm** | 0.9989 | 0.9989 | **0.9643** |
| 2 (overall) | xgboost | 0.9989 | 0.9989 | 0.9571 |
| 4 (overall) | catboost | 0.9973 | 0.9975 | 0.9212 |

### 3.4 Per-class breakdown (ตัวอย่าง Random Forest — โมเดล proposal)

|             | precision | recall | f1 |
|:------------|----------:|-------:|---:|
| BENIGN      | 0.9991    | 0.9990 | 0.9990 |
| Bot         | 0.6304    | 0.6905 | 0.6591 |
| Brute Force | 1.0000    | 0.9818 | 0.9908 |
| DDoS        | 1.0000    | 0.9985 | 0.9993 |
| DoS         | 0.9960    | 0.9950 | 0.9955 |
| PortScan    | 0.9951    | 0.9990 | 0.9971 |
| Web Attack  | 0.9783    | 0.9783 | 0.9783 |

**ข้อสังเกต:**
- RF ทำงานดีกับทุก class ยกเว้น Bot (F1=0.66)
- Bot ยากเพราะมีแค่ 208 rows ในชุดเทรน (208 / 284,786 = 0.07%)

### 3.5 Confusion Matrix

ไฟล์: `results/figures/confusion_matrix_<model>.png`

ค่าใน cell เป็น **normalized recall** — แถวรวมเท่ากับ 1.0 (1 = ทำนายถูก 100%)

**Random Forest:**
```
            BENIGN  Bot  Brute Force  DDoS  DoS  PortScan  Web Attack
BENIGN      1.00    0    0            0     0    0         0
Bot         0.31    0.69 0            0     0    0         0
Brute Force 0.02    0    0.98         0     0    0         0
...
```

---

## Section 4: Proposal Phase 4 — วิเคราะห์โมเดลด้วย SHAP

### 4.1 ทำไมต้อง SHAP

Proposal กำหนดให้วิเคราะห์ feature importance — โดยแนะนำ:
- ใช้ feature_importance ใน Random Forest
- หรือใช้ SHAP (เลือก SHAP เพราะ:)
  1. ใช้ได้กับทุกโมเดล (RF, MLP, LR — ไม่ใช่แค่ tree)
  2. อธิบายได้ทั้ง global (ภาพรวม) และ local (รายตัว)
  3. มี per-class explanation ที่ feature_importance ปกติทำไม่ได้

### 4.2 รัน SHAP

**คำสั่ง:**
```powershell
# ทุกโมเดล
python main.py --stage explain --model all

# ตัวเดียว
python main.py --stage explain --model rf
```

**Explainer ที่ใช้:**

| โมเดล | Explainer | เหตุผล |
|---|---|---|
| Random Forest | TreeExplainer | exact + fast |
| MLP | KernelExplainer | model-agnostic (slow แต่ใช้ได้) |
| Logistic Regression | KernelExplainer | same as MLP |
| XGBoost, LightGBM | TreeExplainer | |
| CatBoost | CatBoost native `get_feature_importance(type="ShapValues")` — ไม่ใช้ shap.TreeExplainer เพราะ segfault บน Windows |

### 4.3 ผลลัพธ์ (ต่อโมเดล)

โฟลเดอร์ `results/shap/<model>/`:
- `summary_bar.png` — top 20 features (mean |SHAP|)
- `summary_BENIGN.png`, `summary_Bot.png`, ... 7 PNG ต่อ class (beeswarm)
- `top_features.json` — top 10 features overall + per class

รายงานรวม: `results/shap/shap_report.md`

### 4.4 Top Features ที่ทุกโมเดลเลือก (Sanity check ผ่าน)

จากการวิเคราะห์ SHAP ทุกโมเดลใช้ feature ที่ **มีความหมายทาง security**:

| Feature | ทำไมถึงสำคัญ |
|---|---|
| `Destination Port` | ระบุบริการ — port 80 = web, 22 = SSH, 21 = FTP |
| `Init_Win_bytes_forward/backward` | TCP window size — DoS attacks มัก window ผิดปกติ |
| `min_seg_size_forward` | Packet segmentation pattern — เห็นใน scan tools |
| `Flow IAT Min` | Inter-Arrival Time น้อย = traffic เร็วผิดปกติ |
| `PSH Flag Count` | TCP control flag — สแกนพอร์ตเห็น pattern นี้ |

**ไม่พบ artifact features** เช่น Flow ID, timestamp, IP-based features → ยืนยันว่าโมเดล **เรียนรู้ network behavior จริง** ไม่ใช่ lab artifact

### 4.5 Per-Class Insights (จาก SHAP ของ Random Forest)

| Class | Top 3 features |
|---|---|
| BENIGN | Destination Port, Init_Win_bytes_forward, Init_Win_bytes_backward |
| Bot | Destination Port, Init_Win_bytes_backward, Init_Win_bytes_forward |
| Brute Force | Destination Port, min_seg_size_forward, Bwd Packet Length Min |
| DDoS | Destination Port, Fwd Packet Length Max, Avg Fwd Segment Size |
| DoS | Destination Port, Bwd Packet Length Std, Init_Win_bytes_backward |
| PortScan | Total Length of Fwd Packets, PSH Flag Count, Bwd Packet Length Min |
| Web Attack | Init_Win_bytes_backward, Fwd IAT Std, Init_Win_bytes_forward |

---

## Section 5: Proposal Phase 5 — แสดงผลและสรุป

### 5.1 Streamlit Dashboard

**คำสั่งรัน:**
```powershell
streamlit run dashboard/app.py
```

เปิด browser ที่ `http://localhost:8501`

**6 หน้าที่มี:**

| หน้า | ไฟล์ | แสดงอะไร |
|---|---|---|
| Landing | `dashboard/app.py` | KPI strip, status pills, nav cards |
| 1. Dataset Overview | `pages/01_Dataset_Overview.py` | row counts, class balance, bar chart |
| 2. EDA | `pages/02_EDA.py` | 4 tabs (class / missing / correlation / distributions) |
| 3. Model Performance | `pages/03_Model_Performance.py` | radar chart + confusion matrix + classification report |
| 4. Model Comparison | `pages/04_Model_Comparison.py` | leaderboard + grouped bar + download CSV |
| 5. SHAP | `pages/05_SHAP.py` | top features overall + per class + beeswarm |
| 6. Predict New CSV | `pages/06_Predict_New_CSV.py` | upload CSV → validate schema → predict + probabilities |

### 5.2 Markdown Reports (สำหรับใส่ในรายงาน senior project)

| ไฟล์ | เนื้อหา |
|---|---|
| `reports/model_comparison.md` | ตารางเปรียบเทียบทุกโมเดล + per-class breakdown |
| `results/shap/shap_report.md` | top features ของแต่ละโมเดล + per-class explanation |
| `docs/implementation_log.md` | ไฟล์นี้ — บันทึกการพัฒนาตาม 5 phase |

### 5.3 สรุปและข้อจำกัด (สำหรับ defense)

**สรุปข้อค้นพบ:**

1. **Linear model ไม่เพียงพอ** — LR ได้ F1-macro แค่ 0.57 เพราะ CICIDS attacks มี non-linear decision boundary
2. **Tree-based models ดีกว่ามาก** — RF ได้ F1-macro 0.95 (เพิ่มจาก 0.57 ของ LR = 66% improvement)
3. **NN (MLP) ทำงานได้แต่ไม่เก่ง class น้อย** — F1-macro 0.81, Web Attack F1 แค่ 0.23
4. **Class น้อยที่สุดยากที่สุด** — Bot (208 rows) ทุกโมเดลทำได้ F1 0.60–0.77 ขณะ DDoS (13K rows) ได้ ~1.00

**ข้อจำกัดของโครงงาน (ต้องระบุใน defense):**

1. **Lab dataset** — CICIDS2017 เป็น traffic จำลอง ไม่ใช่ enterprise จริง
2. **Old dataset** — 2017 attack patterns เปลี่ยนไปใน 9 ปี
3. **Class imbalance** — แม้ใช้ class_weight='balanced' แล้วก็ยังไม่แก้หมด
4. **ตัดคลาสที่น้อยเกินไป** — Heartbleed, Infiltration ตัดออกเพราะ < 10 samples
5. **ไม่ได้ทดสอบกับ traffic จริง** — proposal ระบุชัดว่า "ไม่ได้พัฒนาเป็นระบบใช้งานจริง"

**Future Work:**
- Cross-dataset evaluation (CICIDS2018, UNSW-NB15)
- Reproduce attack ด้วย nmap + CICFlowMeter
- Resampling techniques (SMOTE, ADASYN) สำหรับ class น้อย
- Deep learning models (CNN, LSTM) สำหรับ flow sequence

---

## Section 6: Bonus Features ที่ทำเกิน Proposal

> ⚠️ ส่วนนี้ **เกิน scope ของ proposal** — proposal ระบุชัด "ไม่ได้พัฒนาเป็นระบบใช้งานจริงเต็มรูปแบบ" และจำกัด 3 โมเดลพื้นฐาน
>
> สิ่งเหล่านี้ทำเพื่อให้โครงงานโดดเด่นขึ้น — **ทำหรือไม่ทำขึ้นกับการอนุมัติของอาจารย์**

### 6.1 โมเดลเพิ่มเติม (ทำแล้ว)

นอกจาก LR, RF, MLP — เพิ่มอีก 3 โมเดล:

| Model | F1-weighted | F1-macro | ใช้ตัดสินใจอะไร |
|---|---|---|---|
| XGBoost | 0.9989 | 0.9571 | Gradient boosting baseline |
| LightGBM | 0.9989 | **0.9643** | **Best overall model** |
| CatBoost | 0.9975 | 0.9212 | Compare boosting libraries |

### 6.2 โมเดลที่ยังไม่ทำ (รอ approval)

ตามสเปกผู้พัฒนาขอเสนอเพิ่ม (ยังไม่ทำ):
- Decision Tree (single CART)
- Extra Trees (extra randomization)
- K-Nearest Neighbours
- Support Vector Machine (LinearSVC)
- SMOTE / ADASYN / Random Under Sampling

### 6.3 Infrastructure / MLOps (รอ approval)

ยังไม่ทำ:
- FastAPI REST API (`/predict`, `/predict_batch`, `/health`)
- Docker + docker-compose
- Makefile
- GitHub Actions CI/CD

---

## Section 7: วิธีรันโครงงานทีละขั้น

### Step 0: Setup environment

```powershell
# 1. clone (ถ้ายังไม่มี)
cd C:\Users\ks\Documents\GitHub\cyber_attack_classification

# 2. activate venv
.\.venv\Scripts\Activate.ps1

# 3. ตรวจ environment
python -c "import sklearn, xgboost, lightgbm, catboost, shap, streamlit; print('OK')"
```

### Step 1: ดาวน์โหลด CICIDS2017 (one-time)

ดูคู่มือใน `docs/dataset_preparation.md`

```powershell
ls data\raw\*.csv
# คาดหวัง: 8 ไฟล์
```

### Step 2: รัน full pipeline

```powershell
# วิธีที่ 1: รันทีละ stage (เห็นแต่ละขั้นชัด)
python main.py --stage eda          # ~30 sec
python main.py --stage preprocess   # ~1-2 min
python main.py --stage train        # ~5-10 min (no tuning)
python main.py --stage evaluate     # ~1 min
python main.py --stage explain      # ~5 min (SHAP ทุกโมเดล)

# วิธีที่ 2: รันทั้งหมดพร้อมกัน
python main.py --stage all
```

### Step 3: เปิด Dashboard

```powershell
streamlit run dashboard/app.py
```

### Step 4: รัน Tests

```powershell
pytest --no-cov
# คาดหวัง: 61 passed
```

### Step 5: ทำนายผลข้อมูลใหม่

```powershell
python main.py --stage predict `
    --input  path\to\new_traffic.csv `
    --output predictions.csv `
    --model  lightgbm
```

---

## Section 8: ผลลัพธ์สำคัญที่ต้องอธิบายอาจารย์

### 8.1 ตัวเลขที่ต้องจำ

| ตัวเลข | ความหมาย |
|---|---|
| **2,830,743** | จำนวน rows ดิบใน CICIDS2017 |
| **300,000** | rows ที่ใช้จริง (stratified subsample) |
| **284,786** | rows หลัง cleaning + duplicate removal |
| **7** | classes ที่ใช้ train (BENIGN + 6 attack types) |
| **77** | features หลัง drop duplicate column |
| **6** | models ที่ train (LR, RF, MLP ตาม proposal + 3 bonus) |
| **61** | unit tests ที่ผ่าน |
| **0.9989** | F1-weighted ดีที่สุด (LightGBM) |
| **0.5652** | F1-macro ของ LR — ยืนยันว่า linear model ไม่พอ |

### 8.2 คำตอบสำหรับคำถามที่อาจารย์น่าจะถาม

**Q: ทำไมโครงงานเทรนแค่ 7 classes ไม่ใช่ 10 ตาม CICIDS?**

A: 3 classes (Heartbleed, Infiltration, Other) มี samples น้อยกว่า 10 rows ในชุดข้อมูล stratified subsample ทำให้ไม่สามารถ stratified split + 5-fold CV ได้ proposal อนุญาตให้ "ศึกษาเฉพาะบางประเภท"

**Q: ทำไม Logistic Regression ห่วยกว่าตัวอื่น?**

A: LR เป็น linear model — สามารถวาดเส้นแบ่งแบบ hyperplane เท่านั้น แต่ network attack features มี non-linear interaction (เช่น `Destination Port` มี categorical pattern, `Flow Duration` มี long-tail distribution) ทำให้ LR ไม่สามารถจับ pattern ได้ ดู F1-macro = 0.57 vs RF = 0.95

**Q: ทำไมเลือก SHAP ไม่ใช่ feature_importance ปกติ?**

A: feature_importance ใช้ได้แค่กับ tree-based models (RF, XGB, etc.) แต่ proposal บอกให้วิเคราะห์ทุกโมเดล — รวมถึง MLP และ LR ด้วย SHAP ใช้ได้ทุกโมเดลผ่าน KernelExplainer และให้ per-class explanation ที่ feature_importance ไม่มี

**Q: รู้ได้ไงว่าโมเดลใช้งานได้จริง?**

A: 3 วิธี (ดู `docs/testing.md`):
1. **SHAP sanity check** — top features ทุกตัวคือ network behavior จริง (Destination Port, TCP flags) ไม่ใช่ lab artifact
2. **Confusion matrix** — class ที่มี samples พอ (>1000) ทุกตัวได้ F1 > 0.97
3. **Cross-dataset test** (อนาคต) — รันบน CICIDS2018 ดู accuracy gap

**Q: ทำไม Bot ได้ F1 ต่ำ?**

A: Bot มีแค่ 208 rows จาก 284,786 = 0.07% ของข้อมูล ทำให้โมเดลเรียนรู้ pattern ไม่พอ ใน confusion matrix Bot มักถูกทำนายเป็น BENIGN (FN สูง)

### 8.3 ไฟล์สำคัญที่ต้องเปิดให้อาจารย์ดู

| สิ่งที่อาจารย์ขอ | ไฟล์ที่ตอบ |
|---|---|
| "ขอดู metrics" | `reports/model_comparison.md` |
| "ขอดู confusion matrix" | `results/figures/confusion_matrix_*.png` |
| "ขอดู SHAP" | `results/shap/shap_report.md` + `results/shap/<model>/` |
| "ขอดู code" | `src/` (มี `architecture.md` อธิบาย) |
| "รัน dashboard ให้ดู" | `streamlit run dashboard/app.py` |
| "Tests ทำงานยัง" | `pytest --no-cov` |
| "ทดลองทำนายข้อมูลใหม่" | Dashboard หน้า 6 หรือ `python main.py --stage predict ...` |

---

## ภาคผนวก: คำสั่งทดสอบทั้งหมด

ดูใน `docs/testing.md` — มีคู่มือทดสอบ 13 ขั้นตอน ภาษาไทย+อังกฤษ ครอบคลุม:
1. Environment check
2. Unit tests (61 tests)
3. Code quality (ruff + black)
4. Synthetic data smoke test
5. Real dataset preparation check
6. Quick real-data smoke test
7. Per-stage testing (eda/preprocess/train/evaluate/explain/predict)
8. Dashboard testing
9. Programmatic API testing
10. Full test suite summary
11. Troubleshooting
12. CI-style one-liner
13. Pre-push checklist
