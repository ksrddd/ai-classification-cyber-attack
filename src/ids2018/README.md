# CSE-CIC-IDS2018 Pipeline (แยกขาดจาก CICIDS2017)

โมดูลนี้เป็น **pipeline อิสระ** สำหรับชุดข้อมูล CSE-CIC-IDS2018 โดยเฉพาะ
ไม่มีการ import ฟังก์ชัน preprocessing หรือ feature names จากฝั่ง CICIDS2017
(`src/data`, `src/features`, `src/models`) แม้แต่จุดเดียว — เพราะสองชุดข้อมูล
ใช้ชื่อคอลัมน์คนละแบบ (`Tot Fwd Pkts` vs `Total Fwd Packets`) และสะกดชื่อ label
ต่างกัน การเอามาปนกันจะทำให้ข้อมูลเสียหายแบบเงียบ ๆ

## โครงสร้างไฟล์

| ไฟล์ | หน้าที่ |
|---|---|
| `config.py` | ค่าคงที่ทั้งหมด: path, sample size, hyper-parameters, label aliases |
| `data_loader.py` | Stratified sampling แบบ 2 pass บนข้อมูล 13M แถวโดยไม่โหลดเข้า RAM ทั้งก้อน |
| `preprocessing.py` | ตัดคอลัมน์ขยะ, จัดการ inf/NaN, split 70/30, scaling, label encoding |
| `models.py` | ตัวสร้างโมเดลทั้ง 7 ตัว + การบันทึกลงดิสก์ |
| `evaluate.py` | Metrics, confusion matrix, ตารางเปรียบเทียบ |
| `train_ids2018.py` | CLI ที่ร้อยทุกขั้นตอนเข้าด้วยกัน |

## วิธีใช้

```bash
# ตรวจ data path ก่อน (ทำครบทุกขั้นยกเว้นการเทรน)
python -m src.ids2018.train_ids2018 --dry-run

# รันเต็ม 7 โมเดล
python -m src.ids2018.train_ids2018 --raw-dir D:/CSE-CIC-IDS2018

# เทรนเฉพาะบางโมเดล
python -m src.ids2018.train_ids2018 --models xgboost lightgbm catboost

# ใช้ RobustScaler แทน StandardScaler
python -m src.ids2018.train_ids2018 --scaler robust

# ใช้ GPU
python -m src.ids2018.train_ids2018 --accelerator gpu
```

ดูตัวเลือกทั้งหมดด้วย `--help`

### เทียบกับ CLI ของฝั่ง CICIDS2017 (`main.py`)

ไปป์ไลน์นี้แยกขาดจาก 2017 จึงมี CLI คนละชุด แฟล็กที่พอเทียบกันได้:

| `main.py` (2017) | `train_ids2018` (2018) |
|---|---|
| `--stage train` | ไม่มี — สคริปต์นี้ทำ train อย่างเดียว (`--dry-run` = preprocessing เท่านั้น) |
| `--model all` | `--models` (ค่า default = ครบทั้ง 7 อยู่แล้ว) |
| `--accelerator gpu` | `--accelerator gpu` (ชื่อเดียวกัน) |
| `--gpu-devices 0` | `--gpu-devices 0` (ชื่อเดียวกัน) |
| `--preset full` | `--sample-size` |
| `--run-name X` | `--output-dir` / `--models-dir` (default แยกตามขนาดให้แล้ว) |
| `--split-manifest ...` | ไม่มี — 2018 แบ่ง stratified 70/30 ของตัวเอง |
| `--profile overnight`, `--skip-tuning` | ไม่มี — ไปป์ไลน์นี้ไม่มีขั้น hyperparameter tuning |

### GPU ทำอะไรได้บ้าง

| โมเดล | `--accelerator gpu` |
|---|---|
| XGBoost | ✅ `device="cuda"` |
| CatBoost | ✅ `task_type="GPU"` |
| Stacking | ✅ เฉพาะ base learner ที่เป็น XGBoost (RF/LightGBM ยังอยู่ CPU) |
| LightGBM | ❌ wheel จาก PyPI ไม่ได้ build GPU มา |
| RF / MLP / LogReg | ❌ scikit-learn ไม่มี CUDA backend |

**ขั้น preprocessing ไม่ใช้ GPU เลย** — `read_csv`, `to_numeric`, `StandardScaler`
เป็น CPU ล้วนทั้งหมด

ก่อนเริ่มเทรน โค้ดจะ**ทดสอบ CUDA จริงก่อน** (fit ปัญหาเล็ก ๆ บนการ์ด) ถ้าไม่ผ่าน
จะ raise ทันทีแทนที่จะตกไปใช้ CPU เงียบ ๆ แล้วรายงานว่าเป็นเวลาของ GPU

### ทดลองหลายขนาดแบบไล่ทีละขั้น

```bash
python -m src.ids2018.train_ids2018 --sample-size 300000
python -m src.ids2018.train_ids2018 --sample-size 500000
python -m src.ids2018.train_ids2018 --sample-size 1000000
```

รันแยกกันได้เลย ไม่ทับกัน เพราะ path ทุกอย่างถูกแยกตามขนาดอัตโนมัติ:

```
data/ids2018/sample_300k.parquet   results/ids2018/300k/   models/ids2018/300k/
data/ids2018/sample_500k.parquet   results/ids2018/500k/   models/ids2018/500k/
data/ids2018/sample_1m.parquet     results/ids2018/1m/     models/ids2018/1m/
```

**ตัวอย่างถูกออกแบบให้ซ้อนกัน (nested)** — 300k ⊂ 500k ⊂ 1M แถวทุกแถวใน
300k จะอยู่ใน 500k และ 1M ด้วย ตัวแปรเดียวที่เปลี่ยนระหว่างการรันคือ
"ข้อมูลมากขึ้น" ไม่ใช่ "ได้แถวคนละชุด" — ผลเปรียบเทียบจึงตีความได้ตรงไปตรงมา

การเลือกแถวเป็น deterministic ทั้งหมด: seed เดิม → ได้แถวเดิมเป๊ะทุกครั้ง
ไม่ใช่การสุ่มใหม่ทุกรอบ (ดู `tests/test_ids2018_sampling.py` ที่ยืนยันคุณสมบัตินี้ไว้)

นอกจากนี้ **pass 1 ถูก cache ร่วมกัน** ที่ `data/ids2018/corpus_label_index.npz`
การรันครั้งแรกเท่านั้นที่ต้องสแกน CSV 6.7 GB ครั้งที่ 2 และ 3 ข้ามไปเลย
(cache จะถูกยกเลิกอัตโนมัติถ้าไฟล์ CSV ต้นทางเปลี่ยน — ตรวจจาก ชื่อ/ขนาด/mtime)

ถ้าอยากได้พฤติกรรมเดิม (แต่ละขนาดสุ่มแยกอิสระด้วย `StratifiedShuffleSplit`)
ใช้ `--sampling independent`

## ขั้นตอนการทำงาน

### 1. Stratified Sampling 300,000 แถว (สองรอบ)

ข้อมูลดิบ ~13 ล้านแถว / 6.7 GB ใหญ่เกินกว่าจะโหลดเข้า RAM
`data_loader.py` จึงสแกนสองรอบ:

1. **Pass 1** อ่านเฉพาะคอลัมน์ `Label` แบบ chunk → ได้ code `int16` หนึ่งตัวต่อหนึ่งแถว
   (13M แถว ≈ 26 MB) พร้อมตาราง offset ของแต่ละไฟล์ แล้ว cache ลง `.npz`
2. **Plan** เลือกตำแหน่งแถวจาก label vector นั้นให้ครบตามจำนวนที่ขอพอดี —
   **ไม่ใช้ `df.sample()`** ทำให้สัดส่วนทั้ง 15 คลาสเท่ากับของเดิม
3. **Pass 2** อ่านไฟล์ซ้ำด้วย chunk ขนาดเดิม แล้วหยิบเฉพาะแถวที่เลือกไว้

ผลลัพธ์ถูก cache เป็น Parquet ที่ `data/ids2018/sample_<ขนาด>.parquet`
รันครั้งต่อไปจะข้าม pass ทั้งสองทันที (ใช้ `--rebuild-sample` เพื่อบังคับสร้างใหม่)

**โหมดการสุ่ม 2 แบบ** (ทั้งคู่เป็น stratified ไม่มีแบบไหนใช้ simple random sampling):

| โหมด | วิธีทำ | ใช้เมื่อไหร่ |
|---|---|---|
| `nested` (default) | shuffle แต่ละคลาสหนึ่งครั้งด้วย seed คงที่ แล้วตัดเอา prefix ตามโควตาสัดส่วน | เทียบหลายขนาด — ได้ subset ซ้อนกันจริง |
| `independent` | `StratifiedShuffleSplit` ของ scikit-learn | รันขนาดเดียวจบ |

`nested` การันตี subset ได้เพราะโควตาต่อคลาสเป็น **monotone** ตามจำนวนที่ขอ
(เศษจากการปัดถูกโยนให้คลาสใหญ่สุดคลาสเดียว ไม่ใช้ largest-remainder ที่อาจทำให้
โควตาของคลาสหนึ่ง *ลดลง* ตอนยอดรวมโตขึ้น — Alabama paradox — ซึ่งจะทำให้ subset พัง)

**ความถูกต้องของตำแหน่งแถว** คือสิ่งเดียวที่ออกแบบนี้พลาดแบบเงียบ ๆ ได้
pass 2 จึงตรวจ label ของทุกแถวที่หยิบมาเทียบกับที่ pass 1 บันทึกไว้ และ raise
ทันทีถ้าไม่ตรง

### 2. ปัญหาเฉพาะของ CIC-IDS2018 ที่โค้ดจัดการให้

| ปัญหา | วิธีจัดการ |
|---|---|
| `02-20-2018.csv` มี schema ต่างจากไฟล์อื่น (มี `Flow ID`, `Src IP`, `Src Port`, `Dst IP` เพิ่ม 4 คอลัมน์) และใหญ่ 4 GB | อ่านตาม header ของแต่ละไฟล์ แล้วตัดคอลัมน์ identifier ทิ้งใน `split_features_labels()` |
| **ทุกไฟล์** มีบรรทัด header ซ้ำแทรกกลางไฟล์ | ทำเครื่องหมายเป็น code `-1` แต่ **ไม่ลบออกตอนอ่าน** (ลบแล้วตำแหน่งแถวจะเลื่อนไม่ตรงกันระหว่างสอง pass) แล้วไม่เลือกมันมาเป็นตัวอย่าง |
| `Timestamp` | ตัดทิ้ง — แต่ละการโจมตีรันคนละช่วงเวลา timestamp จึงเป็น label proxy (target leakage) |
| `Dst Port` | ตัดทิ้งตาม spec (cardinality สูงและผูกกับการเซ็ตแล็บ เช่น HOIC ยิงพอร์ต 80 เสมอ) — เก็บไว้ได้ด้วย `--keep-dst-port` |
| `Flow Byts/s`, `Flow Pkts/s` มีค่า `Infinity` | flow ที่มีแพ็กเก็ตเดียวมี duration = 0 จึงหารด้วยศูนย์ — แทน ±inf ด้วย NaN แล้ว impute ด้วย median |
| คอลัมน์ที่เป็นศูนย์ทั้งหมด (`Fwd Byts/b Avg`, `Bwd Blk Rate Avg`, ...) | ตรวจ zero-variance **บน train split เท่านั้น** แล้วตัดทิ้ง |
| ค่าที่ parse เป็นตัวเลขไม่ได้ | `pd.to_numeric(errors="coerce")` → NaN → impute ด้วย median ของ train |
| Memory | อ่านแบบ chunk, downcast เป็น `float32`, `gc.collect()` หลังทุก stage |

### 3. คลาสหายากมาก

`SQL Injection` มีแค่ ~87 แถวจาก 13 ล้าน (0.0007%) การสุ่มตามสัดส่วนตรง ๆ
ที่อัตรา 2.3% จะได้แค่ ~2 แถว ซึ่งน้อยเกินกว่าจะแบ่ง 70/30 แบบ stratified ได้
(scikit-learn ต้องการอย่างน้อย 2 ตัวอย่างต่อคลาส)

`--min-per-class` (ค่าเริ่มต้น `2`) จึงเติมแถวให้คลาสที่ขาด แล้วหักจำนวนเท่ากัน
ออกจากคลาสที่ใหญ่ที่สุด เพื่อให้ยอดรวมยังเป็น 300,000 พอดี — กระทบไม่กี่แถว
สัดส่วนคลาสอื่นจึงไม่เปลี่ยน ตั้งเป็น `0` ถ้าต้องการสุ่มตามสัดส่วนล้วน ๆ

### 4. Train / Test Split

`train_test_split(..., stratify=y)` → train 210,000 / test 90,000 แถว

สถิติทุกอย่างที่อาจรั่ว (median สำหรับ impute, ค่า mean/std ของ scaler,
การตัดสินว่าคอลัมน์ไหนคงที่) เรียนจาก **train split เท่านั้น** แล้วนำไปใช้กับ test

### 5. โมเดลทั้ง 7 ตัว

| โมเดล | จัดการ imbalance ด้วย |
|---|---|
| XGBoost | balanced `sample_weight` (XGBoost ไม่มี `class_weight`) |
| LightGBM | `class_weight="balanced"` |
| CatBoost | `auto_class_weights="Balanced"` |
| Random Forest | `class_weight="balanced_subsample"` |
| MLP (128, 64) | — (ใช้ early stopping) |
| Logistic Regression | `class_weight="balanced"` |
| Stacking (RF + LGBM + XGB → LogReg) | balanced `sample_weight` ที่ส่งต่อลง base learners |

Stacking ใช้ base learner ที่เบากว่าเวอร์ชันเดี่ยว เพราะ `StackingClassifier`
ต้อง refit ทุกตัว `cv + 1` ครั้ง

### 6. ผลลัพธ์ที่ได้

```
results/ids2018/<ขนาด>/
├── model_comparison.csv        # ตารางเปรียบเทียบทั้ง 7 โมเดลในก้อนเดียว
├── model_comparison.md         # ฉบับ Markdown สำหรับใส่รายงาน
├── train_ids2018.log
└── <model_name>/
    ├── metrics.json            # accuracy, precision/recall/F1 (macro + weighted), เวลา
    ├── confusion_matrix.csv
    ├── confusion_matrix.png    # heatmap normalise ตามแถว (= recall ของแต่ละคลาส)
    └── per_class_report.csv    # precision/recall/F1/support รายคลาส 15 คลาส

models/ids2018/<ขนาด>/
├── preprocessor.joblib         # ใช้ transform ข้อมูลใหม่ให้เหมือนตอนเทรน
├── label_encoder.joblib
├── metadata.json               # ชื่อ feature, คอลัมน์ที่ตัดทิ้ง, รายชื่อคลาส, seed, โหมดสุ่ม
├── xgboost.joblib + xgboost.json
├── catboost.joblib + catboost.cbm
├── lightgbm.joblib + lightgbm.txt
└── <อื่น ๆ>.joblib
```

> **อ่านตาราง comparison อย่างไร:** ดู `f1_macro` เป็นหลัก ไม่ใช่ `accuracy`
> เพราะ Benign คิดเป็น ~83% ของข้อมูล โมเดลที่จับ SQL Injection ไม่ได้เลย
> ก็ยังได้ accuracy และ weighted F1 ใกล้ 1.00 อยู่ดี — `f1_macro` และ
> `balanced_accuracy` ให้น้ำหนักทุกคลาสเท่ากันจึงเปิดโปงจุดนั้น

## Dependencies

ใช้ตัวเดียวกับ `requirements.txt` ของโปรเจกต์ (`pandas`, `numpy`,
`scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `joblib`, `pyarrow`,
`matplotlib`, `tabulate`) ไม่ต้องติดตั้งอะไรเพิ่ม
