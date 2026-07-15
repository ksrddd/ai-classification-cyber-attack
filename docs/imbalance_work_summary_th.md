# สรุปงานแก้ปัญหา Class Imbalance และแผนเทรนใหม่ทุกโมเดล

วันที่จัดทำ: 2 กรกฎาคม 2026  
แผนเริ่มเทรนจริง: เที่ยงคืนเข้าสู่วันที่ 3 กรกฎาคม 2026
เทรนและตรวจรับผลเสร็จ: 3 กรกฎาคม 2026

## 1. ปัญหาที่พบ

ปัญหาหลักคือโมเดลมักทำนายคลาส `Infiltration` เป็น `BENIGN` แม้ accuracy
รวมจะสูงมาก สาเหตุที่ accuracy ยังดูดีเป็นเพราะ `BENIGN` มีจำนวนมากกว่าคลาส
โจมตีอย่างชัดเจน ดังนั้น accuracy เพียงตัวเดียวไม่สามารถบอกคุณภาพของโมเดลใน
คลาส `Infiltration` ได้

ข้อมูล cleaned cache ปัจจุบันมีประมาณ 13.94 ล้านแถว โดยมีจำนวนสำคัญดังนี้:

| Class | จำนวนแถว |
|---|---:|
| BENIGN | 12,167,406 |
| Infiltration | 139,377 |
| DDoS | 903,617 |
| DoS | 390,088 |
| Bot | 146,483 |
| Brute Force | 103,244 |
| PortScan | 90,694 |
| Web Attack | 3,002 |
| Heartbleed | 11 |

ใน pipeline เดิม เมื่อใช้ budget 300,000 แถว จะเหลือ Infiltration ใน train
ประมาณ 2,400 แถว ทั้งที่มีข้อมูลจริงมากกว่า 139,000 แถว จึงไม่ควรเริ่มจากการ
สร้างข้อมูลสังเคราะห์จำนวนมากโดยยังทิ้งข้อมูลจริงส่วนใหญ่ไป

## 2. สิ่งผิดปกติที่ตรวจพบเพิ่มเติม

### 2.1 ข้อมูล CICIDS2017 และ CSE-CIC-IDS2018 มีคอลัมน์ไม่ครบเหมือนกัน

หลังรวมข้อมูลพบ NaN ในคอลัมน์ source-specific เช่น `Protocol` และ packet-rate
aliases บางตัว Tree models บางชนิดรับ NaN ได้ แต่ SMOTE ซึ่งใช้ nearest
neighbors รับไม่ได้ จึงเพิ่ม `SimpleImputer(strategy="median")` ไว้ใน pipeline
ก่อน scaler/sampler เพื่อให้ทุกโมเดลเห็น preprocessing ชุดเดียวกัน

## 3. การแก้ลำดับ Data Split เพื่อป้องกัน Leakage

ลำดับใหม่เป็นดังนี้:

1. โหลดและ clean corpus
2. เลือก test holdout ตามสัดส่วนธรรมชาติของข้อมูลก่อน
3. ห้ามแก้สมดุล test set
4. ทำ targeted sampling หรือ over-sampling เฉพาะ train
5. ถ้ามี cross-validation ให้ sampler ทำงานเฉพาะ train fold ของแต่ละ fold
6. ประเมินผลบน natural-distribution test set

จุดสำคัญคือ SMOTE จะไม่เห็นข้อมูล validation/test ล่วงหน้า จึงไม่เกิด leakage
จากการสร้าง synthetic sample ก่อน split

## 4. Imbalance strategies ที่เพิ่มเข้ามา

ทุกโมเดลสามารถเลือก strategy เดียวกันผ่าน CLI ได้:

| Strategy | วิธีทำงาน |
|---|---|
| `class_weight` | ใช้น้ำหนักคลาสหรือ sample weight ของโมเดล |
| `targeted` | เก็บ Infiltration จริงเพิ่ม และลด quota ของ majority โดยไม่สร้างข้อมูลปลอม |
| `random_over` | ทำซ้ำแถว Infiltration จริงภายใน train fold |
| `borderline_smote` | สร้าง synthetic Infiltration เฉพาะบริเวณ decision boundary |
| `smoteenn` | SMOTE แล้วใช้ ENN ช่วยล้างบริเวณที่ noisy/overlap |

`target_ratio=0.20` หมายถึงหลัง resampling ต้องการให้จำนวน target class เท่ากับ
ประมาณ 20% ของ majority class ไม่ได้ทำให้ทุกคลาสสมดุลแบบ 1:1

เมื่อใช้ `targeted`, `random_over`, `borderline_smote` หรือ `smoteenn` ระบบจะปิด
class weight ของ classifier เพื่อป้องกันการชดเชย imbalance ซ้ำสองชั้น

## 5. การรองรับทุกโมเดล

ก่อนแก้ canonical `train.py` รองรับจริงเพียง Random Forest, XGBoost และ
LightGBM แม้ CLI จะแสดงโมเดลอื่นด้วย ตอนนี้รองรับครบ 7 โมเดลแล้ว:

| Model | การจัดการเมื่อใช้ `class_weight` |
|---|---|
| Random Forest | `class_weight="balanced_subsample"` |
| XGBoost | คำนวณ balanced `sample_weight` ภายในทุก fit/fold |
| LightGBM | `class_weight="balanced"` |
| CatBoost | `auto_class_weights="Balanced"` |
| MLP | balanced `sample_weight` ผ่าน `BalancedMLPClassifier` |
| Logistic Regression | `class_weight="balanced"` |
| Stacking | ถ่วงน้ำหนักทั้ง LGBM/XGB/RF base learners และ Logistic meta learner |

สำหรับ Stacking ถ้าใช้ sampler ระบบจะปิดน้ำหนักทั้ง base learners และ meta learner
ทั้งหมด เพื่อไม่ให้เกิด double correction

## 6. การปรับปรุงด้านความถูกต้องของ Artifact

- เพิ่ม imbalance protocol version ใน metrics
- ไม่ reuse โมเดลเก่าที่สร้างจาก split/preprocessing protocol คนละรุ่น
- model artifact จะถูก reuse เฉพาะเมื่อ strategy, target class และ target ratio
  ตรงกับ run ปัจจุบัน
- CatBoost prediction ถูก normalize จาก shape `(n, 1)` เป็น `(n,)`
- แก้ feature-name drift ของ LightGBM และ Stacking
- เพิ่ม run name แยก experiment เพื่อไม่ให้ผลใหม่ทับ baseline โดยไม่ตั้งใจ

## 7. ผลการทดสอบที่ทำแล้ว

### 7.1 Automated tests

- Tests ทั้งโปรเจกต์ผ่าน 116 tests
- Imbalance tests ผ่าน 54 tests โดยเปิด warnings-as-errors
- Ruff lint ผ่าน
- Python compile check ผ่าน
- PowerShell syntax ของสคริปต์เทรนคืนนี้ผ่าน

Tests ครอบคลุม:

- ทุกโมเดล build ได้กับทั้ง 5 strategies
- ทุกโมเดล fit ได้ทั้ง class-weight และ resampling
- sampler อยู่ภายใน pipeline
- test set ไม่เปลี่ยนเมื่อเปลี่ยน train strategy
- train/test ไม่มี row overlap
- CLI aliases และ `--models all` ทำงานถูกต้อง

### 7.2 Smoke test ทุกโมเดล

รันครบทั้ง 7 โมเดลสองรอบ ได้แก่ `class_weight` และ `borderline_smote`
โดยใช้ holdout เดียวกันและเปิด warnings-as-errors ผลเบื้องต้นของ Infiltration:

| Model | Class-weight F1 | BorderlineSMOTE F1 | Candidate จาก smoke |
|---|---:|---:|---|
| Random Forest | 0.4000 | 0.3000 | Class weight |
| XGBoost | 0.2791 | 0.3243 | BorderlineSMOTE |
| LightGBM | 0.2703 | 0.2927 | BorderlineSMOTE |
| CatBoost | 0.1468 | 0.2791 | BorderlineSMOTE |
| MLP | 0.0509 | 0.1714 | BorderlineSMOTE |
| Logistic Regression | 0.0423 | 0.0270 | Class weight |
| Stacking | 0.1159 | 0.3529 | BorderlineSMOTE |

ข้อควรระวัง: smoke test มี Infiltration ใน test เพียง 19 แถว จึงใช้เพื่อยืนยัน
ว่าโค้ดทำงานและเลือก candidate เท่านั้น ห้ามใช้เป็นผลวิจัยสุดท้าย

### 7.3 XGBoost บนชุด 300k

ทดสอบ XGBoost เพิ่มบน natural holdout 60,000 แถว ซึ่งมี Infiltration 599 แถว:

| Strategy | Precision | Recall | Infiltration F1 | Accuracy | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Class weight | 0.0953 | 0.5292 | 0.1614 | 0.9443 | 0.8422 |
| BorderlineSMOTE 0.20 | 0.2470 | 0.3422 | 0.2869 | 0.9823 | 0.8635 |

Class weight จับ Infiltration ได้มากกว่า แต่ false positives สูงมาก ส่วน
BorderlineSMOTE ให้ precision, Infiltration F1, accuracy และ macro F1 สมดุลกว่า

## 8. แผนเทรนใหม่ทุกโมเดลคืนนี้

สคริปต์ที่เตรียมไว้คือ:

```text
train_all_imbalance.ps1
```

สคริปต์จะเทรนตามลำดับ Logistic Regression, Random Forest, CatBoost, XGBoost,
LightGBM, MLP และ Stacking โดยทำทีละตัวเพื่อลดการแย่ง RAM/CPU และตั้ง Windows
process priority เป็น `BelowNormal`

### 8.1 ก่อนเที่ยงคืน

เปิด PowerShell แล้วรัน:

```powershell
Set-Location "D:\Projects\Senior Project\ai-classification-cyber-attack"
.\.venv\Scripts\python.exe -m pytest tests\test_train_imbalance.py -q --no-cov
```

ควรเห็น `54 passed` จากนั้นตรวจว่า:

- ต่อสายชาร์จ
- ปิด sleep/hibernate ชั่วคราว
- มีพื้นที่ disk ว่างเพียงพอ
- ปิดโปรแกรมที่ใช้ RAM/CPU สูง
- ไม่ใช้ `--refresh-cache` ถ้าไม่มีการเปลี่ยน raw CSV

### 8.2 คำสั่งแนะนำเวลา 00:00

รันทุกโมเดลด้วย BorderlineSMOTE, ratio 0.20 และ preset 8 GB:

```powershell
.\train_all_imbalance.ps1 `
  -RunName "all_models_20260703_borderline" `
  -Preset 8gb `
  -Strategy borderline_smote `
  -TargetRatio 0.20
```

คำสั่งนี้เปิด hyperparameter search, full-train CV และ label-shuffle sanity
check จึงอาจใช้เวลาหลายชั่วโมง โดยเฉพาะ CatBoost, MLP และ Stacking

ถ้าต้องการเทรน baseline ให้ครบก่อนแบบเร็วกว่า:

```powershell
.\train_all_imbalance.ps1 `
  -RunName "all_models_20260703_borderline_fast" `
  -Preset 8gb `
  -Strategy borderline_smote `
  -TargetRatio 0.20 `
  -SkipTuning -SkipCV -SkipLabelShuffle
```

### 8.3 ถ้าเครื่องหรือ PowerShell หยุดกลางทาง

รันคำสั่งเดิมซ้ำโดยใช้ `RunName` เดิม ระบบจะข้ามโมเดลที่เสร็จและ strategy
ตรงกัน แล้วทำต่อจากตัวที่ยังไม่มี artifact

อย่าใส่ `-Force` ตอน resume เพราะ `-Force` จะสั่งเทรนโมเดลที่เสร็จแล้วใหม่

### 8.4 การติดตามผล

ระหว่างทำงานดู progress ใน PowerShell ได้โดยตรง และดูสรุปสถานะล่าสุดที่:

```text
results/all_models_20260703_borderline/training_session_summary.csv
```

ผลของแต่ละโมเดลจะอยู่ในโฟลเดอร์เดียวกัน:

```text
results/latest/
  <model>.joblib
  <model>_metrics.json
  <model>_per_class.csv
  <model>_confusion_matrix.png
  metrics.json
  report.md
```

## 9. Metric ที่ต้องใช้ตัดสินผลหลังเทรน

อย่าเลือกโมเดลจาก accuracy อย่างเดียว ให้ตรวจอย่างน้อย:

1. Infiltration precision
2. Infiltration recall
3. Infiltration F1 หรือ F2
4. จำนวน Infiltration ที่ถูกทำนายเป็น BENIGN ใน confusion matrix
5. Macro F1
6. Balanced accuracy
7. จำนวน BENIGN ที่ถูกทำนายผิดเป็น Infiltration
8. CV mean/std และ shuffled-label sanity result

ถ้าต้องการลดกรณี Infiltration หลุดเป็น BENIGN ให้เน้น recall/F2 แต่ต้องรายงาน
false positives ควบคู่กันเสมอ เพราะการเพิ่ม recall อย่างเดียวอาจทำให้ระบบแจ้ง
เตือนผิดจำนวนมาก

## 10. ไฟล์สำคัญที่เพิ่มหรือแก้

- `train.py` — split/resampling/model support/metrics protocol
- `main.py` — CLI imbalance arguments และ model forwarding
- `train_all_imbalance.ps1` — สคริปต์เทรนทุกโมเดลคืนนี้
- `tests/test_train_imbalance.py` — regression tests ทุก model/strategy
- `docs/imbalance_experiments.md` — วิธีใช้แต่ละ strategy
- `scripts/promote_best_models.py` — รวมโมเดลที่ผ่านการคัดเลือกแบบ atomic
- `results/latest/metrics.json` — ผลรวมของโมเดลที่ใช้งานจริงทั้ง 7 ตัว
- `results/latest/report.md` — ตารางสรุปผลสำหรับ Dashboard

## 11. ผลเทรนสุดท้ายที่นำขึ้น Dashboard

ชุดทดสอบเป็น natural holdout 60,000 แถว และไม่มีการทำ oversampling บน test set
โมเดล 6 ตัวใช้ target-only BorderlineSMOTE ที่อัตรา Infiltration ต่อ majority 0.20
ส่วน LightGBM ใช้ class weight เพราะผล BorderlineSMOTE ของ LightGBM เกิด model
collapse จึงไม่นำผลรอบนั้นมาใช้งาน

| Model | Strategy | Accuracy | Balanced Acc. | Macro F1 | Infiltration Precision | Infiltration Recall | Infiltration F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Random Forest | BorderlineSMOTE | 0.9833 | 0.9113 | 0.8733 | 0.2615 | 0.3222 | 0.2887 |
| CatBoost | BorderlineSMOTE | 0.9792 | 0.8880 | 0.8491 | 0.2131 | 0.3689 | 0.2702 |
| XGBoost | BorderlineSMOTE | 0.9806 | 0.8789 | 0.8435 | 0.2233 | 0.3556 | 0.2743 |
| LightGBM | Class weight | 0.9780 | 0.8712 | 0.8172 | 0.1727 | 0.2888 | 0.2161 |
| MLP | BorderlineSMOTE | 0.9665 | 0.9047 | 0.8132 | 0.1429 | 0.3689 | 0.2060 |
| Stacking | BorderlineSMOTE | 0.9843 | 0.7922 | 0.7593 | 0.2769 | 0.3139 | 0.2942 |
| Logistic Regression | BorderlineSMOTE | 0.9605 | 0.7971 | 0.7474 | 0.1580 | 0.2070 | 0.1792 |

Random Forest เป็นโมเดลที่สมดุลที่สุดเมื่อจัดอันดับด้วย Macro F1 ส่วน Stacking
มี Infiltration F1 สูงสุด และ CatBoost/MLP มี Infiltration recall สูงสุด
Dashboard อ่านเฉพาะโมเดลสุดท้ายจาก `results/latest/` และแสดงตัวเลือกครบ 7 โมเดล
ผล smoke, ผลทดลองระหว่างทาง และ SHAP ของโมเดลเก่าถูกลบหลังตรวจ Dashboard สำเร็จ
