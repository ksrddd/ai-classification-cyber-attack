# สรุปการแก้ False Negative และผลเทรนโมเดลใหม่

วันที่ดำเนินการ: 5–6 กรกฎาคม 2026

## เป้าหมาย

แก้ปัญหา `Infiltration` ถูกทำนายเป็น `BENIGN` มากเกินไป โดยลด False Negative (FN)
ของ `Infiltration` ให้ได้จริงครบทุกโมเดล และแสดง Recall, F2, FPR, FN และ FP บน Dashboard
แทนการตัดสินจาก Accuracy หรือ Macro F1 เพียงอย่างเดียวs

Recall ของ `Infiltration` คำนวณจาก `TP / (TP + FN)` ดังนั้น Recall สูงขึ้นหมายถึง
โมเดลตรวจจับ `Infiltration` ที่มีอยู่จริงได้มากขึ้น และ FN ลดลง

## สิ่งที่แก้ใน Training Pipeline

1. แยกข้อมูลเป็น train 240,000 แถว, calibration 60,000 แถว และ test 60,000 แถว
2. calibration และ test คงสัดส่วนธรรมชาติของข้อมูลและไม่ซ้อนทับกับ train
3. ใช้ข้อมูล `Infiltration` จริงประมาณ 104,447 แถวใน targeted training แทนการสร้างข้อมูล
   synthetic จำนวนมาก ทั้งที่ corpus มีข้อมูลจริงเพียงพอ
4. ใช้ target-F2 เป็น metric สำหรับหา hyperparameters เพื่อให้น้ำหนัก Recall มากกว่า Precision
5. เพิ่ม threshold calibration สำหรับ `Infiltration` แยกแต่ละโมเดล โดยเลือก threshold ที่ทำให้
   target F1 สูงสุดภายใต้ FPR ceiling จาก calibration set เท่านั้น แล้วประเมินครั้งสุดท้ายบน test set
   ที่ไม่เคยใช้เลือก threshold
6. บันทึก threshold ไว้ใน model artifact ทำให้การเรียก `predict()` ใน Dashboard ใช้กฎเดียวกับ
   ตอนประเมินผล
7. เพิ่ม Recall, F2, FPR, FP, FN และจำนวน FN ที่ไหลไป `BENIGN` ใน metrics/report/Dashboard
8. รองรับ threshold/FPR ceiling รายโมเดล เพราะ score distribution ของแต่ละ algorithm ไม่เหมือนกัน
9. LightGBM ใช้ class weight เพราะ targeted 1:1 ทำให้ probability จับตัวเป็นก้อนและเกิด FPR สูงผิดปกติ
10. เพิ่ม `--reuse-best-params` เพื่อ recovery เฉพาะโมเดลโดยไม่ต้องค้นหา hyperparameters ใหม่

## ผลสุดท้ายบน Natural Test Set

Test set มี 60,000 แถว และมี `Infiltration` 599 แถวเท่ากันทุกโมเดล

| Model | Threshold | FP แบบเน้น Recall | FN แบบเน้น Recall | FP แบบสมดุล | FN แบบสมดุล | FP+FN ลดลง |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.959728 | 1,262 | 318 | 290 | 424 | 866 |
| XGBoost | 0.905566 | 1,347 | 331 | 337 | 444 | 897 |
| LightGBM | 0.893617 | 1,858 | 324 | 311 | 428 | 1,443 |
| CatBoost | 0.918689 | 1,299 | 311 | 454 | 384 | 772 |
| MLP | 0.963101 | 1,852 | 369 | 355 | 431 | 1,435 |
| Logistic Regression | 0.738480 | 3,764 | 458 | 1,413 | 502 | 2,307 |
| Stacking | 0.948477 | 1,242 | 310 | 489 | 384 | 679 |

Balanced-F1 calibration ลด FP รวมจาก 12,624 เหลือ 3,649 และลดความผิดพลาดข้ามสองคลาส
`FP + FN` จาก 15,045 เหลือ 6,646 หรือลดลง 8,399 รายการ (55.83%) พร้อมเพิ่ม Accuracy และ
Macro-F1 ของทุกโมเดลเมื่อเทียบกับ profile ที่เน้น Recall

ไม่สามารถทำให้ทั้ง `BENIGN → Infiltration` และ `Infiltration → BENIGN` เป็นศูนย์ด้วย threshold
เพียงค่าเดียวได้ เนื่องจาก feature distribution ของสองคลาสมีส่วนทับซ้อนกัน การเพิ่ม threshold ลด FP
แต่เพิ่ม FN เสมอ จึงใช้ target F1 เป็นจุดทำงานเริ่มต้นที่สมดุลกว่า และเก็บ threshold แบบเน้น Recall
ไว้ใน artifact สำหรับการตรวจสอบย้อนหลัง

## การตรวจสอบก่อนนำขึ้น Dashboard

- Automated tests ผ่าน 121 tests
- Ruff lint, Python compile และ PowerShell parser ผ่าน
- โหลด model artifact และเรียก `predict()`/`predict_proba()` สำเร็จครบ 7 โมเดล
- probability เป็น finite และผลรวมแต่ละแถวเท่ากับ 1
- threshold ของทุก model เป็น finite และถูก serialize อยู่ใน artifact
- Dashboard home, Model Comparison และ Model Performance render โดยไม่มี exception
- Model Comparison อ่านโมเดลครบ 7 ตัวและจัด Stacking เป็นโมเดลที่ FN ต่ำสุด
- Model Performance แสดง Threshold, Recall, F2, FPR, FN และ FN ไป `BENIGN`

## ไฟล์ผลลัพธ์ที่ใช้งานจริง

ผลที่ผ่านการตรวจถูกโปรโมตแบบ atomic ไปที่ `results/latest/` ประกอบด้วย model artifact,
metrics, per-class report, confusion matrix, `before_after_fn.csv`, `final_manifest.json`
และ `report.md` ส่วนผล smoke test และ source run ที่ซ้ำกับ `latest` ถูกลบหลังตรวจสอบสำเร็จ

Dashboard เปิดใช้งานที่ `http://localhost:8501`
