# Dashboard Guide — Next.js + FastAPI

คู่มือเปิด dashboard เพื่อดูผลโมเดล AI ที่เทรนเสร็จแล้ว ใช้ Next.js (frontend) + FastAPI (backend) สื่อสารผ่าน REST API

---

## 1. โครงสร้าง

```
+---------------------+         +------------------------+
|  Browser            |  HTTP   |  Next.js dev server    |
|  localhost:3000     | ------> |  npm run dev           |
+---------------------+         |  - React 18            |
                                |  - Tailwind            |
                                |  - Recharts            |
                                +-----------+------------+
                                            | fetch
                                            v
                                +------------------------+
                                |  FastAPI backend       |
                                |  localhost:8000        |
                                |  uvicorn api.main:app  |
                                +-----------+------------+
                                            | read
                                            v
                                +------------------------+
                                |  models/*.joblib       |
                                |  results/metrics/*.json|
                                |  data/processed/*.parq |
                                +------------------------+
```

ต้องเปิดทั้งคู่ — frontend อ่าน API ของ backend

---

## 2. เตรียมก่อนเปิด dashboard

ต้องทำหลัง train เสร็จแล้ว (ดู `training_guide.pdf`)

### 2.1 ก็อปโมเดลที่เทรนเสร็จไปยัง canonical path

Dashboard อ่าน `models/<name>.joblib` ไม่ใช่ `results/latest/`

```powershell
Copy-Item results\latest\random_forest.joblib models\
Copy-Item results\latest\xgboost.joblib models\
Copy-Item results\latest\lightgbm.joblib models\
Copy-Item results\latest\label_encoder.joblib models\
```

ตรวจ:

```powershell
ls models\*.joblib
```

ควรเห็น 4 ไฟล์ขึ้นไป

### 2.2 ติดตั้ง Node.js (ครั้งเดียวต่อเครื่อง)

Dashboard ใช้ Next.js ต้องการ Node.js 18+

ดาวน์โหลดที่ https://nodejs.org → เลือก LTS (เลข 20.x ขึ้นไป)

ตรวจ:

```powershell
node --version
npm --version
```

ต้องได้ทั้งคู่

### 2.3 ติดตั้ง npm packages ของ Next.js (ครั้งเดียวต่อ clone)

```powershell
cd web
npm install
cd ..
```

ขั้นนี้ ~3-5 นาที (ดาวน์โหลด ~300 MB ลง node_modules/)

---

## 3. เปิด dashboard — ต้องเปิด 2 terminal

### 3.1 Terminal A — FastAPI backend

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --reload --port 8000
```

รอจน console เห็น:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

ทดสอบ API ได้ที่ http://localhost:8000/docs (FastAPI auto-doc — Swagger UI)

**ห้ามปิด terminal นี้** ตลอดที่ใช้ dashboard

### 3.2 Terminal B — Next.js frontend

```powershell
cd web
npm run dev
```

รอจน console เห็น:

```
   ▲ Next.js 14.2.35
   - Local:        http://localhost:3000
 ✓ Ready in X.Xs
```

เปิด browser → http://localhost:3000

---

## 4. หน้าใน dashboard

| URL | หน้า | แสดง |
|---|---|---|
| / | Home | ภาพรวม dataset + จำนวน class + n_features |
| /dataset | Dataset | ตาราง label distribution + missing/duplicate counts |
| /eda | EDA | กราฟ class distribution + feature stats |
| /compare | Compare | ตารางเทียบ accuracy/f1/precision/recall ของทุก model |
| /performance | Performance | per-model confusion matrix + per-class metrics |
| /shap | SHAP | top features ที่ model ใช้ตัดสิน (ถ้ามี SHAP analysis แล้ว) |
| /predict | Predict | upload CSV ใหม่ ดู prediction |

---

## 5. ทดสอบ end-to-end

### 5.1 ทดสอบ backend ตอบ

ก่อนเปิด frontend ลอง:

```powershell
curl http://localhost:8000/api/models
```

ควรได้ JSON ที่มีชื่อโมเดลที่เทรน เช่น:

```json
{"models": ["random_forest", "xgboost", "lightgbm"]}
```

ถ้าได้แล้ว frontend ใช้งานได้

### 5.2 ทดสอบ Predict (upload CSV ใหม่)

1. เข้าหน้า /predict
2. เลือกโมเดล (เช่น xgboost)
3. กด upload — เลือก CSV ที่มี 77 features เหมือนตอนเทรน (ใช้ scripts/generate_sample.py สร้างได้)
4. กด predict
5. ดูผลในตาราง preview

---

## 6. หยุด dashboard

ทั้ง 2 terminal: **Ctrl+C** หยุดได้เลย

---

## 7. Build production (optional)

ถ้าจะ deploy หรือทำ portfolio:

```powershell
cd web
npm run build
npm run start
```

`npm run start` จะ serve ที่ port 3000 เหมือนเดิม แต่เป็น production build — เร็วกว่า dev mode ~3-5 เท่า ใช้สำหรับ demo อาจารย์

backend ก็เปลี่ยนเป็น production (ตัด `--reload`, ใส่ `--workers` สำหรับ concurrent requests):

```powershell
uvicorn api.main:app --port 8000 --workers 2
```

---

## 8. Troubleshooting

| อาการ | สาเหตุ | แก้ |
|---|---|---|
| API 404 /api/models | ลืมก็อป joblib ไป models/ | ดูข้อ 2.1 |
| API 500 ImportError | ยังไม่ activate venv | .\\.venv\\Scripts\\Activate.ps1 |
| frontend แสดง "API ... -> ECONNREFUSED" | backend ยังไม่เปิด | เปิด terminal A ก่อน |
| frontend แสดง "API ... -> 404" | path ผิด หรือ endpoint ไม่มีข้อมูล | ลอง curl เช็คตามข้อ 5.1 |
| npm install เด้ง permission error | run as admin หรือใช้ Yarn แทน | npm install --force |
| dashboard ขึ้น "model not found" หลัง retrain | ลืมก็อป joblib ใหม่ทับ | ทำซ้ำข้อ 2.1 |
| port 3000 ใช้แล้ว | มีอย่างอื่นเปิดอยู่ | npm run dev -- -p 3001 |
| port 8000 ใช้แล้ว | มีอย่างอื่นเปิดอยู่ | uvicorn ... --port 8001 + ตั้ง NEXT_PUBLIC_API_URL=http://localhost:8001 |

---

## 9. คำสั่งสรุป

**ครั้งเดียวต่อเครื่อง** — ติดตั้ง Node.js 20+ จาก nodejs.org ก่อน แล้ว:

```powershell
cd web
npm install
cd ..
```

**ทุกครั้งที่จะเปิด dashboard** — Terminal A (backend):

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --reload --port 8000
```

Terminal B (frontend, เปิด terminal ใหม่):

```powershell
cd web
npm run dev
```

แล้วเปิด browser ไปที่ `http://localhost:3000`

---

## 10. หลัง retrain โมเดลใหม่

1. รัน train.py (ดู `training_guide.pdf`)
2. ก็อปโมเดล: `Copy-Item results\latest\*.joblib models\`
3. backend จะ pick up ใน 60 วินาที (cache revalidation), หรือ Ctrl+C + restart uvicorn ให้อัปเดตทันที
4. refresh browser
