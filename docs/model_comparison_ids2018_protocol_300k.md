# Split Protocol and Hyperparameter Tuning on CSE-CIC-IDS2018 (300k)

## ผลของโปรโตคอลการแบ่งข้อมูลและการปรับไฮเปอร์พารามิเตอร์ บน CSE-CIC-IDS2018 (300k)

**Bilingual document / เอกสารสองภาษา** — Part I is English, Part II is Thai (ภาษาไทย). Every number comes from run bundles in this repository: `results/ids2018/300k/`, `results/ids2018/300k_temporal/` and `results/ids2018/300k_temporal_tuned/`. ตัวเลขทั้งหมดมาจากผลการรันจริงในโปรเจกต์นี้

### Scope / ขอบเขต

Three runs, one corpus, one sample. The 300,000-row stratified sample, the preprocessing, the seed and the seven model architectures are identical across all three. Only two things change: **how the train/test split is drawn**, and **whether hyperparameters were searched**. This isolates the two protocol choices from everything else.

สามการรัน คลังข้อมูลเดียว ตัวอย่างชุดเดียว — ตัวอย่าง 300,000 แถว การเตรียมข้อมูล ค่า seed และสถาปัตยกรรมโมเดลทั้ง 7 ตัว เหมือนกันทุกประการ เปลี่ยนแค่ 2 อย่าง: **วิธีแบ่ง train/test** และ **ค้นหาไฮเปอร์พารามิเตอร์หรือไม่**

---

# PART I — ENGLISH

## 1. Executive Summary

The experiment was designed to answer one question: does a chronological split, or hyperparameter tuning, materially change what these seven models can do on CSE-CIC-IDS2018?

**The answer is that the question cannot be answered at this sample size, and the reason is more useful than the answer would have been.**

Macro-F1 moved by up to +0.099 between protocols. The 95% bootstrap confidence interval on macro-F1 is on average **0.128 wide**. Every effect measured is smaller than the interval around it. Not one pair of protocols is separable.

The cause is structural, not statistical noise that more bootstrap samples would resolve. Macro-F1 averages fifteen classes with equal weight. Three of those classes have **1, 2 and 4 test flows**. A single flow changing its predicted class moves macro-F1 by 1/15 = **0.067** — larger than any protocol effect in the table below.

### The 2×2, filled in

| Split | Untuned | Tuned |
|---|---|---|
| Random stratified 70/30 | `300k` ✓ | not run |
| Chronological (per capture-day, per class) | `300k_temporal` ✓ | `300k_temporal_tuned` ✓ |

The missing cell is the one that would produce the highest headline number and the least trustworthy one; it was deliberately not run.

### Effect decomposition, macro-F1

| Model | Random+untuned | Temporal+untuned | Temporal+tuned | Split effect | Tuning effect |
|---|---|---|---|---|---|
| CatBoost | 0.6763 | 0.7277 | 0.7751 | +0.0514 | +0.0474 |
| Random Forest | 0.6869 | 0.6833 | 0.7290 | −0.0036 | +0.0457 |
| XGBoost | 0.6854 | 0.7037 | 0.7021 | +0.0183 | −0.0016 |
| LightGBM | 0.6870 | 0.7026 | 0.6819 | +0.0156 | −0.0208 |
| Logistic Regression | 0.6296 | 0.6725 | 0.6725 | +0.0429 | 0.0000 |
| Stacking | 0.6886 | 0.6603 | 0.6580 | −0.0283 | −0.0023 |
| MLP | 0.6698 | 0.7168 | 0.6003 | +0.0470 | **−0.1165** |

Every value in the two right-hand columns is inside the confidence interval of the numbers it was computed from. The signs are not stable across models: tuning helped two models, hurt three, and did nothing for two.

### Three findings that survive the noise

**1. Accuracy is uninformative here, and was before this experiment.** All twenty-one model-protocol combinations sit between 0.9651 and 0.9839. Benign is 74,769 of 90,011 test flows, so the majority-class baseline is 0.8307. Accuracy is the metric that changes least between protocols precisely because it is dominated by the class no protocol affects.

**2. The Stacking ensemble becomes extremely conservative under a chronological split.** Its false-alarm rate falls from 0.135% (101 false positives) under the random split to **0.021%** (16 false positives) under the chronological one — a sixfold reduction, the largest single change in the entire study. It pays for this with slightly more missed attacks (834 → 944). This is the one result with a plausible operational reading, and it is visible in the false-alarm column rather than in macro-F1.

**3. Tuning on a subsample can actively harm the final model.** MLP lost 0.1165 macro-F1 after tuning. The search selects on a 60,001-row subsample and the winner is then refit on all 209,989 rows; a configuration that is optimal at 60k need not be optimal at 210k, and for a neural network with early stopping it plainly was not. The tree ensembles were far less sensitive to this.

## 2. What each run is

All three share: 300,000-row stratified sample from the ~13M-row corpus, `MIN_PER_CLASS=2`, seed 42, 15 classes, 69 features after dropping identifier, timestamp and constant columns, `class_weighting = none`, CPU only.

### 2.1 `300k` — random stratified

A stratified 70/30 row split, 210,000 / 90,000. This is the protocol most published CICIDS results use. Its weakness is stated on the dashboard itself: flows produced by a single attack burst can land on both sides of the split, so the model may be scored on traffic nearly identical to traffic it trained on. Every score under this protocol should be read as an upper bound.

### 2.2 `300k_temporal` — chronological, untuned

For every `(capture day, class)` group, the earliest 70% of flows by wall-clock `Timestamp` train the model and the latest 30% test it. Result: 209,989 / 90,011. Groups of one row go entirely to train.

Grouping by day as well as class matters for the two classes that ran on more than one day (`Infilteration`, `Brute Force -Web`): a single global cutoff per class would send one whole day to test.

`Timestamp` is used **only** to order rows. It is dropped before the preprocessor sees it, and an assertion enforces that — the column is a near-perfect label proxy because each attack ran inside its own capture window, and leaking it would inflate every score in the bundle. The feature count is 69 both here and under the random split, which is the observable confirmation that nothing leaked.

**Honest limit of this protocol.** Twelve of the fifteen classes occur on exactly one capture day. For those, the split orders flows *within a single attack burst* rather than across independent days. It removes the random split's optimism — no test flow precedes the training flows of its own class — but it does not demonstrate generalisation to a separately-staged future campaign. That would require a capture-holdout or cross-dataset protocol, neither of which this corpus supports at class granularity.

### 2.3 `300k_temporal_tuned` — chronological, tuned

Same split. Each of six models received a randomised search: 20 candidates × 3-fold cross-validation, scored on macro-F1, run on a 60,001-row stratified subsample of the training split. The winning configuration was then refit on the full 209,989 training rows.

Stacking was not searched. `StackingClassifier` refits three base learners `cv+1` times per candidate fit, so a 20×3 search would cost roughly 720 base-learner fits for the two hyperparameters it actually exposes. It instead inherits the tuned parameters of its base learners.

## 3. Results

### 3.1 `300k` — random stratified, untuned

| Model | Accuracy | F1-macro | 95% CI | MCC | False-alarm rate | Missed (FN) |
|---|---|---|---|---|---|---|
| Stacking | 0.9839 | 0.6886 | [0.6759, 0.8035] | 0.9465 | 0.135% | 834 |
| LightGBM | 0.9837 | 0.6870 | [0.6747, 0.8020] | 0.9460 | 0.139% | 842 |
| Random Forest | 0.9828 | 0.6869 | [0.6762, 0.8004] | 0.9431 | 0.199% | 874 |
| XGBoost | 0.9837 | 0.6854 | [0.6728, 0.8003] | 0.9460 | 0.118% | 860 |
| CatBoost | 0.9836 | 0.6763 | [0.6668, 0.7877] | 0.9454 | **0.062%** | 914 |
| MLP | 0.9816 | 0.6698 | [0.6594, 0.7808] | 0.9389 | 0.128% | 1,004 |
| Logistic Regression | 0.9651 | 0.6296 | [0.6168, 0.7339] | 0.8823 | 0.815% | 1,912 |

Leader is Stacking; 3 of 7 rivals are not separable from it at p > 0.05.

### 3.2 `300k_temporal` — chronological, untuned

| Model | Accuracy | F1-macro | 95% CI | MCC | False-alarm rate | Missed (FN) |
|---|---|---|---|---|---|---|
| CatBoost | 0.9833 | 0.7277 | [0.6803, 0.8054] | 0.9444 | 0.131% | 920 |
| MLP | 0.9814 | 0.7168 | [0.6688, 0.7944] | 0.9380 | 0.163% | 1,038 |
| XGBoost | 0.9835 | 0.7037 | [0.6682, 0.8102] | 0.9451 | 0.166% | 867 |
| LightGBM | 0.9832 | 0.7026 | [0.6682, 0.8056] | 0.9441 | 0.159% | 880 |
| Random Forest | 0.9827 | 0.6833 | [0.6767, 0.7899] | 0.9424 | 0.205% | 917 |
| Logistic Regression | 0.9676 | 0.6725 | [0.6185, 0.7507] | 0.8905 | 0.544% | 1,925 |
| Stacking | 0.9837 | 0.6603 | [0.6421, 0.7652] | 0.9458 | **0.021%** | 944 |

Leader is CatBoost; 2 of 6 rivals are not separable from it.

### 3.3 `300k_temporal_tuned` — chronological, tuned

| Model | Accuracy | F1-macro | 95% CI | MCC | False-alarm rate | Missed (FN) |
|---|---|---|---|---|---|---|
| CatBoost | 0.9835 | 0.7751 | [0.7116, 0.8673] | 0.9451 | 0.147% | 891 |
| Random Forest | 0.9831 | 0.7290 | [0.6797, 0.8094] | 0.9440 | 0.155% | 896 |
| XGBoost | 0.9834 | 0.7021 | [0.6665, 0.8085] | 0.9448 | 0.152% | 867 |
| LightGBM | 0.9836 | 0.6819 | [0.6735, 0.7883] | 0.9455 | 0.119% | 889 |
| Logistic Regression | 0.9676 | 0.6725 | [0.6185, 0.7507] | 0.8905 | 0.544% | 1,925 |
| Stacking | 0.9836 | 0.6580 | [0.6401, 0.7610] | 0.9455 | **0.023%** | 952 |
| MLP | 0.9808 | 0.6003 | [0.5932, 0.6934] | 0.9360 | 0.209% | 1,007 |

Leader is CatBoost; 3 of 6 rivals are not separable from it.

Logistic Regression is byte-identical between §3.2 and §3.3 because the search selected `C = 1.0`, which is the untuned default. That is a search result, not a bug: the linear model's regularisation strength was already at its optimum on this data.

## 4. Why none of this is measurable

Per-class F1 under `300k_temporal_tuned` for the four smallest classes:

| Class | Test flows | CatBoost | LightGBM | Random Forest | XGBoost | MLP | Stacking | LogReg |
|---|---|---|---|---|---|---|---|---|
| SQL Injection | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Brute Force -XSS | 2 | 0.667 | 0.000 | 0.667 | 0.000 | 0.000 | 0.000 | 0.667 |
| Brute Force -Web | 4 | 0.667 | 0.000 | 0.000 | 0.400 | 0.000 | 0.000 | 0.000 |
| DDOS attack-LOIC-UDP | 10 | 1.000 | 0.947 | 0.947 | 0.889 | 0.000 | 0.750 | 0.824 |

SQL Injection has one test flow. Its F1 can only ever be 0.000 or 1.000, and one flow is worth 0.067 of macro-F1 on its own. Brute Force -XSS has two, and its column swings between 0.000 and 0.667 across models that are otherwise nearly identical.

These three classes contribute 3/15 = 20% of the macro-F1 weight while carrying 7 of 90,011 test flows — 0.008% of the evidence. Macro-F1 at this sample size is therefore not primarily a measurement of the model. Comparing protocols through it compares which protocol happened to place a luckier handful of flows in the test set.

This is the same limitation the v2 report identified for the random-split run. Changing the split protocol did not fix it and could not have: the constraint comes from the corpus, where SQL Injection is roughly 87 rows in 13 million.

## 5. Method notes

### 5.1 Cross-validation with single-row classes

Under the chronological split, SQL Injection contributes **one** training row and Brute Force -XSS two. Plain `StratifiedKFold(3)` warns and then places the single-row class in a *validation* fold, where the model has by construction never seen it — its F1 is 0 for reasons unrelated to the hyperparameters, and the penalty varies with the fold seed.

The search therefore uses a splitter that pins any class with fewer than `n_splits` members to every training fold and excludes it from every validation fold. The consequence is recorded rather than hidden: **the search score says nothing about `SQL Injection` or `Brute Force -XSS`.** Both are named in `tuning.json` for every model.

### 5.2 Search budget

20 candidates, 3 folds, macro-F1, on a 60,001-row stratified subsample that keeps every row of every class under 1,000 members. Search ranges are centred on the untuned defaults so the search can confirm them rather than being forced away. LightGBM's `reg_lambda` floor is 1.0 — a correctness bound, not a tuning choice: at 0 a leaf holding SQL Injection's single row has unbounded optimal value and the softmax overflows.

### 5.3 Reproducibility

The chronological split is deterministic and RNG-free; it depends only on row content. Every prediction underlying §3 and §4 was re-derived from the serialised models by rebuilding the test split from scratch, and each re-derived confusion matrix was compared cell-for-cell against the one training wrote. All twenty-one comparisons matched exactly.

## 6. What would actually settle the question

Nothing in this study is fixed by more models or more tuning. The binding constraint is test-set support for the rare classes.

1. **Report macro-F1 over learnable classes only,** with the excluded classes listed and their support given. The current figure averages three classes that cannot be measured.
2. **Raise the sample size.** SQL Injection is ~87 rows in the full corpus; at 300k it is 2. A 1M or full-corpus run is the only way to give it a test set.
3. **Change the question to one this data can answer** — false-alarm rate at fixed recall, or leave-one-attack-out detection, both of which are measured on tens of thousands of flows rather than on one.

The Stacking false-alarm result in §1 is an example of the third: it rests on 74,769 benign flows and is the only finding here that a confidence interval does not swallow.

---

# PART II — ภาษาไทย

## 1. บทสรุปผู้บริหาร

การทดลองนี้ออกแบบมาเพื่อตอบคำถามเดียว: การแบ่งข้อมูลตามเวลา หรือการปรับไฮเปอร์พารามิเตอร์ ทำให้ความสามารถของโมเดลทั้ง 7 ตัวบน CSE-CIC-IDS2018 เปลี่ยนไปอย่างมีนัยสำคัญหรือไม่

**คำตอบคือ ที่ขนาดตัวอย่างเท่านี้ตอบไม่ได้ และเหตุผลที่ตอบไม่ได้มีประโยชน์กว่าคำตอบเสียอีก**

ค่า Macro-F1 ขยับได้ถึง +0.099 ระหว่างโปรโตคอล แต่ช่วงความเชื่อมั่น 95% แบบ bootstrap ของ Macro-F1 **กว้างเฉลี่ย 0.128** ทุก effect ที่วัดได้เล็กกว่าช่วงความเชื่อมั่นของตัวมันเอง ไม่มีคู่ไหนแยกออกจากกันได้เลย

สาเหตุเป็นเชิงโครงสร้าง ไม่ใช่ noise ที่เพิ่มจำนวน bootstrap แล้วจะหาย — Macro-F1 เฉลี่ย 15 คลาสด้วยน้ำหนักเท่ากัน แต่ 3 คลาสในนั้นมี test flow เพียง **1, 2 และ 4 แถว** การที่ flow เดียวเปลี่ยนคลาสที่ทำนาย ทำให้ Macro-F1 ขยับ 1/15 = **0.067** ซึ่งใหญ่กว่าทุก effect ในตาราง

### ตาราง 2×2 ที่เติมแล้ว

| การแบ่งข้อมูล | ไม่ tune | tune |
|---|---|---|
| Random stratified 70/30 | `300k` ✓ | ไม่ได้รัน |
| ตามเวลา (ต่อวัน ต่อคลาส) | `300k_temporal` ✓ | `300k_temporal_tuned` ✓ |

ช่องที่ขาดคือช่องที่จะให้ตัวเลขสวยที่สุดและเชื่อถือได้น้อยที่สุด จึงตั้งใจไม่รัน

### การแยก effect ของ Macro-F1

| โมเดล | Random+ไม่ tune | Temporal+ไม่ tune | Temporal+tune | ผลจาก split | ผลจาก tune |
|---|---|---|---|---|---|
| CatBoost | 0.6763 | 0.7277 | 0.7751 | +0.0514 | +0.0474 |
| Random Forest | 0.6869 | 0.6833 | 0.7290 | −0.0036 | +0.0457 |
| XGBoost | 0.6854 | 0.7037 | 0.7021 | +0.0183 | −0.0016 |
| LightGBM | 0.6870 | 0.7026 | 0.6819 | +0.0156 | −0.0208 |
| Logistic Regression | 0.6296 | 0.6725 | 0.6725 | +0.0429 | 0.0000 |
| Stacking | 0.6886 | 0.6603 | 0.6580 | −0.0283 | −0.0023 |
| MLP | 0.6698 | 0.7168 | 0.6003 | +0.0470 | **−0.1165** |

ทุกค่าใน 2 คอลัมน์ขวาอยู่ในช่วงความเชื่อมั่นของตัวเลขที่ใช้คำนวณมันขึ้นมา และเครื่องหมายก็ไม่คงที่ข้ามโมเดล — tuning ช่วย 2 ตัว ทำให้แย่ลง 3 ตัว และไม่มีผลกับอีก 2 ตัว

### ข้อค้นพบ 3 ข้อที่รอดจาก noise

**1. Accuracy ไม่ให้ข้อมูลอะไรเลยในงานนี้ และไม่ให้มาตั้งแต่ก่อนการทดลอง** ทั้ง 21 คู่ของโมเดล-โปรโตคอลอยู่ระหว่าง 0.9651 ถึง 0.9839 เพราะ Benign คือ 74,769 จาก 90,011 test flows ทำให้ baseline ของการเดาคลาสใหญ่สุดอยู่ที่ 0.8307 อยู่แล้ว Accuracy เป็นตัวเลขที่เปลี่ยนน้อยที่สุดระหว่างโปรโตคอล เพราะมันถูกครอบงำโดยคลาสที่ไม่มีโปรโตคอลไหนไปแตะ

**2. Stacking กลายเป็นโมเดลที่ระมัดระวังมากเมื่อใช้การแบ่งตามเวลา** อัตราแจ้งเตือนเท็จลดจาก 0.135% (101 false positives) ในการแบ่งแบบสุ่ม เหลือ **0.021%** (16 false positives) ในการแบ่งตามเวลา — ลดลง 6 เท่า เป็นการเปลี่ยนแปลงที่ใหญ่ที่สุดในงานทั้งหมด แลกมากับการพลาด attack เพิ่มขึ้นเล็กน้อย (834 → 944) นี่เป็นผลลัพธ์เดียวที่อ่านในเชิงปฏิบัติการได้อย่างสมเหตุสมผล และมันปรากฏในคอลัมน์ false-alarm ไม่ใช่ใน Macro-F1

**3. การ tune บนตัวอย่างย่อยอาจทำให้โมเดลสุดท้ายแย่ลงจริง** MLP เสีย Macro-F1 ไป 0.1165 หลัง tuning เพราะการค้นหาคัดเลือกบนตัวอย่างย่อย 60,001 แถว แล้วนำค่าที่ชนะไป refit บนข้อมูลเต็ม 209,989 แถว — ค่าที่ดีที่สุดที่ 60k ไม่จำเป็นต้องดีที่สุดที่ 210k และสำหรับ neural network ที่มี early stopping มันชัดเจนว่าไม่ใช่ ส่วนกลุ่ม tree ensemble อ่อนไหวกับเรื่องนี้น้อยกว่ามาก

## 2. แต่ละการรันคืออะไร

ทั้งสามใช้ร่วมกัน: ตัวอย่าง stratified 300,000 แถวจากคลังข้อมูล ~13 ล้านแถว, `MIN_PER_CLASS=2`, seed 42, 15 คลาส, 69 features หลังตัดคอลัมน์ระบุตัวตน เวลา และคอลัมน์ค่าคงที่ออก, `class_weighting = none`, ใช้ CPU อย่างเดียว

### 2.1 `300k` — สุ่มแบบ stratified

แบ่งแถวแบบ stratified 70/30 ได้ 210,000 / 90,000 เป็นโปรโตคอลที่งานตีพิมพ์เรื่อง CICIDS ส่วนใหญ่ใช้ จุดอ่อนของมันถูกระบุไว้บนหน้า dashboard เอง: flow ที่เกิดจาก attack burst เดียวกันสามารถตกไปอยู่ทั้งสองฝั่งของการแบ่งได้ ทำให้โมเดลถูกวัดด้วย traffic ที่แทบจะเหมือนกับที่มันเคยเทรน **ทุกคะแนนภายใต้โปรโตคอลนี้ต้องอ่านว่าเป็นขอบบน**

### 2.2 `300k_temporal` — ตามเวลา ไม่ tune

สำหรับทุกกลุ่ม `(วันที่บันทึก, คลาส)` flow ที่เก่าที่สุด 70% ตาม `Timestamp` จริงใช้เทรน และ 30% ล่าสุดใช้ทดสอบ ได้ 209,989 / 90,011 กลุ่มที่มีแถวเดียวถูกส่งไปฝั่งเทรนทั้งหมด

การจัดกลุ่มตามวันด้วย ไม่ใช่ตามคลาสอย่างเดียว มีผลกับ 2 คลาสที่ทำงานข้ามวัน (`Infilteration`, `Brute Force -Web`) — ถ้าตัดจุดเดียวต่อคลาสจะทำให้ทั้งวันหนึ่งกลายเป็น test ทั้งหมด

`Timestamp` ถูกใช้**เพียงเพื่อเรียงลำดับ** และถูกตัดทิ้งก่อนถึงตัว preprocessor โดยมี assertion บังคับไว้ เพราะคอลัมน์นี้เป็นตัวแทนของ label เกือบสมบูรณ์ (แต่ละ attack ทำงานในกรอบเวลาของตัวเอง) ถ้าปล่อยให้รั่วจะทำให้ทุกคะแนนใน bundle พองขึ้น จำนวน feature เท่ากับ 69 ทั้งที่นี่และในการแบ่งแบบสุ่ม ซึ่งเป็นหลักฐานที่สังเกตได้ว่าไม่มีอะไรรั่ว

**ข้อจำกัดที่ต้องพูดตรงๆ ของโปรโตคอลนี้** — 12 จาก 15 คลาสเกิดขึ้นในวันบันทึกเพียงวันเดียว สำหรับคลาสเหล่านั้น การแบ่งจึงเป็นการเรียงลำดับ*ภายใน attack burst เดียว* ไม่ใช่ข้ามวันที่เป็นอิสระต่อกัน มันตัดความมองโลกในแง่ดีของการแบ่งแบบสุ่มออกได้ — ไม่มี test flow ใดเกิดก่อน training flow ของคลาสตัวเอง — แต่**ไม่ได้พิสูจน์ว่าโมเดลจะทำงานได้กับแคมเปญโจมตีครั้งใหม่ที่จัดฉากแยกกัน** การจะพิสูจน์ข้อนั้นต้องใช้โปรโตคอลแบบกันทั้ง capture ไว้ หรือข้าม dataset ซึ่งคลังข้อมูลนี้รองรับไม่ได้ที่ระดับคลาส

### 2.3 `300k_temporal_tuned` — ตามเวลา + tune

ใช้การแบ่งเดียวกัน โมเดล 6 ตัวได้รับการค้นหาแบบสุ่ม: 20 ผู้สมัคร × cross-validation 3 fold วัดด้วย Macro-F1 รันบนตัวอย่างย่อยแบบ stratified 60,001 แถวของฝั่งเทรน แล้วนำค่าที่ชนะไป refit บนข้อมูลเทรนเต็ม 209,989 แถว

Stacking ไม่ถูกค้นหา เพราะ `StackingClassifier` ต้อง refit base learner ทั้ง 3 ตัว `cv+1` รอบต่อการ fit หนึ่งครั้ง การค้นหา 20×3 จะกินราว 720 base-learner fits เพื่อแลกกับไฮเปอร์พารามิเตอร์เพียง 2 ตัวที่มันเปิดให้ปรับจริง มันจึงรับค่าที่ tune แล้วของ base learner ไปใช้แทน

## 3. ผลลัพธ์

### 3.1 `300k` — สุ่ม ไม่ tune

| โมเดล | Accuracy | F1-macro | ช่วงเชื่อมั่น 95% | MCC | อัตราแจ้งเตือนเท็จ | พลาด (FN) |
|---|---|---|---|---|---|---|
| Stacking | 0.9839 | 0.6886 | [0.6759, 0.8035] | 0.9465 | 0.135% | 834 |
| LightGBM | 0.9837 | 0.6870 | [0.6747, 0.8020] | 0.9460 | 0.139% | 842 |
| Random Forest | 0.9828 | 0.6869 | [0.6762, 0.8004] | 0.9431 | 0.199% | 874 |
| XGBoost | 0.9837 | 0.6854 | [0.6728, 0.8003] | 0.9460 | 0.118% | 860 |
| CatBoost | 0.9836 | 0.6763 | [0.6668, 0.7877] | 0.9454 | **0.062%** | 914 |
| MLP | 0.9816 | 0.6698 | [0.6594, 0.7808] | 0.9389 | 0.128% | 1,004 |
| Logistic Regression | 0.9651 | 0.6296 | [0.6168, 0.7339] | 0.8823 | 0.815% | 1,912 |

ผู้นำคือ Stacking และ 3 จาก 7 คู่แข่งแยกจากมันไม่ได้ที่ p > 0.05

### 3.2 `300k_temporal` — ตามเวลา ไม่ tune

| โมเดล | Accuracy | F1-macro | ช่วงเชื่อมั่น 95% | MCC | อัตราแจ้งเตือนเท็จ | พลาด (FN) |
|---|---|---|---|---|---|---|
| CatBoost | 0.9833 | 0.7277 | [0.6803, 0.8054] | 0.9444 | 0.131% | 920 |
| MLP | 0.9814 | 0.7168 | [0.6688, 0.7944] | 0.9380 | 0.163% | 1,038 |
| XGBoost | 0.9835 | 0.7037 | [0.6682, 0.8102] | 0.9451 | 0.166% | 867 |
| LightGBM | 0.9832 | 0.7026 | [0.6682, 0.8056] | 0.9441 | 0.159% | 880 |
| Random Forest | 0.9827 | 0.6833 | [0.6767, 0.7899] | 0.9424 | 0.205% | 917 |
| Logistic Regression | 0.9676 | 0.6725 | [0.6185, 0.7507] | 0.8905 | 0.544% | 1,925 |
| Stacking | 0.9837 | 0.6603 | [0.6421, 0.7652] | 0.9458 | **0.021%** | 944 |

ผู้นำคือ CatBoost และ 2 จาก 6 คู่แข่งแยกจากมันไม่ได้

### 3.3 `300k_temporal_tuned` — ตามเวลา + tune

| โมเดล | Accuracy | F1-macro | ช่วงเชื่อมั่น 95% | MCC | อัตราแจ้งเตือนเท็จ | พลาด (FN) |
|---|---|---|---|---|---|---|
| CatBoost | 0.9835 | 0.7751 | [0.7116, 0.8673] | 0.9451 | 0.147% | 891 |
| Random Forest | 0.9831 | 0.7290 | [0.6797, 0.8094] | 0.9440 | 0.155% | 896 |
| XGBoost | 0.9834 | 0.7021 | [0.6665, 0.8085] | 0.9448 | 0.152% | 867 |
| LightGBM | 0.9836 | 0.6819 | [0.6735, 0.7883] | 0.9455 | 0.119% | 889 |
| Logistic Regression | 0.9676 | 0.6725 | [0.6185, 0.7507] | 0.8905 | 0.544% | 1,925 |
| Stacking | 0.9836 | 0.6580 | [0.6401, 0.7610] | 0.9455 | **0.023%** | 952 |
| MLP | 0.9808 | 0.6003 | [0.5932, 0.6934] | 0.9360 | 0.209% | 1,007 |

ผู้นำคือ CatBoost และ 3 จาก 6 คู่แข่งแยกจากมันไม่ได้

Logistic Regression มีค่าเหมือนกันทุกหลักระหว่าง §3.2 กับ §3.3 เพราะการค้นหาเลือก `C = 1.0` ซึ่งเป็นค่าเริ่มต้นอยู่แล้ว นี่คือผลของการค้นหา ไม่ใช่ข้อผิดพลาด — ค่าความเข้มของ regularisation ของโมเดลเชิงเส้นอยู่ที่จุดเหมาะสมตั้งแต่แรก

## 4. ทำไมทั้งหมดนี้จึงวัดไม่ได้

ค่า F1 รายคลาสของ 4 คลาสที่เล็กที่สุด ภายใต้ `300k_temporal_tuned`:

| คลาส | test flows | CatBoost | LightGBM | Random Forest | XGBoost | MLP | Stacking | LogReg |
|---|---|---|---|---|---|---|---|---|
| SQL Injection | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Brute Force -XSS | 2 | 0.667 | 0.000 | 0.667 | 0.000 | 0.000 | 0.000 | 0.667 |
| Brute Force -Web | 4 | 0.667 | 0.000 | 0.000 | 0.400 | 0.000 | 0.000 | 0.000 |
| DDOS attack-LOIC-UDP | 10 | 1.000 | 0.947 | 0.947 | 0.889 | 0.000 | 0.750 | 0.824 |

SQL Injection มี test flow เดียว ค่า F1 ของมันเป็นได้แค่ 0.000 หรือ 1.000 เท่านั้น และ flow เดียวนั้นมีน้ำหนักถึง 0.067 ของ Macro-F1 ส่วน Brute Force -XSS มี 2 แถว คอลัมน์ของมันจึงเหวี่ยงระหว่าง 0.000 กับ 0.667 ข้ามโมเดลที่โดยรวมแล้วแทบไม่ต่างกันเลย

สามคลาสนี้กินน้ำหนัก Macro-F1 ไป 3/15 = 20% ในขณะที่ถือ test flow เพียง 7 จาก 90,011 แถว หรือ **0.008% ของหลักฐานทั้งหมด** Macro-F1 ที่ขนาดตัวอย่างเท่านี้จึงไม่ใช่การวัดตัวโมเดลเป็นหลัก และการเอามันมาเทียบโปรโตคอล คือการเทียบว่าโปรโตคอลไหนบังเอิญวาง flow ไม่กี่แถวลงใน test set ได้โชคดีกว่ากัน

นี่คือข้อจำกัดเดียวกับที่รายงาน v2 ระบุไว้สำหรับการรันแบบสุ่ม การเปลี่ยนโปรโตคอลการแบ่งไม่ได้แก้ปัญหานี้ และแก้ไม่ได้อยู่แล้ว เพราะข้อจำกัดมาจากตัวคลังข้อมูล ที่ SQL Injection มีอยู่ราว 87 แถวใน 13 ล้านแถว

## 5. บันทึกวิธีการ

### 5.1 Cross-validation กับคลาสที่มีแถวเดียว

ภายใต้การแบ่งตามเวลา SQL Injection มี training row **แถวเดียว** และ Brute Force -XSS มี 2 แถว การใช้ `StratifiedKFold(3)` ตรงๆ จะเตือนแล้วเอาคลาสแถวเดียวไปไว้ฝั่ง *validation* ซึ่งโมเดลไม่เคยเห็นมาก่อนโดยโครงสร้าง ค่า F1 ของมันจึงเป็น 0 ด้วยเหตุผลที่ไม่เกี่ยวกับไฮเปอร์พารามิเตอร์เลย และบทลงโทษนี้ยังเปลี่ยนไปตาม seed ของ fold

การค้นหาจึงใช้ splitter ที่ตรึงคลาสซึ่งมีสมาชิกน้อยกว่า `n_splits` ไว้ในทุก training fold และกันออกจากทุก validation fold ผลที่ตามมาถูกบันทึกไว้ ไม่ได้ซ่อน: **คะแนนของการค้นหาไม่ได้บอกอะไรเกี่ยวกับ `SQL Injection` และ `Brute Force -XSS` เลย** ทั้งสองถูกระบุชื่อไว้ใน `tuning.json` ของทุกโมเดล

### 5.2 งบการค้นหา

20 ผู้สมัคร, 3 fold, วัดด้วย Macro-F1, บนตัวอย่างย่อยแบบ stratified 60,001 แถว ที่เก็บทุกแถวของทุกคลาสที่มีสมาชิกต่ำกว่า 1,000 ไว้ครบ ช่วงการค้นหาถูกวางให้อยู่รอบค่าเริ่มต้นที่ไม่ tune เพื่อให้การค้นหา*ยืนยัน*ค่าเดิมได้ ไม่ใช่ถูกบังคับให้หนีจากมัน ค่า `reg_lambda` ของ LightGBM มีพื้นที่ 1.0 ซึ่งเป็นขอบเขตความถูกต้อง ไม่ใช่ทางเลือกในการ tune — ที่ค่า 0 ใบไม้ที่ถือ SQL Injection แถวเดียวจะมีค่าเหมาะสมที่ไม่มีขอบเขต และ softmax จะล้น

### 5.3 การทำซ้ำได้

การแบ่งตามเวลาเป็น deterministic และไม่ใช้ตัวสุ่มเลย ผลขึ้นกับเนื้อหาของแถวเท่านั้น ทุกการทำนายที่อยู่เบื้องหลัง §3 และ §4 ถูกสร้างขึ้นใหม่จากโมเดลที่บันทึกไว้ โดยประกอบ test split ขึ้นมาใหม่ตั้งแต่ต้น และ confusion matrix ที่ได้ถูกเทียบกับที่ training เขียนไว้ทีละเซลล์ ทั้ง 21 การเปรียบเทียบตรงกันทุกช่อง

## 6. สิ่งที่จะตอบคำถามนี้ได้จริง

ไม่มีอะไรในงานนี้ที่แก้ได้ด้วยการเพิ่มโมเดลหรือ tune หนักขึ้น ข้อจำกัดที่แท้จริงคือจำนวน test flow ของคลาสหายาก

1. **รายงาน Macro-F1 เฉพาะคลาสที่เรียนรู้ได้** พร้อมระบุคลาสที่ถูกกันออกและจำนวน flow ของมัน ตัวเลขปัจจุบันเฉลี่ยรวม 3 คลาสที่วัดไม่ได้เข้าไปด้วย
2. **เพิ่มขนาดตัวอย่าง** SQL Injection มีราว 87 แถวในคลังเต็ม แต่ที่ 300k เหลือ 2 แถว การรันที่ 1M หรือคลังเต็มเป็นทางเดียวที่จะให้มันมี test set
3. **เปลี่ยนคำถามเป็นคำถามที่ข้อมูลนี้ตอบได้** เช่น อัตราแจ้งเตือนเท็จที่ recall คงที่ หรือการทดสอบแบบกัน attack ออกทั้งคลาส (leave-one-attack-out) ซึ่งวัดบน flow หลักหมื่นแทนที่จะวัดบน flow เดียว

ผลของ Stacking ในหัวข้อ 1 เป็นตัวอย่างของข้อ 3 — มันตั้งอยู่บน benign flow 74,769 แถว และเป็นข้อค้นพบเดียวในงานนี้ที่ช่วงความเชื่อมั่นไม่กลืนมันหายไป
