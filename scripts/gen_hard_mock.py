import csv
import json
import random
from pathlib import Path

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
with (PROJECT_ROOT / "data/processed/feature_names.json").open(encoding="utf-8") as handle:
    features = json.load(handle)

def row(**kwargs):
    r = {f: 0 for f in features}
    r.update(kwargs)
    return r

def jitter(val, pct=0.15):
    return val * (1 + random.uniform(-pct, pct))

rows = []
labels = []

# 1. Low-and-Slow DoS (packet ช้า ดูเหมือน BENIGN)
for _ in range(4):
    r = row(**{
        'Destination Port': random.choice([80, 443, 8080]),
        'Flow Duration': int(jitter(8000000, 0.3)),
        'Total Fwd Packets': random.randint(60, 120),
        'Total Backward Packets': random.randint(2, 8),
        'Total Length of Fwd Packets': random.randint(90000, 180000),
        'Total Length of Bwd Packets': random.randint(300, 800),
        'Fwd Packet Length Max': 1500, 'Fwd Packet Length Min': 1500,
        'Fwd Packet Length Mean': 1500.0, 'Fwd Packet Length Std': 0.0,
        'Bwd Packet Length Max': 200, 'Bwd Packet Length Min': 40,
        'Bwd Packet Length Mean': jitter(80, 0.3), 'Bwd Packet Length Std': jitter(30, 0.4),
        'Flow Bytes/s': jitter(22000, 0.4),
        'Flow Packets/s': jitter(14, 0.3),
        'Flow IAT Mean': jitter(100000, 0.5),
        'Flow IAT Std': jitter(180000, 0.5),
        'Flow IAT Max': random.randint(500000, 2000000),
        'Flow IAT Min': random.randint(100, 2000),
        'Fwd IAT Total': int(jitter(7800000, 0.2)),
        'Fwd IAT Mean': jitter(95000, 0.4),
        'Fwd IAT Max': random.randint(400000, 1500000),
        'Fwd IAT Min': random.randint(50, 500),
        'SYN Flag Count': 1, 'ACK Flag Count': random.randint(60, 110),
        'PSH Flag Count': random.randint(40, 90), 'FIN Flag Count': 1,
        'Fwd Header Length': random.randint(1200, 2400),
        'Bwd Header Length': random.randint(40, 160),
        'Fwd Packets/s': jitter(12, 0.3), 'Bwd Packets/s': jitter(0.8, 0.5),
        'Min Packet Length': 1500, 'Max Packet Length': 1500,
        'Packet Length Mean': jitter(1200, 0.1), 'Packet Length Std': jitter(400, 0.3),
        'Packet Length Variance': jitter(160000, 0.3),
        'Down/Up Ratio': jitter(0.05, 0.5),
        'Average Packet Size': jitter(1200, 0.1),
        'Avg Fwd Segment Size': 1500.0, 'Avg Bwd Segment Size': jitter(80, 0.3),
        'Subflow Fwd Packets': random.randint(60, 120),
        'Subflow Fwd Bytes': random.randint(90000, 180000),
        'Subflow Bwd Packets': random.randint(2, 8),
        'Init_Win_bytes_forward': random.choice([8192, 16384, 65535]),
        'Init_Win_bytes_backward': random.choice([64, 128, 256]),
        'act_data_pkt_fwd': random.randint(55, 115), 'min_seg_size_forward': 20,
    })
    rows.append(r)
    labels.append('DoS (low-and-slow)')

# 2. Bot C2 ที่ดูเหมือน HTTPS BENIGN (encrypted periodic callback)
for _ in range(4):
    r = row(**{
        'Destination Port': random.choice([443, 8443, 4443]),
        'Flow Duration': int(jitter(25000000, 0.4)),
        'Total Fwd Packets': random.randint(8, 20),
        'Total Backward Packets': random.randint(6, 18),
        'Total Length of Fwd Packets': random.randint(1200, 3600),
        'Total Length of Bwd Packets': random.randint(900, 2800),
        'Fwd Packet Length Max': random.randint(200, 600),
        'Fwd Packet Length Min': random.randint(40, 80),
        'Fwd Packet Length Mean': jitter(180, 0.3),
        'Bwd Packet Length Max': random.randint(200, 500),
        'Bwd Packet Length Mean': jitter(160, 0.3),
        'Flow Bytes/s': jitter(210, 0.4),
        'Flow Packets/s': jitter(1.1, 0.3),
        'Flow IAT Mean': jitter(1800000, 0.5),
        'Flow IAT Std': jitter(200000, 0.3),
        'Flow IAT Max': int(jitter(2200000, 0.2)),
        'Flow IAT Min': int(jitter(1400000, 0.2)),
        'Fwd IAT Mean': jitter(3600000, 0.3),
        'Fwd IAT Std': jitter(300000, 0.2),
        'Bwd IAT Mean': jitter(3500000, 0.3),
        'Bwd IAT Std': jitter(280000, 0.2),
        'SYN Flag Count': 1, 'ACK Flag Count': random.randint(8, 18),
        'PSH Flag Count': random.randint(5, 14), 'FIN Flag Count': 1,
        'Fwd Header Length': random.randint(160, 400),
        'Bwd Header Length': random.randint(120, 360),
        'Fwd Packets/s': jitter(0.9, 0.3), 'Bwd Packets/s': jitter(0.7, 0.3),
        'Min Packet Length': random.randint(40, 80),
        'Max Packet Length': random.randint(300, 700),
        'Packet Length Mean': jitter(175, 0.3),
        'Down/Up Ratio': jitter(0.9, 0.2),
        'Average Packet Size': jitter(175, 0.3),
        'Avg Fwd Segment Size': jitter(180, 0.3),
        'Avg Bwd Segment Size': jitter(160, 0.3),
        'Subflow Fwd Packets': random.randint(8, 20),
        'Init_Win_bytes_forward': random.choice([65535, 32768]),
        'Init_Win_bytes_backward': random.choice([65535, 32768]),
        'act_data_pkt_fwd': random.randint(6, 18), 'min_seg_size_forward': 20,
        'Idle Mean': jitter(22000000, 0.3),
        'Idle Std': jitter(200000, 0.2),
        'Idle Max': int(jitter(23000000, 0.2)),
        'Idle Min': int(jitter(21000000, 0.2)),
    })
    rows.append(r)
    labels.append('Bot (encrypted C2)')

# 3. Slow PortScan (ช้า หลาย connection ดูเหมือน BENIGN)
for _ in range(4):
    r = row(**{
        'Destination Port': random.randint(1, 65535),
        'Flow Duration': int(jitter(3000000, 0.8)),
        'Total Fwd Packets': random.randint(2, 5),
        'Total Backward Packets': random.randint(0, 2),
        'Total Length of Fwd Packets': random.randint(120, 480),
        'Total Length of Bwd Packets': random.randint(0, 200),
        'Fwd Packet Length Max': random.randint(60, 120),
        'Fwd Packet Length Min': random.randint(40, 70),
        'Fwd Packet Length Mean': jitter(70, 0.2),
        'Flow Bytes/s': jitter(180, 0.6),
        'Flow Packets/s': jitter(1.8, 0.5),
        'Flow IAT Mean': jitter(900000, 0.6),
        'Flow IAT Std': jitter(700000, 0.5),
        'SYN Flag Count': random.choice([0, 1]),
        'ACK Flag Count': random.randint(1, 4),
        'RST Flag Count': random.choice([0, 0, 0, 1]),
        'Fwd Header Length': random.randint(40, 100),
        'Bwd Header Length': random.randint(0, 40),
        'Fwd Packets/s': jitter(1.5, 0.4), 'Bwd Packets/s': jitter(0.4, 0.6),
        'Min Packet Length': random.randint(40, 70),
        'Max Packet Length': random.randint(80, 150),
        'Packet Length Mean': jitter(65, 0.2),
        'Down/Up Ratio': jitter(0.4, 0.5),
        'Average Packet Size': jitter(65, 0.2),
        'Avg Fwd Segment Size': jitter(70, 0.2),
        'Subflow Fwd Packets': random.randint(2, 5),
        'Init_Win_bytes_forward': random.choice([1024, 2048, 4096, -1]),
        'Init_Win_bytes_backward': random.choice([0, -1, 512]),
        'act_data_pkt_fwd': random.randint(1, 4), 'min_seg_size_forward': 20,
    })
    rows.append(r)
    labels.append('PortScan (slow)')

# 4. Web Attack ซ่อนใน HTTPS (payload ใหญ่ผิดปกติ)
for _ in range(4):
    r = row(**{
        'Destination Port': 443,
        'Flow Duration': int(jitter(800000, 0.4)),
        'Total Fwd Packets': random.randint(4, 9),
        'Total Backward Packets': random.randint(3, 8),
        'Total Length of Fwd Packets': random.randint(15000, 60000),
        'Total Length of Bwd Packets': random.randint(2000, 8000),
        'Fwd Packet Length Max': random.randint(8000, 16000),
        'Fwd Packet Length Min': random.randint(200, 600),
        'Fwd Packet Length Mean': jitter(4500, 0.4),
        'Fwd Packet Length Std': jitter(3800, 0.4),
        'Bwd Packet Length Max': random.randint(1000, 4000),
        'Bwd Packet Length Min': random.randint(60, 200),
        'Bwd Packet Length Mean': jitter(900, 0.4),
        'Flow Bytes/s': jitter(82000, 0.4),
        'Flow Packets/s': jitter(16, 0.3),
        'Flow IAT Mean': jitter(55000, 0.4),
        'Flow IAT Std': jitter(80000, 0.4),
        'Flow IAT Max': random.randint(200000, 600000),
        'Flow IAT Min': random.randint(500, 5000),
        'SYN Flag Count': 1, 'ACK Flag Count': random.randint(5, 15),
        'PSH Flag Count': random.randint(3, 8), 'FIN Flag Count': 1,
        'Fwd Header Length': random.randint(80, 180),
        'Bwd Header Length': random.randint(60, 160),
        'Fwd Packets/s': jitter(9, 0.3), 'Bwd Packets/s': jitter(7, 0.3),
        'Min Packet Length': random.randint(60, 200),
        'Max Packet Length': random.randint(8000, 16000),
        'Packet Length Mean': jitter(3200, 0.3),
        'Packet Length Std': jitter(3600, 0.3),
        'Packet Length Variance': jitter(12960000, 0.3),
        'Down/Up Ratio': jitter(0.7, 0.3),
        'Average Packet Size': jitter(3200, 0.3),
        'Avg Fwd Segment Size': jitter(4500, 0.4),
        'Avg Bwd Segment Size': jitter(900, 0.4),
        'Subflow Fwd Packets': random.randint(4, 9),
        'Init_Win_bytes_forward': random.choice([65535, 32768]),
        'Init_Win_bytes_backward': random.choice([8192, 16384]),
        'act_data_pkt_fwd': random.randint(3, 8), 'min_seg_size_forward': 20,
    })
    rows.append(r)
    labels.append('Web Attack (hidden in HTTPS)')

# 5. Brute Force SSH ที่ช้า (delay ระหว่าง attempt)
for _ in range(4):
    r = row(**{
        'Destination Port': 22,
        'Flow Duration': int(jitter(2500000, 0.5)),
        'Total Fwd Packets': random.randint(8, 20),
        'Total Backward Packets': random.randint(6, 18),
        'Total Length of Fwd Packets': random.randint(800, 2400),
        'Total Length of Bwd Packets': random.randint(600, 2000),
        'Fwd Packet Length Max': random.randint(100, 300),
        'Fwd Packet Length Min': random.randint(30, 60),
        'Fwd Packet Length Mean': jitter(90, 0.3),
        'Bwd Packet Length Max': random.randint(100, 400),
        'Bwd Packet Length Mean': jitter(120, 0.3),
        'Flow Bytes/s': jitter(1600, 0.5),
        'Flow Packets/s': jitter(12, 0.4),
        'Flow IAT Mean': jitter(180000, 0.5),
        'Flow IAT Std': jitter(250000, 0.4),
        'Flow IAT Max': random.randint(600000, 2000000),
        'Flow IAT Min': random.randint(500, 8000),
        'Fwd IAT Mean': jitter(260000, 0.4),
        'Fwd IAT Std': jitter(300000, 0.4),
        'SYN Flag Count': 1, 'ACK Flag Count': random.randint(7, 17),
        'PSH Flag Count': random.randint(5, 14), 'FIN Flag Count': 1,
        'RST Flag Count': random.choice([0, 0, 1]),
        'Fwd Header Length': random.randint(160, 400),
        'Bwd Header Length': random.randint(120, 360),
        'Fwd Packets/s': jitter(7, 0.4), 'Bwd Packets/s': jitter(6, 0.4),
        'Min Packet Length': random.randint(30, 60),
        'Max Packet Length': random.randint(150, 400),
        'Packet Length Mean': jitter(105, 0.3),
        'Down/Up Ratio': jitter(0.85, 0.2),
        'Average Packet Size': jitter(105, 0.3),
        'Avg Fwd Segment Size': jitter(90, 0.3),
        'Avg Bwd Segment Size': jitter(120, 0.3),
        'Subflow Fwd Packets': random.randint(8, 20),
        'Init_Win_bytes_forward': random.choice([4096, 8192]),
        'Init_Win_bytes_backward': random.choice([4096, 8192]),
        'act_data_pkt_fwd': random.randint(6, 18), 'min_seg_size_forward': 20,
    })
    rows.append(r)
    labels.append('Brute Force (slow SSH)')

# 6. DDoS amplification (DNS/NTP reflection — request เล็ก response ใหญ่)
for _ in range(4):
    r = row(**{
        'Destination Port': random.choice([53, 123, 161, 19]),
        'Flow Duration': int(jitter(500, 0.8)),
        'Total Fwd Packets': random.randint(1, 3),
        'Total Backward Packets': random.randint(1, 4),
        'Total Length of Fwd Packets': random.randint(60, 200),
        'Total Length of Bwd Packets': random.randint(3000, 9000),
        'Fwd Packet Length Max': random.randint(60, 200),
        'Fwd Packet Length Min': random.randint(50, 80),
        'Fwd Packet Length Mean': jitter(90, 0.2),
        'Bwd Packet Length Max': random.randint(3000, 9000),
        'Bwd Packet Length Min': random.randint(2000, 5000),
        'Bwd Packet Length Mean': jitter(5000, 0.3),
        'Flow Bytes/s': jitter(17000000, 0.5),
        'Flow Packets/s': jitter(10000, 0.5),
        'Flow IAT Mean': jitter(100, 0.5),
        'Flow IAT Std': jitter(50, 0.5),
        'Flow IAT Max': random.randint(200, 1000),
        'Flow IAT Min': random.randint(10, 100),
        'SYN Flag Count': 0, 'ACK Flag Count': random.randint(1, 4),
        'PSH Flag Count': 0, 'FIN Flag Count': 0,
        'Fwd Header Length': random.randint(20, 60),
        'Bwd Header Length': random.randint(20, 80),
        'Fwd Packets/s': jitter(4000, 0.5), 'Bwd Packets/s': jitter(6000, 0.5),
        'Min Packet Length': random.randint(50, 80),
        'Max Packet Length': random.randint(3000, 9000),
        'Packet Length Mean': jitter(2500, 0.4),
        'Down/Up Ratio': jitter(45, 0.3),
        'Average Packet Size': jitter(2500, 0.4),
        'Avg Fwd Segment Size': jitter(90, 0.2),
        'Avg Bwd Segment Size': jitter(5000, 0.3),
        'Subflow Fwd Packets': random.randint(1, 3),
        'Init_Win_bytes_forward': 0,
        'Init_Win_bytes_backward': 0,
        'act_data_pkt_fwd': random.randint(1, 3), 'min_seg_size_forward': 8,
    })
    rows.append(r)
    labels.append('DDoS (amplification)')

out_path = PROJECT_ROOT / "data/sample/test_hard_cases.csv"
with out_path.open('w', newline='', encoding='utf-8') as f:
    fieldnames = features + ['true_label']
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    w.writeheader()
    for r, lbl in zip(rows, labels, strict=True):
        r['true_label'] = lbl
        w.writerow(r)

print(f'Written {len(rows)} rows -> {out_path}')
for lbl in dict.fromkeys(labels):
    print(f'  {labels.count(lbl)}x  {lbl}')
