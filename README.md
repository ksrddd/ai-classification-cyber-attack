# AI-Based Cyber Attack Classification

ระบบจำแนกประเภทการโจมตีทางไซเบอร์จาก Network Logs ด้วย Machine Learning + Explainable AI

> Senior Project — Faculty of Information Technology, KMITL
> Sirachet Chotithakulanon (66070191) · Sukhum Ridmethakul (66070315)
> Advisor: Asst. Prof. Dr. Prapan Pavarangkoon

---

## Project Status

| Phase | Status |
|-------|--------|
| Phase 1 — Requirement analysis | done |
| Phase 2 — Architecture design  | done |
| Phase 3 — Project scaffold     | done |
| Phase 4 — Data engineering     | not started |
| Phase 5 — Feature engineering  | not started |
| Phase 6 — Model development    | not started |
| Phase 7 — Optimization         | not started |
| Phase 8 — Evaluation           | not started |
| Phase 9 — SHAP / XAI           | not started |
| Phase 10 — Dashboard           | not started |
| Phase 11 — Testing             | not started |
| Phase 12 — MLOps prep          | not started |
| Phase 13 — Documentation       | not started |

---

## What this project does

Reads CICIDS2017 network-flow records and classifies each flow into one of:

- `BENIGN`
- `DoS Hulk`
- `PortScan`
- `FTP-Patator`

Three models are trained and compared: **Logistic Regression**, **Random Forest**, **MLP**.
Predictions are explained using **SHAP**. A **Streamlit** dashboard exposes everything visually.

The system is structured as a real ML project — config-driven, tested, reproducible.

---

## Installation

```powershell
# 1. Clone (or open) the project
cd C:\Users\ks\projects\cyber_attack_classification

# 2. Create + activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -U pip
pip install -r requirements.txt
pip install -e .[dev]   # editable install + dev tools
```

Python 3.10 or newer is required.

---

## Get the dataset

CICIDS2017 must be requested from UNB CIC:
<https://www.unb.ca/cic/datasets/ids-2017.html>

After download, place the CSVs under `data/raw/`:

```
data/raw/
├── Monday-WorkingHours.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv        # FTP-Patator
├── Wednesday-workingHours.pcap_ISCX.csv
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
├── Friday-WorkingHours-Morning.pcap_ISCX.csv
├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
└── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv     # DoS Hulk
```

You don't need every file — `config.yaml::data.required_files` lists the minimum subset
needed for the 4 target classes.

---

## Architecture

See [docs/architecture.md](docs/architecture.md). High-level shape:

```
Raw CSV  →  Loader  →  Cleaning  →  Pipeline(Scaler + Model)  →  CV + GridSearch
                                                                ↓
                                                          Trained .joblib
                                                                ↓
                            Metrics, Confusion Matrix, SHAP, Streamlit Dashboard
```

Key design rules:
- **No data leakage** — all preprocessing lives inside a sklearn `Pipeline`,
  scaler fits per CV fold.
- **Reproducible** — single `RANDOM_STATE=42` constant, stratified splits everywhere.
- **Config-driven** — paths, classes, hyperparameter grids in `src/config/config.yaml`.

---

## Usage (planned — wired up in Phase 6+)

```powershell
# Full pipeline end-to-end
python main.py --stage all

# Individual stages
python main.py --stage eda
python main.py --stage preprocess
python main.py --stage train     --model rf
python main.py --stage evaluate
python main.py --stage explain
python main.py --stage predict   --input my_traffic.csv

# Dashboard
streamlit run dashboard/app.py
```

---

## Project structure

```
cyber_attack_classification/
├── data/{raw,interim,processed,sample}/
├── notebooks/      # 01_EDA → 05_SHAP
├── src/
│   ├── config/         # constants, YAML loader
│   ├── data/           # CSV loader, EDA, schema
│   ├── features/       # cleaning, encoding, selection, sklearn Pipeline
│   ├── models/         # LR / RF / MLP wrappers, GridSearch tuner
│   ├── evaluation/     # metrics, confusion matrix, comparison report
│   ├── explainability/ # SHAP analyzer
│   ├── visualization/  # plot helpers
│   ├── utils/          # logging, I/O, seeds
│   └── pipelines/      # train / evaluate / explain / predict
├── models/         # *.joblib (gitignored)
├── reports/        # generated markdown reports
├── results/{metrics,figures,shap}/
├── dashboard/      # Streamlit app + 6 pages
├── tests/          # pytest
├── docs/
├── main.py         # CLI entry point
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Development

```powershell
# Run tests
pytest

# Lint + format
ruff check src tests
black src tests

# Coverage report (opens results/coverage/index.html)
pytest --cov-report=html
```

---

## Out of scope (deliberate)

- Deep learning beyond MLP (no CNN/LSTM/Transformer)
- Real-time / streaming inference
- Cloud deployment
- GPU training
- pcap parsing — we consume the pre-extracted CICIDS flow CSVs

These are listed as "Future Work" rather than gaps.

---

## License

MIT. See `LICENSE`.
