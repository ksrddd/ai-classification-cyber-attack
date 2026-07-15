# Notebooks

Narrated exploration + teaching companion to the code in `src/`.

| Notebook | Phase | Purpose |
|----------|-------|---------|
| `01_EDA.ipynb`           | 4 | Load combined CICIDS2017 + CSE-CIC-IDS2018, inspect, plot class distribution |
| `02_Preprocessing.ipynb` | 4 | Clean, encode, split, scale — show why each step |
| `03_Training.ipynb`      | 6 | Train LR / RF / MLP, run CV |
| `04_Evaluation.ipynb`    | 8 | Metrics, confusion matrices, comparison report |
| `05_SHAP.ipynb`          | 9 | TreeExplainer on RF, summary + force plots |

**Rule**: notebooks import from `src/`; they don't re-implement logic.
That way the panel can re-run the notebook and the code lives in one
canonical place.

These files are committed empty until each phase fills them in.
