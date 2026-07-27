"""FastAPI backend — serves ML results to the Next.js dashboard."""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.constants import (  # noqa: E402
    FIGURES_DIR,
    NON_SCALAR_METRIC_KEYS,  # noqa: E402
)
from src.config.loader import (  # noqa: E402
    get_classification_mode,
    load_config,
)
from src.inference.predictor import (  # noqa: E402
    CHAMPION_PATH,  # noqa: E402
    list_saved_models,
    predict_dataframe,
    published_run_dir,
)

app = FastAPI(title="CyberML API", version="1.0.0")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# A fresh clone has no results/ tree yet, and StaticFiles refuses to mount a
# missing directory at import time. Create it so the API starts before any
# stage has run.
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")

def _run_metrics() -> dict:
    """Aggregate metrics.json of the published run."""
    p = published_run_dir() / "metrics.json"
    if not p.exists():
        raise HTTPException(
            404,
            "No trained run found. Train and promote one first: "
            "python main.py --stage train --run-name <name>",
        )
    return json.loads(p.read_text(encoding="utf-8"))


_cfg = None
def cfg():
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg


def _safe_component(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise HTTPException(400, "Invalid path component")
    return value


async def _read_upload_limited(
    file: UploadFile,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> bytes:
    """Read an upload in bounded chunks and stop as soon as the limit is exceeded."""
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(UPLOAD_CHUNK_BYTES):
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(413, f"CSV exceeds {max_bytes // (1024 * 1024)} MB limit")
        chunks.append(chunk)
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

@app.get("/api/overview")
def get_overview():
    """Corpus + split summary, read from the published run bundle.

    Earlier versions read data/processed/{train,test}.parquet and
    label_classes.json, artifacts of the legacy src/pipelines path that no
    longer exists. The run's own metrics.json already carries every one of
    these numbers -- the same file /api/run reads -- so there is nothing left
    to compute separately, and the class list here can never drift from the
    9 classes the models were actually trained on.
    """
    payload = _run_metrics()
    class_names = payload.get("class_names", [])
    per_train = payload.get("per_class_n_train", {})
    per_test = payload.get("per_class_n_test", {})
    label_distribution = {
        name: int(per_train.get(name, 0)) + int(per_test.get(name, 0))
        for name in class_names
    }

    c = cfg()
    return {
        "mode": get_classification_mode(c),
        "labels": class_names,
        "row_counts": {
            "train": payload.get("n_train", 0),
            "test": payload.get("n_test", 0),
        },
        "label_distribution": label_distribution,
        "n_features": payload.get("n_features", 0),
        "shape": [
            payload.get("n_train", 0) + payload.get("n_test", 0),
            payload.get("n_features", 0),
        ],
        "classification_mode": "multiclass",
        "missing_total": 0,
        "infinite_total": 0,
        "duplicate_row_count": 0,
        "label_classes": class_names,
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@app.get("/api/models")
def get_models():
    return {"models": list_saved_models()}


@app.get("/api/models/{name}/metrics")
def get_model_metrics(name: str):
    """Per-model metrics straight from the published run bundle."""
    name = _safe_component(name)
    p = published_run_dir() / f"{name}_metrics.json"
    if not p.exists():
        raise HTTPException(404, f"No metrics for {name}")
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/api/models/{name}/report")
def get_classification_report(name: str):
    """Per-class precision/recall/F1 table for one model."""
    name = _safe_component(name)
    p = published_run_dir() / f"{name}_per_class.csv"
    if not p.exists():
        raise HTTPException(404, f"No report for {name}")
    df = pd.read_csv(p, index_col=0)
    return df.round(4).to_dict()


# ---------------------------------------------------------------------------
# Comparison + rankings
# ---------------------------------------------------------------------------

@app.get("/api/compare")
def get_compare():
    """Every model in the published run, keyed by model name.

    Nested per-class mappings are dropped so each value is a scalar the
    dashboard can render in a table cell.
    """
    payload = _run_metrics()
    rows = {}
    for item in payload.get("models", []):
        row = {
            k: v for k, v in item.items()
            if k not in NON_SCALAR_METRIC_KEYS and k != "model"
        }
        rows[str(item.get("model"))] = row
    if not rows:
        raise HTTPException(404, "Published run contains no model results")
    return rows


@app.get("/api/rankings")
def get_rankings():
    """Overall / Security-focused / Deployment winners and the rules used."""
    champion = CHAMPION_PATH
    if not champion.exists():
        raise HTTPException(404, "No promoted run; run --stage promote first")
    payload = json.loads(champion.read_text(encoding="utf-8"))
    return {
        "run_id": payload.get("run_id"),
        "champion_model": payload.get("champion_model"),
        "selection": payload.get("selection", {}),
        "rankings": payload.get("rankings", {}),
    }


@app.get("/api/run")
def get_run():
    """Run-level provenance: split protocol, class list, sample sizes."""
    payload = _run_metrics()
    return {
        k: payload.get(k)
        for k in (
            "run_name", "split_protocol", "dataset_id", "random_state",
            "n_train", "n_test", "n_features", "n_classes", "class_names",
            "per_class_n_train", "per_class_n_test", "hardware",
        )
    }


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

@app.get("/api/shap/{name}")
def get_shap(name: str):
    name = _safe_component(name)
    p = published_run_dir() / "shap" / name / "top_features.json"
    if not p.exists():
        raise HTTPException(404, f"No SHAP for {name}")
    return json.loads(p.read_text())


@app.get("/api/shap/{name}/figures/{filename}")
def get_shap_figure(name: str, filename: str):
    name = _safe_component(name)
    filename = _safe_component(filename)
    p = published_run_dir() / "shap" / name / filename
    if not p.exists():
        raise HTTPException(404, f"No SHAP figure {filename} for {name}")
    return FileResponse(str(p))


# ---------------------------------------------------------------------------
# Figures (confusion matrices, etc.)
# ---------------------------------------------------------------------------

@app.get("/api/figures/{filename}")
def get_figure(filename: str):
    filename = _safe_component(filename)
    p = FIGURES_DIR / filename
    if not p.exists():
        raise HTTPException(404, filename)
    return FileResponse(str(p))


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

@app.post("/api/predict")
async def predict(model: str, file: UploadFile = File(...)):  # noqa: B008
    from src.data.schema import clean_column_names
    from src.features.validator import load_expected_features, validate_inference_csv

    content = await _read_upload_limited(file)
    try:
        df = pd.read_csv(io.BytesIO(content), low_memory=False, encoding="latin-1")
    except Exception as e:
        raise HTTPException(400, f"Cannot read CSV: {e}") from e

    df = clean_column_names(df)
    expected = load_expected_features()
    report = validate_inference_csv(df, expected_features=expected)

    if not report.ok:
        raise HTTPException(422, report.message)

    result = predict_dataframe(df, model_name=model, include_probabilities=True)
    preds = result.predictions

    class_counts = preds["predicted_label"].value_counts().to_dict()
    records = preds.head(200).to_dict(orient="records")

    return {
        "model_run_id": result.run_id,
        "contract_version": "feature_columns_v1",
        "n_rows": len(preds),
        "class_counts": class_counts,
        "preview": records,
        "columns": list(preds.columns),
    }
