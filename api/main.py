"""FastAPI backend — serves ML results to the Next.js dashboard."""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.constants import FIGURES_DIR, METRICS_DIR, PROCESSED_DIR, SHAP_DIR  # noqa: E402
from src.config.loader import (  # noqa: E402
    get_active_target_labels,
    get_classification_mode,
    load_config,
)
from src.inference.predictor import list_saved_models, predict_dataframe  # noqa: E402

app = FastAPI(title="CyberML API", version="1.0.0")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")

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
    eda_path = METRICS_DIR / "eda_summary.json"
    eda = json.loads(eda_path.read_text()) if eda_path.exists() else {}

    row_counts: dict[str, int] = {}
    for key, fname in [("train", "train.parquet"), ("test", "test.parquet")]:
        p = PROCESSED_DIR / fname
        if p.exists():
            row_counts[key] = pq.read_metadata(str(p)).num_rows

    classes_path = PROCESSED_DIR / "label_classes.json"
    classes = json.loads(classes_path.read_text()) if classes_path.exists() else []

    c = cfg()
    return {
        "mode": get_classification_mode(c),
        "labels": list(get_active_target_labels(c)),
        "row_counts": row_counts,
        "label_distribution": eda.get("label_distribution", {}),
        "n_features": eda.get("n_features", 78),
        "shape": eda.get("shape", [0, 0]),
        "classification_mode": eda.get("classification_mode", ""),
        "missing_total": eda.get("missing_total", 0),
        "infinite_total": eda.get("infinite_total", 0),
        "duplicate_row_count": eda.get("duplicate_row_count", 0),
        "label_classes": classes,
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@app.get("/api/models")
def get_models():
    return {"models": list_saved_models()}


@app.get("/api/models/{name}/metrics")
def get_model_metrics(name: str):
    name = _safe_component(name)
    for suffix in ("_test.json", "_val.json"):
        p = METRICS_DIR / f"{name}{suffix}"
        if p.exists():
            return json.loads(p.read_text())
    raise HTTPException(404, f"No metrics for {name}")


@app.get("/api/models/{name}/report")
def get_classification_report(name: str):
    name = _safe_component(name)
    p = METRICS_DIR / f"classification_report_{name}.csv"
    if not p.exists():
        raise HTTPException(404, f"No report for {name}")
    df = pd.read_csv(p, index_col=0)
    return df.round(4).to_dict()


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@app.get("/api/compare")
def get_compare():
    p = METRICS_DIR / "model_comparison.csv"
    if not p.exists():
        raise HTTPException(404, "model_comparison.csv not found")
    df = pd.read_csv(p, index_col=0)
    return df.round(4).to_dict(orient="index")


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

@app.get("/api/shap/{name}")
def get_shap(name: str):
    name = _safe_component(name)
    p = SHAP_DIR / name / "top_features.json"
    if not p.exists():
        raise HTTPException(404, f"No SHAP for {name}")
    return json.loads(p.read_text())


@app.get("/api/shap/{name}/figures/{filename}")
def get_shap_figure(name: str, filename: str):
    name = _safe_component(name)
    filename = _safe_component(filename)
    p = SHAP_DIR / name / filename
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
