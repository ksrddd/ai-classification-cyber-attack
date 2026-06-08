"""Inference layer -- batch prediction on user-supplied CSVs.

Used by:
- ``main.py --stage predict``
- the Streamlit dashboard's Predict-New-CSV page

Public entry point: :func:`src.inference.predictor.predict_csv`.
"""

from src.inference.predictor import (
    PredictionResult,
    predict_csv,
    predict_dataframe,
)

__all__ = ["PredictionResult", "predict_csv", "predict_dataframe"]
