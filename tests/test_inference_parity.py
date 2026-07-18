from __future__ import annotations

import numpy as np
import pandas as pd

from src.inference.predictor import prepare_inference_features


def test_inference_preserves_nan_for_pipeline_imputer() -> None:
    frame = pd.DataFrame({"a": [np.inf, 2.0], "b": [np.nan, 3.0]})
    prepared = prepare_inference_features(frame, ["a", "b"])
    assert prepared.isna().to_numpy().sum() == 2
    assert prepared.iloc[0, 0] != 0
