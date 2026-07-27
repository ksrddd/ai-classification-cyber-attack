"""Regression tests for the canonical cache preprocessing route."""

from __future__ import annotations

import json

import pandas as pd

import main
import train


def test_canonical_cleaning_normalises_abbreviated_rate_columns() -> None:
    """Some CICIDS2017 redistributions ship the short CICFlowMeter names.

    Without normalisation they become differently-named features, so a model
    trained on one distribution cannot score the other.
    """
    raw = pd.DataFrame({
        "Fwd Pkts/s": [1.0, 2.0],
        "Bwd Pkts/s": [3.0, 4.0],
        "Label": ["BENIGN", "DDoS"],
    })

    cleaned = train._clean_one_frame(raw, train.CONFIG)

    assert "Fwd Packets/s" in cleaned
    assert "Bwd Packets/s" in cleaned
    assert "Fwd Pkts/s" not in cleaned
    assert "Bwd Pkts/s" not in cleaned


def test_canonical_cleaning_drops_leaky_and_schema_only_columns() -> None:
    raw = pd.DataFrame({
        "Flow Duration": [1.0, 2.0],
        " Destination Port": [80, 443],
        "Timestamp": ["5/7/2017 8:47", "5/7/2017 8:48"],
        "Protocol": [6, 17],
        "Label": ["BENIGN", "DDoS"],
    })

    cleaned = train._clean_one_frame(raw, train.CONFIG)

    # Column names are stripped of the leading space CICIDS2017 ships with.
    assert "Destination Port" in cleaned
    # Leaky and schema-only columns never reach the feature matrix.
    assert "Timestamp" not in cleaned
    assert "Protocol" not in cleaned
    assert not cleaned.isna().any().any()


def test_preprocess_stage_uses_canonical_cache_builder(monkeypatch, capsys) -> None:
    summary = {"cache_path": "data/processed/cicids2017_clean.parquet", "rows": 12}
    monkeypatch.setattr(train, "preprocess_cache", lambda *, force: summary)

    assert main.main(["--stage", "preprocess", "--refresh-cache"]) == 0
    assert json.loads(capsys.readouterr().out) == summary


def test_preprocess_stage_forwards_refresh_flag(monkeypatch, capsys) -> None:
    seen: dict[str, object] = {}

    def fake(*, force: bool) -> dict:
        seen["force"] = force
        return {"rows": 1}

    monkeypatch.setattr(train, "preprocess_cache", fake)

    assert main.main(["--stage", "preprocess"]) == 0
    capsys.readouterr()
    assert seen == {"force": False}


def test_cleaning_records_original_csv_row_order() -> None:
    """_row_index must survive cleaning and keep raw CSV positions.

    It is the ordering key for the temporal split, so a row dropped by
    cleaning has to leave a gap rather than renumber its successors.
    """
    raw = pd.DataFrame({
        "Flow Duration": [1.0, 2.0, float("inf"), 4.0],
        "Label": ["BENIGN", "Label", "DDoS", "DDoS"],
    })

    cleaned = train._clean_one_frame(raw, train.CONFIG)

    # Row 1 is a corrupted header row, row 2 has a non-finite feature.
    assert cleaned[train.ROW_INDEX_COLUMN].tolist() == [0, 3]
    assert cleaned[train.ROW_INDEX_COLUMN].dtype == "int64"


def test_row_index_is_metadata_not_a_feature() -> None:
    """A model must never see the row number: it perfectly encodes the split."""
    assert train.ROW_INDEX_COLUMN in train.CONFIG["metadata_columns"]

    raw = pd.DataFrame({
        "Flow Duration": [1.0, 2.0],
        "Label": ["BENIGN", "DDoS"],
    })
    cleaned = train._clean_one_frame(raw, train.CONFIG)
    feature_cols = [
        c for c in cleaned.columns
        if c != train.CONFIG["label_column"]
        and c not in train.CONFIG["metadata_columns"]
    ]
    assert train.ROW_INDEX_COLUMN not in feature_cols


def test_cache_path_is_the_single_cicids2017_corpus() -> None:
    from src.config.constants import CLEAN_CACHE_PATH

    cache = train.resolve_cache_path(train.CONFIG)

    assert cache.name == "cicids2017_clean.parquet"
    # src/ modules resolve the same file without importing root train.py.
    assert cache == CLEAN_CACHE_PATH
    assert train.DATASET_ID == "CICIDS2017"


def test_every_capture_file_is_required() -> None:
    """Each attack class lives in exactly one capture.

    A silently-missing file would drop a whole class from the corpus, so the
    loader must refuse to build a partial cache rather than train on one.
    """
    assert len(train.CICIDS2017_CAPTURES) == 8

    from src.data.temporal_split import CICIDS2017_SOURCES

    assert set(train.CICIDS2017_CAPTURES) == set(CICIDS2017_SOURCES)


def test_gpu_pipeline_sets_cuda_on_supported_models() -> None:
    xgb = train.build_pipeline("xgboost", 3, 42, accelerator="gpu").named_steps["clf"]
    cat = train.build_pipeline("catboost", 3, 42, accelerator="gpu").named_steps["clf"]

    assert xgb.get_params()["device"] == "cuda"
    assert cat.get_params()["task_type"] == "GPU"
