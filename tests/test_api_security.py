from __future__ import annotations

import asyncio
import io

import pytest
from fastapi import HTTPException, UploadFile

from api.main import _read_upload_limited, _safe_component


def test_safe_component_rejects_path_traversal() -> None:
    assert _safe_component("random_forest-v1.json") == "random_forest-v1.json"
    with pytest.raises(HTTPException) as exc_info:
        _safe_component("../secret")
    assert exc_info.value.status_code == 400


def test_upload_reader_stops_at_limit() -> None:
    upload = UploadFile(filename="large.csv", file=io.BytesIO(b"123456789"))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(_read_upload_limited(upload, max_bytes=8))
    assert exc_info.value.status_code == 413
