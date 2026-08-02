"""Isolated CSE-CIC-IDS2018 pipeline.

This package is deliberately self-contained: it shares **no** preprocessing
code, feature names, or label mappings with the CICIDS2017 pipeline that
lives under ``src/data``, ``src/features`` and ``src/models``. The two
datasets have different column names ("Tot Fwd Pkts" vs "Total Fwd Packets"),
different label spellings, and different junk columns, so mixing them
silently corrupts either side.

Entry point: ``python -m src.ids2018.train_ids2018 --help``
"""

from __future__ import annotations

__all__ = ["config", "data_loader", "preprocessing", "models", "evaluate"]
