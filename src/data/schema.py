"""CICIDS2017 schema constants + validation helpers.

This module lists the expected columns, dtypes, and the set of label
values we consider valid. The loader calls into here before returning
a DataFrame, so a schema drift fails loudly and early.

NOTE: The full CICIDS2017 column list (~80 cols) is documented in
``docs/architecture.md``. Filling in ``EXPECTED_COLUMNS`` is a Phase-4
task — we keep this module's surface stable now so callers can import it.
"""

from __future__ import annotations

# TODO(phase 4): populate with the 78 numeric flow features + Label column.
EXPECTED_COLUMNS: tuple[str, ...] = ()

# TODO(phase 4): validate_schema(df) -> None
# Should:
#   1. Strip leading/trailing whitespace from column names.
#   2. Assert EXPECTED_COLUMNS is a subset of df.columns.
#   3. Assert the label column exists and only contains known labels.
