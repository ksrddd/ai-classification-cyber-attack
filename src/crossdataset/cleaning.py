"""One cleaning rule set applied to both datasets.

The advisor's protocol says to clean each dataset separately. Separately does
not mean differently: if 2017 drops its non-finite rows and 2018 imputes them,
the two corpora have passed through different filters and any gap measured
between them carries that difference inside it.

The rules below are the ones the 2017 handoff already applied, lifted verbatim
so that re-running them on the 2017 parquet is a no-op (``assert_idempotent``
checks exactly that) and applying them to 2018 brings it onto the same footing:

1. Coerce every feature column to numeric; unparseable tokens become NaN.
2. Replace ``+/-inf`` with NaN -- one sentinel for "broken value", not two.
3. Drop any row with a non-finite feature.
4. Drop exact duplicate rows, comparing features and label only.

Step 3 is a deletion, not an imputation, and that is deliberate. Imputing a
median teaches the model that a flow with an undefined byte rate looks like a
typical flow, which is false: ``Flow Bytes/s`` is infinite precisely when the
duration is zero, a single-packet flow. The 2017 corpus was built this way and
the cheapest way to make the two comparable is to follow it.

Step 4 compares features and label but ignores metadata, so two flows that are
identical measurements from different capture files still collapse to one. That
is the 2017 behaviour; keeping both would let one dataset carry duplicate
evidence the other does not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CleaningReport:
    """What a cleaning pass removed. Returned rather than logged only, so a
    caller can record it in a run manifest and a test can assert on it."""

    rows_in: int
    rows_out: int
    non_finite_values: int
    rows_dropped_non_finite: int
    rows_dropped_duplicate: int

    @property
    def rows_dropped(self) -> int:
        return self.rows_in - self.rows_out

    @property
    def kept_fraction(self) -> float:
        return self.rows_out / self.rows_in if self.rows_in else 1.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_dropped": self.rows_dropped,
            "non_finite_values": self.non_finite_values,
            "rows_dropped_non_finite": self.rows_dropped_non_finite,
            "rows_dropped_duplicate": self.rows_dropped_duplicate,
            "kept_fraction": round(self.kept_fraction, 6),
        }


def clean(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "Label",
) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply the shared rules and return the cleaned frame plus a report.

    Columns outside ``feature_cols`` and ``label_col`` ride along untouched --
    metadata such as ``_row_index`` must survive, because it is the only
    chronological ordering signal the 2017 export carries.
    """
    rows_in = len(df)
    out = df.copy()

    numeric = out[feature_cols].apply(pd.to_numeric, errors="coerce")
    non_finite_mask = ~np.isfinite(numeric.to_numpy(dtype="float64", na_value=np.nan))
    non_finite_values = int(non_finite_mask.sum())
    out[feature_cols] = numeric.replace([np.inf, -np.inf], np.nan)

    keep = ~out[feature_cols].isna().any(axis=1)
    rows_dropped_non_finite = int((~keep).sum())
    out = out.loc[keep]

    before_dedup = len(out)
    out = out.drop_duplicates(subset=[*feature_cols, label_col])
    rows_dropped_duplicate = before_dedup - len(out)

    report = CleaningReport(
        rows_in=rows_in,
        rows_out=len(out),
        non_finite_values=non_finite_values,
        rows_dropped_non_finite=rows_dropped_non_finite,
        rows_dropped_duplicate=rows_dropped_duplicate,
    )
    logger.info(
        "clean: %d -> %d rows (%.2f%% kept); dropped %d non-finite, %d duplicate",
        report.rows_in,
        report.rows_out,
        100 * report.kept_fraction,
        report.rows_dropped_non_finite,
        report.rows_dropped_duplicate,
    )
    return out.reset_index(drop=True), report


def assert_idempotent(
    df: pd.DataFrame, feature_cols: list[str], label_col: str = "Label"
) -> None:
    """Raise if cleaning ``df`` would change it.

    Used on the 2017 corpus, which the handoff already cleaned with these rules.
    If this fires, the rules here have drifted from the ones that produced the
    2017 file, and the two datasets are no longer being treated alike -- which
    is the single thing this module exists to guarantee.
    """
    _, report = clean(df, feature_cols, label_col)
    if report.rows_dropped:
        raise AssertionError(
            f"cleaning removed {report.rows_dropped} row(s) from a corpus that "
            f"should already satisfy the rules "
            f"({report.rows_dropped_non_finite} non-finite, "
            f"{report.rows_dropped_duplicate} duplicate)"
        )
