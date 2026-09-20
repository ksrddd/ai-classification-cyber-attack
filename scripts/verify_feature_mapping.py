"""Prove the 77 feature pairs are the same measurement, not just the same position.

Why
---
``src.crossdataset.schema`` pairs CICIDS2017 and CSE-CIC-IDS2018 columns by
their position in CICFlowMeter's emission order, then states the pairs were
"read back to confirm every pair was semantically the same measurement" -- a
visual check, not evidence. Everything downstream inherits that assumption: a
mis-paired column is **indistinguishable from a dataset fingerprint** under
adversarial validation. It would rank first, get ablated, cross-dataset scores
would genuinely improve, and the write-up would report a dataset effect that is
really our own bug.

The two corpora were also produced by different CICFlowMeter versions, so a
unit change (microseconds vs milliseconds) or a fixed upstream bug (header
length) can make a correctly *named* pair semantically different anyway.

Three checks, weakest to strongest evidence
-------------------------------------------
**A. Invariants (decisive).** Algebraic identities that must hold *within a
row* of either dataset -- ``Fwd Packet Length Mean == Total Length of Fwd
Packets / Total Fwd Packets``, ``Packet Length Variance == Packet Length
Std**2``, the rate columns against ``Flow Duration``. Each is evaluated on each
dataset separately, so nothing depends on the two agreeing. An invariant that
holds on one side and fails on the other localises a mapping or unit error to
the columns it touches. This is the only check that can convict.

**B. Scale audit.** Per feature, on Benign rows only, compare robust location
and spread across datasets. An order-of-magnitude gap is a unit-change
candidate; a large gap in the fraction of zeros or of the ``-1`` sentinel is a
semantics-change candidate. Suggestive, never conclusive: a genuine dataset
difference looks the same.

**C. Pairing sanity, in three parts.**

* *C1 type compatibility* -- can convict. A flag column paired with a byte
  counter, or two columns whose value ranges do not overlap, is wrong however
  the traffic differed.
* *C2 correlation-profile sweep* -- does each column sit in the same
  relational position in both corpora? A feature is described by its Spearman
  correlations with every other column, and the declared partner must have the
  same neighbours. Rank correlation is invariant under any monotone transform,
  so the shifts check B measures do not move it.

  Two earlier versions compared **marginal distributions** and both failed,
  the second provably: it accused ``Flow Duration`` while check A's unit tests
  -- which cannot hold unless ``Flow Duration`` is the flow duration in
  microseconds on both sides -- passed at 0.00%. The fault was the evidence,
  not the metric. A distance between two marginals cannot separate "different
  column" from "same column, different traffic", and since the second effect
  is large here, such a sweep accuses exactly the features that shift most.

  Columns that are the same measurement under two names are resolved as a
  group; no test can tell them apart, and a twin scoring well is not a rival.
* *C3 value-set audit* -- for columns too near-constant for a stable rank
  correlation, the value sets side by side and no ranking at all.

Run::

    python scripts/verify_feature_mapping.py

Writes ``results/crossdataset/feature_mapping_audit/{report.md,findings.json}``
and exits non-zero if any check convicts, so it can gate a re-run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.crossdataset.labels import map_to_shared  # noqa: E402
from src.crossdataset.schema import canonical_features, to_canonical  # noqa: E402

logger = logging.getLogger("verify_feature_mapping")

LABEL_COL = "Label"
FEATURES: list[str] = list(canonical_features(keep_dst_port=True))

IDS2017_PARQUET = PROJECT_ROOT / "data" / "processed" / "cicids2017_clean.parquet"
IDS2018_PARQUET = PROJECT_ROOT / "data" / "ids2018" / "cicids2018_clean.parquet"
OUT_DIR = PROJECT_ROOT / "results" / "crossdataset" / "feature_mapping_audit"

#: A row violates an invariant when |lhs - rhs| exceeds this much of |rhs|.
#: 1% absorbs the float32 storage and CICFlowMeter's own rounding without
#: absorbing a unit change (which is 1000x) or a wrong column (unbounded).
RTOL = 0.01
ATOL = 1e-6

#: Violation rates below this read as "holds"; above the second as "fails".
#: The band between them is reported but never convicts on its own.
RATE_OK = 0.01
RATE_FAIL = 0.10

#: CICFlowMeter emits durations in microseconds. Invariants 19-22 test this
#: directly: if one dataset used a different unit its rate columns break.
MICROSECONDS_PER_SECOND = 1_000_000.0


# ----------------------------------------------------------------------
# Invariants
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Invariant:
    """One within-row identity, plus the columns a failure would implicate."""

    name: str
    columns: tuple[str, ...]
    note: str


def _ratio(num: pd.Series, den: pd.Series) -> tuple[pd.Series, pd.Series]:
    """num/den restricted to rows where den is non-zero."""
    ok = den != 0
    return num[ok] / den[ok], ok


def _close(lhs: pd.Series, rhs: pd.Series) -> float:
    """Fraction of rows where lhs and rhs disagree beyond tolerance."""
    if len(lhs) == 0:
        return float("nan")
    bad = ~np.isclose(lhs.to_numpy(), rhs.to_numpy(), rtol=RTOL, atol=ATOL, equal_nan=False)
    return float(bad.mean())


def _ordered(lo: pd.Series, mid: pd.Series, hi: pd.Series) -> float:
    """Fraction of rows where lo <= mid <= hi is violated beyond tolerance."""
    slack = ATOL + RTOL * hi.abs()
    bad = (mid < lo - slack) | (mid > hi + slack)
    return float(bad.to_numpy().mean()) if len(bad) else float("nan")


def _rate_violation(df: pd.DataFrame, count_cols: list[str], rate_col: str) -> float:
    """Check ``rate_col == sum(count_cols) / (Flow Duration / 1e6)``.

    This is the unit test for ``Flow Duration``. A dataset that stores
    milliseconds rather than microseconds fails here by a factor of 1000 while
    every column involved still looks individually plausible.
    """
    dur = df["Flow Duration"]
    ok = dur > 0
    if not ok.any():
        return float("nan")
    total = df.loc[ok, count_cols].sum(axis=1)
    expected = total / (dur[ok] / MICROSECONDS_PER_SECOND)
    return _close(df.loc[ok, rate_col], expected)


def evaluate_invariants(df: pd.DataFrame) -> dict[str, float]:
    """Violation rate per invariant on one dataset."""
    out: dict[str, float] = {}

    fwd_mean, ok = _ratio(df["Total Length of Fwd Packets"], df["Total Fwd Packets"])
    out["fwd_mean_eq_total_over_count"] = _close(df.loc[ok, "Fwd Packet Length Mean"], fwd_mean)

    bwd_mean, ok = _ratio(df["Total Length of Bwd Packets"], df["Total Backward Packets"])
    out["bwd_mean_eq_total_over_count"] = _close(df.loc[ok, "Bwd Packet Length Mean"], bwd_mean)

    out["variance_eq_std_squared"] = _close(
        df["Packet Length Variance"], df["Packet Length Std"] ** 2
    )

    # CICFlowMeter divides the packet-length sum by n+1 for Pkt Size Avg and by
    # n for Pkt Len Mean -- an off-by-one it has carried for years. Both corpora
    # inherit it, which is exactly what makes it a usable identity. The obvious
    # formula, total bytes over packet count, describes neither corpus: it
    # violates on 55% of 2017 rows and 21% of 2018, because TotLen Fwd/Bwd Pkts
    # count payload while the packet-length list counts headers too.
    total_pkts = df["Total Fwd Packets"] + df["Total Backward Packets"]
    ok = total_pkts > 1
    out["avg_packet_size_eq_mean_times_n_plus_one_over_n"] = _close(
        df.loc[ok, "Average Packet Size"],
        df.loc[ok, "Packet Length Mean"] * (total_pkts[ok] + 1) / total_pkts[ok],
    )

    # CICFlowMeter defines segment size average and packet length mean as the
    # same quantity; they are separate columns purely for historical reasons.
    out["fwd_seg_size_eq_fwd_pkt_len_mean"] = _close(
        df["Avg Fwd Segment Size"], df["Fwd Packet Length Mean"]
    )
    out["bwd_seg_size_eq_bwd_pkt_len_mean"] = _close(
        df["Avg Bwd Segment Size"], df["Bwd Packet Length Mean"]
    )

    out["fwd_pkt_len_min_mean_max_ordered"] = _ordered(
        df["Fwd Packet Length Min"], df["Fwd Packet Length Mean"], df["Fwd Packet Length Max"]
    )
    out["bwd_pkt_len_min_mean_max_ordered"] = _ordered(
        df["Bwd Packet Length Min"], df["Bwd Packet Length Mean"], df["Bwd Packet Length Max"]
    )
    out["pkt_len_min_mean_max_ordered"] = _ordered(
        df["Min Packet Length"], df["Packet Length Mean"], df["Max Packet Length"]
    )
    out["flow_iat_min_mean_max_ordered"] = _ordered(
        df["Flow IAT Min"], df["Flow IAT Mean"], df["Flow IAT Max"]
    )
    out["fwd_iat_min_mean_max_ordered"] = _ordered(
        df["Fwd IAT Min"], df["Fwd IAT Mean"], df["Fwd IAT Max"]
    )
    out["bwd_iat_min_mean_max_ordered"] = _ordered(
        df["Bwd IAT Min"], df["Bwd IAT Mean"], df["Bwd IAT Max"]
    )
    out["active_min_mean_max_ordered"] = _ordered(
        df["Active Min"], df["Active Mean"], df["Active Max"]
    )
    out["idle_min_mean_max_ordered"] = _ordered(df["Idle Min"], df["Idle Mean"], df["Idle Max"])

    out["subflow_fwd_pkts_eq_total"] = _close(
        df["Subflow Fwd Packets"], df["Total Fwd Packets"]
    )
    out["subflow_bwd_pkts_eq_total"] = _close(
        df["Subflow Bwd Packets"], df["Total Backward Packets"]
    )
    out["subflow_fwd_bytes_eq_total"] = _close(
        df["Subflow Fwd Bytes"], df["Total Length of Fwd Packets"]
    )
    out["subflow_bwd_bytes_eq_total"] = _close(
        df["Subflow Bwd Bytes"], df["Total Length of Bwd Packets"]
    )

    out["flow_pkts_per_s_matches_duration"] = _rate_violation(
        df, ["Total Fwd Packets", "Total Backward Packets"], "Flow Packets/s"
    )
    out["flow_byts_per_s_matches_duration"] = _rate_violation(
        df, ["Total Length of Fwd Packets", "Total Length of Bwd Packets"], "Flow Bytes/s"
    )
    out["fwd_pkts_per_s_matches_duration"] = _rate_violation(
        df, ["Total Fwd Packets"], "Fwd Packets/s"
    )
    out["bwd_pkts_per_s_matches_duration"] = _rate_violation(
        df, ["Total Backward Packets"], "Bwd Packets/s"
    )

    dur = df["Flow Duration"]
    slack = ATOL + RTOL * dur.abs()
    out["fwd_iat_total_within_duration"] = float(
        (df["Fwd IAT Total"] > dur + slack).to_numpy().mean()
    )
    out["bwd_iat_total_within_duration"] = float(
        (df["Bwd IAT Total"] > dur + slack).to_numpy().mean()
    )
    return out


INVARIANTS: dict[str, Invariant] = {
    inv.name: inv
    for inv in (
        Invariant("fwd_mean_eq_total_over_count",
                  ("Fwd Packet Length Mean", "Total Length of Fwd Packets", "Total Fwd Packets"),
                  "mean forward packet length is total bytes over packet count"),
        Invariant("bwd_mean_eq_total_over_count",
                  ("Bwd Packet Length Mean", "Total Length of Bwd Packets", "Total Backward Packets"),
                  "mean backward packet length is total bytes over packet count"),
        Invariant("variance_eq_std_squared",
                  ("Packet Length Variance", "Packet Length Std"),
                  "variance is the square of the standard deviation"),
        Invariant("avg_packet_size_eq_mean_times_n_plus_one_over_n",
                  ("Average Packet Size", "Packet Length Mean",
                   "Total Fwd Packets", "Total Backward Packets"),
                  "CICFlowMeter's n+1 off-by-one between Pkt Size Avg and Pkt Len Mean"),
        Invariant("fwd_seg_size_eq_fwd_pkt_len_mean",
                  ("Avg Fwd Segment Size", "Fwd Packet Length Mean"),
                  "CICFlowMeter emits these as the same quantity"),
        Invariant("bwd_seg_size_eq_bwd_pkt_len_mean",
                  ("Avg Bwd Segment Size", "Bwd Packet Length Mean"),
                  "CICFlowMeter emits these as the same quantity"),
        Invariant("fwd_pkt_len_min_mean_max_ordered",
                  ("Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Max"),
                  "min <= mean <= max"),
        Invariant("bwd_pkt_len_min_mean_max_ordered",
                  ("Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Max"),
                  "min <= mean <= max"),
        Invariant("pkt_len_min_mean_max_ordered",
                  ("Min Packet Length", "Packet Length Mean", "Max Packet Length"),
                  "min <= mean <= max"),
        Invariant("flow_iat_min_mean_max_ordered",
                  ("Flow IAT Min", "Flow IAT Mean", "Flow IAT Max"), "min <= mean <= max"),
        Invariant("fwd_iat_min_mean_max_ordered",
                  ("Fwd IAT Min", "Fwd IAT Mean", "Fwd IAT Max"), "min <= mean <= max"),
        Invariant("bwd_iat_min_mean_max_ordered",
                  ("Bwd IAT Min", "Bwd IAT Mean", "Bwd IAT Max"), "min <= mean <= max"),
        Invariant("active_min_mean_max_ordered",
                  ("Active Min", "Active Mean", "Active Max"), "min <= mean <= max"),
        Invariant("idle_min_mean_max_ordered",
                  ("Idle Min", "Idle Mean", "Idle Max"), "min <= mean <= max"),
        Invariant("subflow_fwd_pkts_eq_total",
                  ("Subflow Fwd Packets", "Total Fwd Packets"),
                  "single-subflow flows dominate, so these normally coincide"),
        Invariant("subflow_bwd_pkts_eq_total",
                  ("Subflow Bwd Packets", "Total Backward Packets"),
                  "single-subflow flows dominate, so these normally coincide"),
        Invariant("subflow_fwd_bytes_eq_total",
                  ("Subflow Fwd Bytes", "Total Length of Fwd Packets"),
                  "single-subflow flows dominate, so these normally coincide"),
        Invariant("subflow_bwd_bytes_eq_total",
                  ("Subflow Bwd Bytes", "Total Length of Bwd Packets"),
                  "single-subflow flows dominate, so these normally coincide"),
        Invariant("flow_pkts_per_s_matches_duration",
                  ("Flow Packets/s", "Flow Duration", "Total Fwd Packets", "Total Backward Packets"),
                  "UNIT TEST: rate implies Flow Duration is in microseconds"),
        Invariant("flow_byts_per_s_matches_duration",
                  ("Flow Bytes/s", "Flow Duration", "Total Length of Fwd Packets",
                   "Total Length of Bwd Packets"),
                  "UNIT TEST: rate implies Flow Duration is in microseconds"),
        Invariant("fwd_pkts_per_s_matches_duration",
                  ("Fwd Packets/s", "Flow Duration", "Total Fwd Packets"),
                  "UNIT TEST: rate implies Flow Duration is in microseconds"),
        Invariant("bwd_pkts_per_s_matches_duration",
                  ("Bwd Packets/s", "Flow Duration", "Total Backward Packets"),
                  "UNIT TEST: rate implies Flow Duration is in microseconds"),
        Invariant("fwd_iat_total_within_duration",
                  ("Fwd IAT Total", "Flow Duration"),
                  "inter-arrival total cannot exceed the flow it belongs to"),
        Invariant("bwd_iat_total_within_duration",
                  ("Bwd IAT Total", "Flow Duration"),
                  "inter-arrival total cannot exceed the flow it belongs to"),
    )
}


def verdict(rate_2017: float, rate_2018: float) -> str:
    """Classify one invariant from its two violation rates.

    ``ASYMMETRIC`` is the only verdict that convicts. Both sides failing means
    the identity does not describe CICFlowMeter's output at all -- our model of
    the tool is wrong, not the mapping -- and both sides holding means the
    columns it touches carry the same meaning in both corpora.
    """
    if np.isnan(rate_2017) or np.isnan(rate_2018):
        return "NOT_TESTABLE"
    holds_17, holds_18 = rate_2017 <= RATE_OK, rate_2018 <= RATE_OK
    fails_17, fails_18 = rate_2017 >= RATE_FAIL, rate_2018 >= RATE_FAIL
    if holds_17 and holds_18:
        return "PASS"
    if (holds_17 and fails_18) or (holds_18 and fails_17):
        return "ASYMMETRIC"
    if fails_17 and fails_18:
        return "BOTH_FAIL"
    return "BORDERLINE"


# ----------------------------------------------------------------------
# Scale audit and best-match sweep
# ----------------------------------------------------------------------
QUANTILES = np.linspace(0.01, 0.99, 25)

#: Percentile grid for the pairing sweep. Index i is the i-th percentile, so
#: ``grid[1]`` and ``grid[99]`` are p01/p99 and ``grid[50]`` the median.
PERCENTILE_GRID = np.linspace(0.0, 1.0, 101)

#: A column needs this many distinct values before its distribution shape can
#: identify it. Binary flags cannot: after any normalisation every 0/1 column
#: has the same shape, so ranking them against each other returns ties at
#: distance zero and an arbitrary winner. They are audited by value set instead.
MIN_UNIQUE_CONTINUOUS = 50

#: A column whose most common value covers more than this share has too little
#: variation for a stable rank correlation; its profile would be noise.
MAX_MODE_SHARE = 0.999

#: |Spearman| at or above this in *both* corpora means two columns are the same
#: measurement under two names. Check A proves several such pairs exactly
#: (``Subflow Fwd Bytes`` == ``Total Length of Fwd Packets`` on every row), and
#: they are unidentifiable from each other by any test.
DUPLICATE_RHO = 0.9999

#: The declared partner must beat the runner-up by this factor before the sweep
#: claims to have confirmed anything -- and the runner-up must beat the declared
#: partner by it before the sweep accuses. Everything between is reported as
#: ``AMBIGUOUS``: the check had no power on that feature, which is a fact about
#: the check, not evidence about the mapping.
MARGIN = 1.5

#: float32 holds integers exactly only below 2**24. Above it a genuinely
#: integral column reads as fractional, so integrality is reported but never
#: used to convict.
FLOAT32_EXACT_INT_MAX = 2 ** 24


def profile(series: pd.Series) -> dict[str, float]:
    values = series.to_numpy(dtype="float64")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"median": float("nan"), "p01": float("nan"), "p99": float("nan"),
                "iqr": float("nan"), "zero_frac": float("nan"),
                "neg_one_frac": float("nan"), "n_unique": 0}
    q01, q25, q50, q75, q99 = np.quantile(finite, [0.01, 0.25, 0.50, 0.75, 0.99])
    return {
        "median": float(q50),
        "p01": float(q01),
        "p99": float(q99),
        "iqr": float(q75 - q25),
        "zero_frac": float((finite == 0).mean()),
        "neg_one_frac": float((finite == -1).mean()),
        "n_unique": int(np.unique(finite).size),
    }


def scale_flag(a: dict[str, float], b: dict[str, float]) -> tuple[str, str]:
    """Compare two profiles of the same feature. Returns (flag, reason)."""
    if a["n_unique"] <= 1 and b["n_unique"] <= 1:
        return "CONSTANT_BOTH", "constant in both corpora; carries no signal either way"
    if (a["n_unique"] <= 1) != (b["n_unique"] <= 1):
        side = "2017" if a["n_unique"] <= 1 else "2018"
        return "CONSTANT_ONE_SIDE", f"constant in {side} only -- a perfect dataset fingerprint"

    if abs(a["zero_frac"] - b["zero_frac"]) > 0.5:
        return "ZERO_FRACTION", (
            f"zeros {a['zero_frac']:.1%} vs {b['zero_frac']:.1%} -- "
            "the column may not mean the same thing"
        )
    if abs(a["neg_one_frac"] - b["neg_one_frac"]) > 0.5:
        return "SENTINEL", (
            f"'-1' sentinel {a['neg_one_frac']:.1%} vs {b['neg_one_frac']:.1%}"
        )

    # Median is the natural scale proxy, but many flow features are majority
    # zero; p99 is what separates them there.
    for key in ("median", "p99"):
        x, y = abs(a[key]), abs(b[key])
        if x > 0 and y > 0:
            decades = abs(np.log10(y / x))
            if decades > 1:
                return "SCALE", f"{key} differs by {decades:.1f} decades ({x:.4g} vs {y:.4g})"
            break
    return "OK", ""


@dataclass(frozen=True)
class ColumnShape:
    """Everything the pairing sweep needs to know about one column."""

    quantiles: np.ndarray  # 101 percentiles
    spread: float  # p98 - p02, or full range where that degenerates
    n_unique: int
    integral: bool | None  # None when float32 cannot answer
    top_values: tuple[tuple[float, float], ...]  # (value, share), most common first

    @property
    def cardinality(self) -> str:
        if self.n_unique <= 1:
            return "constant"
        if self.n_unique <= 2:
            return "binary"
        if self.n_unique < MIN_UNIQUE_CONTINUOUS:
            return "low"
        return "high"

    @property
    def mode_share(self) -> float:
        """Share of rows holding the most common value; 1.0 when unknown."""
        return self.top_values[0][1] if self.top_values else 1.0

    @property
    def p01(self) -> float:
        return float(self.quantiles[1])

    @property
    def p99(self) -> float:
        return float(self.quantiles[99])


def describe_column(series: pd.Series) -> ColumnShape:
    values = series.to_numpy(dtype="float64")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return ColumnShape(np.zeros(len(PERCENTILE_GRID)), 0.0, 0, None, ())

    q = np.quantile(finite, PERCENTILE_GRID)
    # p98 - p02 rather than the IQR: flow features are routinely 80% zero, and
    # an IQR of exactly zero was what made the previous version fall back to a
    # different scale for some features and not others, leaving distances that
    # could not be compared across rows.
    spread = float(q[98] - q[2]) or float(q[-1] - q[0])

    uniq, counts = np.unique(finite, return_counts=True)
    order = np.argsort(-counts)[:3]
    top = tuple((float(uniq[i]), float(counts[i] / finite.size)) for i in order)

    integral: bool | None = None
    if np.abs(finite).max() < FLOAT32_EXACT_INT_MAX:
        integral = bool(np.all(finite == np.round(finite)))

    return ColumnShape(q, spread, int(uniq.size), integral, top)


def compatibility(a: ColumnShape, b: ColumnShape) -> tuple[str, str]:
    """Type-level check on one declared pair. This one can convict.

    A distribution distance cannot tell a mis-map from a genuine dataset shift
    -- check B already showed the IAT columns move by decades between corpora.
    Cardinality class and value range can: a flag column paired with a byte
    counter is wrong no matter how the traffic differed.
    """
    ca, cb = a.cardinality, b.cardinality
    if ca == "constant" and cb == "constant":
        return "CONSTANT_BOTH", "no values to compare"
    if {ca, cb} == {"binary", "high"}:
        return "INCOMPATIBLE", f"binary on one side ({ca}/{cb}), continuous on the other"
    if {ca, cb} == {"constant", "high"}:
        return "INCOMPATIBLE", f"constant on one side ({ca}/{cb}), continuous on the other"
    if a.p99 < b.p01 or b.p99 < a.p01:
        return "DISJOINT_RANGE", (
            f"[p01,p99] intervals do not overlap: "
            f"[{a.p01:.4g}, {a.p99:.4g}] vs [{b.p01:.4g}, {b.p99:.4g}]"
        )
    return "OK", ""


def spearman_profiles(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Rank-correlation matrix over ``features``.

    Spearman rather than Pearson because it is invariant under any monotone
    transform. That is the whole point: check B showed the IAT columns shift by
    three to five decades between corpora, and a correlation of ranks does not
    notice a shift, a rescale, or a unit change.
    """
    return df[features].corr(method="spearman")


def find_duplicate_groups(
    r_a: pd.DataFrame, r_b: pd.DataFrame, features: list[str], threshold: float
) -> dict[str, tuple[str, ...]]:
    """Group columns that are the same measurement under two names.

    Check A already proves several of these outright -- ``Subflow Fwd Bytes``
    equals ``Total Length of Fwd Packets`` on every row of both corpora. Such a
    pair has an identical correlation profile by construction, so no test can
    tell which of them a 2018 column corresponds to, and accusing either of
    being mis-paired would be nonsense. They are resolved as a group instead.
    """
    member_of: dict[str, tuple[str, ...]] = {}
    for f in features:
        if f in member_of:
            continue
        twins = tuple(
            g for g in features
            if abs(r_a.at[f, g]) >= threshold and abs(r_b.at[f, g]) >= threshold
        )
        for g in twins:
            member_of[g] = twins
    return member_of


def exact_duplicate_groups(
    a: pd.DataFrame, b: pd.DataFrame, features: list[str]
) -> list[tuple[str, ...]]:
    """Columns holding identical values in every row of *both* corpora.

    Needed alongside the correlation grouping, which can only see columns with
    enough variation to produce a stable rank correlation. The flag columns are
    filtered out of that sweep as near-constant, and two of them --
    ``Fwd URG Flags`` and ``CWE Flag Count`` -- are in fact the same column in
    both corpora. Exact equality settles that where a correlation cannot.

    Hashed rather than compared pairwise: 77 columns would otherwise be 2,926
    comparisons over ~380,000 rows each. Candidates sharing a hash in both
    corpora are then confirmed by an exact comparison, so a collision cannot
    create a false group.
    """
    def digest(frame: pd.DataFrame, col: str) -> str:
        return hashlib.blake2b(
            frame[col].to_numpy(dtype="float64").tobytes(), digest_size=16
        ).hexdigest()

    buckets: dict[tuple[str, str], list[str]] = {}
    for f in features:
        buckets.setdefault((digest(a, f), digest(b, f)), []).append(f)

    groups = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        head = members[0]
        confirmed = tuple(
            m for m in members
            if a[head].equals(a[m]) and b[head].equals(b[m])
        )
        if len(confirmed) > 1:
            groups.append(confirmed)
    return groups


def merge_groups(*sources: list[tuple[str, ...]]) -> list[list[str]]:
    """Connected components over every grouping rule, so a column that is a
    duplicate by one test and by another lands in a single group."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for source in sources:
        for group in source:
            for member in group[1:]:
                parent[find(group[0])] = find(member)

    out: dict[str, list[str]] = {}
    for member in parent:
        out.setdefault(find(member), []).append(member)
    return sorted([sorted(g) for g in out.values() if len(g) > 1])


def sweep_correlation(
    r_a: pd.DataFrame, r_b: pd.DataFrame, features: list[str],
    groups: dict[str, tuple[str, ...]],
) -> list[dict]:
    """Does each declared partner sit in the same relational position?

    For feature ``f`` the evidence is its vector of correlations with every
    other column. A correctly paired 2018 column has the same neighbours --
    ``Fwd Packet Length Mean`` correlates with the forward byte and packet
    totals in both corpora, whichever way the traffic differed. A mis-paired
    one has the wrong neighbours, and no amount of dataset shift can disguise
    that.

    This replaces a sweep over marginal distributions, which could not work:
    a distance between two marginals cannot separate "different column" from
    "same column, different traffic", and since the second effect is large here
    that sweep accused precisely the features check B had shown to shift most.
    """
    rows: list[dict] = []
    for f in features:
        twins = groups.get(f, (f,))
        # Correlations against f's own group are 1 by definition and carry no
        # information about which candidate is right.
        reference = [c for c in features if c not in twins]
        if len(reference) < 2:
            continue
        p_a = r_a.loc[f, reference]

        distance: dict[str, float] = {}
        for g in features:
            # Mask the candidate's self-correlation so every candidate is
            # scored on the same kind of evidence.
            cols = [c for c in reference if c != g]
            distance[g] = float((p_a[cols] - r_b.loc[g, cols]).abs().mean())

        declared = distance[f]
        # A twin scoring better than f is not a competing hypothesis; it is the
        # same measurement. Only genuine rivals may accuse.
        rivals = {g: d for g, d in distance.items() if g not in twins}
        if not rivals:
            continue
        runner_up, runner_up_d = min(rivals.items(), key=lambda kv: kv[1])
        rank = 1 + sum(1 for d in rivals.values() if d < declared)

        if rank == 1 and runner_up_d >= MARGIN * declared:
            verdict_c = "CONFIRMED"
        elif rank > 1 and declared >= MARGIN * runner_up_d:
            verdict_c = "SUSPECT"
        else:
            verdict_c = "AMBIGUOUS"

        rows.append({
            "feature": f,
            "verdict": verdict_c,
            "duplicate_group": list(twins) if len(twins) > 1 else None,
            "rank_of_declared_partner": rank,
            "distance_declared": round(declared, 4),
            "runner_up": runner_up,
            "distance_runner_up": round(runner_up_d, 4),
            "margin": round(runner_up_d / declared, 3) if declared > 0 else None,
        })

    order = {"SUSPECT": 0, "AMBIGUOUS": 1, "CONFIRMED": 2}
    rows.sort(key=lambda r: (order[r["verdict"]], r["rank_of_declared_partner"]))
    return rows


def discrete_audit(
    shapes_a: dict[str, ColumnShape], shapes_b: dict[str, ColumnShape], names: list[str]
) -> list[dict]:
    """Value-set comparison for columns too coarse to rank.

    No distance and no ranking: for a 0/1 flag the only meaningful description
    is how often it fires, and several unrelated flags fire at similar rates.
    Reporting the numbers side by side lets a reader see a flag paired with a
    counter; pretending to rank them does not.
    """
    return [
        {
            "feature": f,
            "cardinality_ids2017": shapes_a[f].cardinality,
            "cardinality_ids2018": shapes_b[f].cardinality,
            "n_unique_ids2017": shapes_a[f].n_unique,
            "n_unique_ids2018": shapes_b[f].n_unique,
            "integral_ids2017": shapes_a[f].integral,
            "integral_ids2018": shapes_b[f].integral,
            "top_values_ids2017": [[v, round(s, 4)] for v, s in shapes_a[f].top_values],
            "top_values_ids2018": [[v, round(s, 4)] for v, s in shapes_b[f].top_values],
        }
        for f in names
    ]


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_canonical(path: Path, dataset: str, n_rows: int, seed: int) -> pd.DataFrame:
    """Stream a subsample of one corpus into the canonical schema.

    Streamed rather than loaded whole: the 2018 corpus is 11.4M rows and this
    audit needs a few hundred thousand. Sampling a fixed fraction of every
    batch spreads the draw across all capture days instead of taking a prefix.
    """
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    fraction = min(1.0, n_rows / total)
    rng = np.random.default_rng(seed)
    logger.info("%s: %s rows in %s, sampling %.2f%%", dataset, f"{total:,}", path.name, 100 * fraction)

    parts: list[pd.DataFrame] = []
    kept = 0
    for batch in pf.iter_batches(batch_size=250_000):
        chunk = batch.to_pandas()
        if fraction < 1.0:
            take = rng.random(len(chunk)) < fraction
            chunk = chunk.loc[take]
        if chunk.empty:
            continue
        frame = to_canonical(chunk, dataset, keep_dst_port=True).astype("float64")
        frame[LABEL_COL] = map_to_shared(chunk[LABEL_COL].astype(str), dataset).to_numpy()
        parts.append(frame)
        kept += len(frame)
        if kept >= n_rows:
            break

    out = pd.concat(parts, ignore_index=True).head(n_rows)
    out = out.loc[out[LABEL_COL].notna()].reset_index(drop=True)
    logger.info("%s: %s rows loaded, %d classes", dataset, f"{len(out):,}", out[LABEL_COL].nunique())
    return out


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def _fmt_rate(x: float) -> str:
    return "n/a" if np.isnan(x) else f"{x:.2%}"


def build_report(inv_rows: list[dict], scale_rows: list[dict], compat_rows: list[dict],
                 sweep_rows: list[dict], duplicate_groups: list[list[str]],
                 discrete_rows: list[dict], n17: int, n18: int) -> str:
    convictions = [r for r in inv_rows if r["verdict"] == "ASYMMETRIC"]
    incompatible = [r for r in compat_rows if r["flag"] in ("INCOMPATIBLE", "DISJOINT_RANGE")]
    suspects = [r for r in sweep_rows if r["verdict"] == "SUSPECT"]
    confirmed = [r for r in sweep_rows if r["verdict"] == "CONFIRMED"]
    ambiguous = [r for r in sweep_rows if r["verdict"] == "AMBIGUOUS"]
    scale_flags = [r for r in scale_rows if r["flag"] not in ("OK", "CONSTANT_BOTH")]

    lines = [
        "# Feature mapping audit: CICIDS2017 <-> CSE-CIC-IDS2018",
        "",
        f"Rows sampled: {n17:,} (2017) / {n18:,} (2018). Features checked: {len(FEATURES)}.",
        "",
        "## Verdict",
        "",
    ]
    if convictions or incompatible:
        lines += [
            f"**FAIL** -- {len(convictions)} asymmetric invariant(s) and {len(incompatible)} "
            "type-incompatible pair(s). The columns they touch do not carry the same meaning "
            "in both datasets. Fix the mapping before running the cross-dataset protocol; "
            "adversarial validation would otherwise rank these as dataset fingerprints.",
            "",
        ]
    else:
        lines += [
            "**PASS** -- no invariant separates the two corpora and no declared pair is "
            "type-incompatible. Every algebraic identity that holds on 2017 also holds on "
            "2018, so the paired columns are the same measurement. Scale findings below are "
            "candidates for adversarial validation, not mapping errors.",
            "",
        ]

    lines += ["## A. Invariants (decisive)", "",
              "| invariant | 2017 | 2018 | verdict | columns |", "|---|---|---|---|---|"]
    for r in inv_rows:
        lines.append(
            f"| `{r['invariant']}` | {_fmt_rate(r['rate_ids2017'])} | "
            f"{_fmt_rate(r['rate_ids2018'])} | **{r['verdict']}** | "
            f"{', '.join(f'`{c}`' for c in r['columns'])} |"
        )
    lines += ["",
              "`ASYMMETRIC` convicts. `BOTH_FAIL` means the identity does not describe "
              "CICFlowMeter's output in either corpus, which indicts the check rather than "
              "the data.", ""]

    lines += ["## B. Scale audit (Benign rows only)", ""]
    if scale_flags:
        lines += ["| feature | flag | detail |", "|---|---|---|"]
        for r in scale_flags:
            lines.append(f"| `{r['feature']}` | {r['flag']} | {r['reason']} |")
        lines += ["",
                  "These are dataset differences, not necessarily mapping errors -- feed "
                  "them to adversarial validation.", ""]
    else:
        lines += ["No feature differs by more than an order of magnitude.", ""]

    lines += ["## C1. Pair type compatibility (can convict)", ""]
    if incompatible:
        lines += ["| feature | flag | detail |", "|---|---|---|"]
        for r in incompatible:
            lines.append(f"| `{r['feature']}` | {r['flag']} | {r['reason']} |")
        lines += ["", "A flag column paired with a counter, or two columns whose value "
                      "ranges do not overlap, is wrong however the traffic differed.", ""]
    else:
        lines += ["Every declared pair is compatible: same cardinality class, overlapping "
                  "value ranges.", ""]

    lines += [
        "## C2. Correlation-profile sweep", "",
        f"Testable: {len(sweep_rows)} of {len(FEATURES)} features. "
        f"**Power of this check: {len(confirmed)} CONFIRMED, {len(ambiguous)} AMBIGUOUS, "
        f"{len(suspects)} SUSPECT.**", "",
        "Each feature is described by its Spearman correlations with every other column, "
        "and the declared 2018 partner must have the same neighbours. Rank correlation is "
        "invariant under any monotone transform, so the three-to-five-decade shifts that "
        "check B found do not move these numbers. `margin` is the runner-up's distance "
        f"over the declared partner's: above {MARGIN} confirmed, below 1/{MARGIN} accused, "
        "between the two the check cannot decide and says so.", "",
    ]
    if duplicate_groups:
        lines += [
            "### Duplicate measurements", "",
            "These columns are the same measurement under more than one name -- check A "
            "proves several of them exactly. Members of a group cannot be told apart by any "
            "test, so they are resolved together rather than accused of matching each "
            "other. Found two ways: identical Spearman rank (catches monotone pairs such "
            "as a standard deviation and its square) and identical values in every row "
            "(catches the near-constant flag columns, which carry too little variation for "
            "a stable correlation).", "",
            f"**{len(duplicate_groups)} group(s): the {len(FEATURES)} columns hold "
            f"{len(FEATURES) - sum(len(g) - 1 for g in duplicate_groups)} distinct "
            "measurements. This matters downstream -- permutation importance splits credit "
            "between duplicates, understating both, so adversarial validation and the "
            "ablation must treat each group as one feature.**", "",
        ]
        for group in duplicate_groups:
            lines.append(f"* {' == '.join(f'`{c}`' for c in group)}")
        lines += [""]

    if suspects:
        lines += ["### SUSPECT", "",
                  "| feature | rank | d(declared) | better match | d | margin |",
                  "|---|---|---|---|---|---|"]
        for r in suspects:
            lines.append(
                f"| `{r['feature']}` | {r['rank_of_declared_partner']} | "
                f"{r['distance_declared']} | `{r['runner_up']}` | "
                f"{r['distance_runner_up']} | {r['margin']} |"
            )
        lines += ["", "Read every row here against check A before acting on it: an "
                      "invariant that passes at 0.00% on both corpora and touches this "
                      "column outranks a correlation distance.", ""]
    else:
        lines += ["No feature is beaten by a genuine rival at the required margin.", ""]

    if confirmed:
        lines += ["### CONFIRMED", "",
                  "| feature | d(declared) | runner-up | margin |", "|---|---|---|---|"]
        for r in confirmed:
            lines.append(
                f"| `{r['feature']}` | {r['distance_declared']} | `{r['runner_up']}` | "
                f"{r['margin']} |"
            )
        lines += [""]
    if ambiguous:
        lines += [
            f"### AMBIGUOUS ({len(ambiguous)})", "",
            "The declared partner is neither clearly best nor clearly beaten -- read as "
            "\"this check cannot decide\", not as a finding.", "",
            ", ".join(f"`{r['feature']}` (rank {r['rank_of_declared_partner']})"
                      for r in ambiguous), "",
        ]

    lines += ["## C3. Discrete features (audited, not ranked)", "",
              "Too coarse for a distribution distance -- every 0/1 column has the same shape "
              "after normalisation, so ranking them returns ties and an arbitrary winner. "
              "The value sets are shown instead.", "",
              "| feature | cardinality 17/18 | n_unique 17/18 | top value 2017 | top value 2018 |",
              "|---|---|---|---|---|"]
    for r in discrete_rows:
        t17 = r["top_values_ids2017"]
        t18 = r["top_values_ids2018"]
        s17 = f"{t17[0][0]:g} ({t17[0][1]:.1%})" if t17 else "n/a"
        s18 = f"{t18[0][0]:g} ({t18[0][1]:.1%})" if t18 else "n/a"
        lines.append(
            f"| `{r['feature']}` | {r['cardinality_ids2017']}/{r['cardinality_ids2018']} | "
            f"{r['n_unique_ids2017']}/{r['n_unique_ids2018']} | {s17} | {s18} |"
        )
    lines += [""]
    return "\n".join(lines)


# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids2017", type=Path, default=IDS2017_PARQUET)
    ap.add_argument("--ids2018", type=Path, default=IDS2018_PARQUET)
    ap.add_argument("--rows", type=int, default=400_000,
                    help="rows to sample per corpus")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])

    for path in (args.ids2017, args.ids2018):
        if not path.is_file():
            logger.error("missing corpus: %s", path)
            return 2

    a = load_canonical(args.ids2017, "ids2017", args.rows, args.seed)
    b = load_canonical(args.ids2018, "ids2018", args.rows, args.seed)

    logger.info("Check A: invariants")
    rates_a, rates_b = evaluate_invariants(a), evaluate_invariants(b)
    inv_rows = [
        {
            "invariant": name,
            "rate_ids2017": rates_a[name],
            "rate_ids2018": rates_b[name],
            "verdict": verdict(rates_a[name], rates_b[name]),
            "columns": list(INVARIANTS[name].columns),
            "note": INVARIANTS[name].note,
        }
        for name in rates_a
    ]
    order = {"ASYMMETRIC": 0, "BORDERLINE": 1, "BOTH_FAIL": 2, "NOT_TESTABLE": 3, "PASS": 4}
    inv_rows.sort(key=lambda r: order[r["verdict"]])
    for r in inv_rows:
        logger.info("  %-12s %-42s 2017=%s 2018=%s", r["verdict"], r["invariant"],
                    _fmt_rate(r["rate_ids2017"]), _fmt_rate(r["rate_ids2018"]))

    logger.info("Check B: scale audit on Benign rows")
    ben_a = a.loc[a[LABEL_COL] == "Benign"]
    ben_b = b.loc[b[LABEL_COL] == "Benign"]
    scale_rows = []
    for f in FEATURES:
        pa, pb = profile(ben_a[f]), profile(ben_b[f])
        flag, reason = scale_flag(pa, pb)
        scale_rows.append({"feature": f, "flag": flag, "reason": reason,
                           "ids2017": pa, "ids2018": pb})
        if flag not in ("OK", "CONSTANT_BOTH"):
            logger.info("  %-18s %-22s %s", flag, f, reason)

    logger.info("Check C: pairing sanity")
    shapes_a = {f: describe_column(ben_a[f]) for f in FEATURES}
    shapes_b = {f: describe_column(ben_b[f]) for f in FEATURES}

    compat_rows = []
    for f in FEATURES:
        flag, reason = compatibility(shapes_a[f], shapes_b[f])
        compat_rows.append({"feature": f, "flag": flag, "reason": reason})
        if flag in ("INCOMPATIBLE", "DISJOINT_RANGE"):
            logger.error("  C1 %-16s %-24s %s", flag, f, reason)

    # Correlation profiles need variation on both sides; a column that is
    # constant, or 99.9% one value, produces a rank correlation of noise.
    correlatable = [
        f for f in FEATURES
        if shapes_a[f].n_unique > 1 and shapes_b[f].n_unique > 1
        and shapes_a[f].mode_share < MAX_MODE_SHARE
        and shapes_b[f].mode_share < MAX_MODE_SHARE
    ]
    uncorrelatable = [f for f in FEATURES if f not in correlatable]
    logger.info("  C2 computing Spearman profiles over %d feature(s)", len(correlatable))
    r_a = spearman_profiles(ben_a, correlatable)
    r_b = spearman_profiles(ben_b, correlatable)

    groups = find_duplicate_groups(r_a, r_b, correlatable, DUPLICATE_RHO)
    rho_groups = sorted({v for v in groups.values() if len(v) > 1})

    # Correlation can only speak for columns with enough variation. Exact value
    # equality covers the rest -- including the flag columns C2 filters out,
    # two of which are the same column in both corpora.
    exact_groups = exact_duplicate_groups(ben_a, ben_b, FEATURES)
    all_groups = merge_groups(rho_groups, exact_groups)
    rho_only = {tuple(g) for g in merge_groups(rho_groups)}
    for group in all_groups:
        how = "rho=1" if tuple(group) in rho_only else "identical values"
        logger.info("  duplicate group (%s): %s", how, " == ".join(group))
    logger.info("  %d group(s) -> %d distinct measurement(s) among %d columns",
                len(all_groups),
                len(FEATURES) - sum(len(g) - 1 for g in all_groups), len(FEATURES))

    sweep_rows = sweep_correlation(r_a, r_b, correlatable, groups)
    for r in sweep_rows:
        if r["verdict"] == "SUSPECT":
            logger.error("  C2 SUSPECT   %-24s rank %d, %s is %.2fx closer",
                         r["feature"], r["rank_of_declared_partner"], r["runner_up"],
                         r["distance_declared"] / max(r["distance_runner_up"], 1e-12))
    logger.info("  C2 power: %d confirmed / %d ambiguous / %d suspect (of %d testable)",
                sum(r["verdict"] == "CONFIRMED" for r in sweep_rows),
                sum(r["verdict"] == "AMBIGUOUS" for r in sweep_rows),
                sum(r["verdict"] == "SUSPECT" for r in sweep_rows), len(sweep_rows))

    discrete_rows = discrete_audit(shapes_a, shapes_b, uncorrelatable)
    logger.info("  C3 %d feature(s) audited by value set, not ranked", len(discrete_rows))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    findings = {
        "n_rows_ids2017": len(a),
        "n_rows_ids2018": len(b),
        "n_features": len(FEATURES),
        "invariants": inv_rows,
        "scale_audit": scale_rows,
        "pair_compatibility": compat_rows,
        "correlation_sweep": sweep_rows,
        "duplicate_groups": all_groups,
        "n_distinct_measurements": len(FEATURES) - sum(len(g) - 1 for g in all_groups),
        "discrete_audit": discrete_rows,
        "invariant_definitions": {k: asdict(v) for k, v in INVARIANTS.items()},
    }
    (args.out_dir / "findings.json").write_text(
        json.dumps(findings, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    report = build_report(inv_rows, scale_rows, compat_rows, sweep_rows,
                          all_groups, discrete_rows,
                          len(a), len(b))
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    logger.info("Wrote %s and findings.json", args.out_dir / "report.md")

    convictions = [r for r in inv_rows if r["verdict"] == "ASYMMETRIC"]
    incompatible = [r for r in compat_rows if r["flag"] in ("INCOMPATIBLE", "DISJOINT_RANGE")]
    if convictions or incompatible:
        logger.error("FAIL: %d asymmetric invariant(s), %d incompatible pair(s) -- "
                     "mapping must be fixed", len(convictions), len(incompatible))
        return 1
    logger.info("PASS: mapping holds under every check that can convict")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
