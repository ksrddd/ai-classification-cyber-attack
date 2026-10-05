"""Audit what the CICIDS2017 "chronological" split is ordered by, and how far that stands in for time.

The 2017 export used here (*MachineLearningCVE*) has no ``Timestamp`` column, so
the split orders flows by ``_row_index``: each flow's position in its raw CSV,
recorded before any row is cleaned away. Everything downstream -- the 70/30
bundle ``cicids2017_temporal_v1`` and the cross-dataset partition --
inherits whatever that ordering is and is not. This script measures it.

Six questions, each answered from the raw files and the results already on disk
(nothing is trained):

1. **Is ``_row_index`` really the raw row position?** Every cleaned row is
   compared with the raw row it claims to be, on three columns.
2. **Does row order follow the published CIC attack timetable?** The check the
   pipeline runs before training looks at one pair of attacks. The documentation
   cites a seven-pair check that is defined in ``src/data/temporal_split.py`` but
   never called; it is run here, and then tightened from "median positions" to
   "any row out of order", over every scheduled pair.
3. **Is each attack one block, or does it trail?** A label's rows are split into
   clusters wherever the gap between neighbours exceeds a fixed share of the
   file, and the rows left over after the first cluster are described.
4. **What does the 70/30 cut put on each side?** Per ``(capture, class)``: where
   the cut falls, and which attack sub-types land in train and which in test.
5. **Is there a global train/test time boundary?** Per capture: the share of
   training rows that come after the first test row.
6. **What does this do to the cross-dataset numbers?** The cross-dataset partition of
   the 300,000-row 2017 corpus, and how much of the random-vs-chronological gap
   in the 2017 ceiling each class carries.

Writes to ``results/split_audit/cicids2017/``. ``--figure-dir`` also draws where
each class's train and test rows sit along every capture.

Run::

    python scripts/audit_chronological_split_2017.py
    python scripts/audit_chronological_split_2017.py --figure-dir docs/experiments/ids2017/report/figures
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.constants import RANDOM_STATE  # noqa: E402
from src.data.temporal_split import (  # noqa: E402
    CICIDS2017_SCHEDULE,
    load_temporal_manifest,
    normalize_raw_label,
    temporal_source_split,
    validate_capture_chronology,
    validate_raw_label_chronology,
    verify_split_against_manifest,
)

logger = logging.getLogger("audit_chronological_split_2017")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_PARQUET = PROJECT_ROOT / "data" / "processed" / "cicids2017_clean.parquet"
MANIFEST = PROJECT_ROOT / "configs" / "splits" / "cicids2017_temporal_70_30.json"
CROSS_RESULTS = PROJECT_ROOT / "results" / "crossdataset"
OUT_DIR = PROJECT_ROOT / "results" / "split_audit" / "cicids2017"

LABEL, SOURCE, ORDER = "Label", "source_file", "_row_index"
BENIGN = "BENIGN"

#: Columns compared between the raw CSV and the cleaned cache. They are integer
#: valued in the raw files, so a float32 round trip must reproduce them exactly.
CHECK_COLUMNS = ("Destination Port", "Flow Duration", "Total Fwd Packets")

#: Extra columns read only to describe what a cluster of rows looks like: the
#: client's initial TCP window is a property of the sending host's stack, so rows
#: that share it, the destination port and the order of magnitude of their
#: durations plausibly come from the same traffic.
SIGNATURE_COLUMNS = ("Init_Win_bytes_forward", "Init_Win_bytes_backward")

#: Two rows of one label further apart than this share of the capture start a new
#: cluster. 5% is far above the spacing inside any attack block (a few rows) and
#: below every between-block gap that matters here, so the cut falls in a genuine
#: gap rather than mid-block. It is a reporting threshold, not a statistical one.
GAP_FRACTION = 0.05

#: Runs whose per-class tables feed the "what does this do to the ceiling" step.
CROSS_RUNS = ("protocol_v1", "protocol_v2")

#: Per-class F1 and the published inflation are each rounded to four decimals, so
#: the two can differ by a few units of the last place and no more.
INFLATION_TOLERANCE = 1e-3


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------
def position_fraction(positions: np.ndarray, n_rows: int) -> np.ndarray:
    """Row positions as a share of the capture, 0 for the first flow and 1 for the last.

    The denominator is ``n_rows - 1``, as in ``temporal_split._position_stats``,
    so numbers here read against the validation report.
    """
    return np.asarray(positions, dtype=float) / float(max(n_rows - 1, 1))


def split_into_clusters(
    positions: np.ndarray, n_rows: int, gap_fraction: float = GAP_FRACTION
) -> list[np.ndarray]:
    """Split one label's row positions wherever two neighbours are far apart.

    "Far" is more than ``gap_fraction`` of the capture's rows. Clusters come back
    in file order; an empty input gives no clusters.
    """
    pos = np.sort(np.asarray(positions))
    if pos.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(pos) > gap_fraction * n_rows) + 1
    return np.split(pos, breaks)


def duration_summary(durations_s: np.ndarray) -> dict[str, float]:
    """Median, 95th percentile and maximum of flow durations in seconds."""
    durations_s = np.asarray(durations_s, dtype=float)
    return {
        "median": float(np.median(durations_s)),
        "p95": float(np.quantile(durations_s, 0.95)),
        "max": float(durations_s.max()),
    }


def cluster_signature(frame: pd.DataFrame, rows: np.ndarray) -> dict[str, float | int]:
    """The dominant destination port and the TCP windows of one cluster's rows."""
    chunk = frame.iloc[rows]
    ports = chunk["Destination Port"].value_counts()
    return {
        "top_destination_port": int(ports.index[0]),
        "top_destination_port_share": float(ports.iloc[0] / len(chunk)),
        "init_win_forward_median": float(chunk["Init_Win_bytes_forward"].median()),
        "init_win_backward_median": float(chunk["Init_Win_bytes_backward"].median()),
    }


def position_clusters(
    positions: np.ndarray,
    n_rows: int,
    gap_fraction: float = GAP_FRACTION,
    durations_s: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Each cluster's row count and where its first and last row sit (shares of the capture).

    With ``durations_s`` (flow duration in seconds, indexed by row position) each
    cluster also carries the spread of its flows' durations. CICFlowMeter closes
    a flow after at most two minutes, so rows written hours after an attack
    window cannot be explained by connections that merely stayed open.
    """
    out: list[dict[str, Any]] = []
    for chunk in split_into_clusters(positions, n_rows, gap_fraction):
        entry: dict[str, Any] = {
            "n": int(chunk.size),
            "first": float(position_fraction(chunk[0], n_rows)),
            "last": float(position_fraction(chunk[-1], n_rows)),
        }
        if durations_s is not None:
            entry["flow_duration_s"] = duration_summary(durations_s[chunk])
        out.append(entry)
    return out


def pair_order_stats(earlier: np.ndarray, later: np.ndarray) -> dict[str, float | int]:
    """How badly two labels' rows disagree with "``earlier`` comes first".

    ``earlier_rows_after_first_later``
        rows of the earlier label that sit after the first row of the later one.
    ``later_rows_before_last_earlier``
        rows of the later label that sit before the last row of the earlier one.
    ``inverted_pairs_share``
        over every (earlier row, later row) pair, the share in the wrong order.
        0 means the two labels are perfectly separated in the published order.

    The median-position check that the pipeline relies on returns "ordered" for
    any pair whose medians are in order, however many rows are not.
    """
    a = np.sort(np.asarray(earlier))
    b = np.sort(np.asarray(later))
    if a.size == 0 or b.size == 0:
        raise ValueError("both labels need at least one row")
    after_first = int(a.size - np.searchsorted(a, b[0], side="right"))
    before_last = int(np.searchsorted(b, a[-1], side="left"))
    inverted = int((a.size - np.searchsorted(a, b, side="right")).sum())
    return {
        "earlier_rows_after_first_later": after_first,
        "later_rows_before_last_earlier": before_last,
        "inverted_pairs_share": inverted / (a.size * b.size),
    }


def chance_of_passing_by_luck(window_counts: Iterable[int]) -> float:
    """Probability that medians in random order pass every capture's order check.

    A capture with ``k`` scheduled sub-labels passes with probability ``1 / k!``
    if its labels sit in no particular order, and the captures are independent.
    This is the power of the median check against a *shuffled* file -- it says
    nothing about a file that is ordered coarsely and shuffled within a block.
    """
    return float(np.prod([1.0 / math.factorial(int(k)) for k in window_counts if int(k) >= 2]))


def minutes(hhmm: str) -> int:
    hours, mins = hhmm.split(":")
    return int(hours) * 60 + int(mins)


# ---------------------------------------------------------------------------
# Reading the raw captures
# ---------------------------------------------------------------------------
def read_capture(raw_dir: Path, source: str) -> pd.DataFrame:
    """One raw capture: normalised sub-label plus the columns the audit reads."""
    wanted = {LABEL, *CHECK_COLUMNS, *SIGNATURE_COLUMNS}
    frame = pd.read_csv(
        raw_dir / source,
        encoding="latin-1",
        low_memory=False,
        usecols=lambda column: column.strip() in wanted,
    )
    frame = frame.rename(columns=str.strip)
    frame[LABEL] = frame[LABEL].map(normalize_raw_label)
    return frame


def label_positions(labels: pd.Series) -> dict[str, np.ndarray]:
    """Row positions of every label, in file order."""
    values = labels.to_numpy()
    return {str(name): np.flatnonzero(values == name) for name in pd.unique(values)}


# ---------------------------------------------------------------------------
# 1-3. Raw-file facts
# ---------------------------------------------------------------------------
def describe_captures(raw: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Per capture and sub-label: size, where it sits, and its clusters."""
    out: dict[str, Any] = {}
    for source, frame in raw.items():
        n_rows = len(frame)
        durations = frame["Flow Duration"].to_numpy(dtype=float) / 1e6
        labels: dict[str, Any] = {}
        for name, pos in label_positions(frame[LABEL]).items():
            frac = position_fraction(pos, n_rows)
            clusters = position_clusters(pos, n_rows, durations_s=durations)
            for entry, rows in zip(clusters, split_into_clusters(pos, n_rows), strict=True):
                entry["signature"] = cluster_signature(frame, rows)
            labels[name] = {
                "n": int(pos.size),
                "min": float(frac.min()),
                "p05": float(np.quantile(frac, 0.05)),
                "median": float(np.median(frac)),
                "p95": float(np.quantile(frac, 0.95)),
                "max": float(frac.max()),
                "clusters": clusters,
            }
        out[source] = {
            "n_rows": int(n_rows),
            "max_flow_duration_s": float(durations.max()),
            "labels": labels,
        }
    return out


def published_order(raw: dict[str, pd.DataFrame], clean: pd.DataFrame) -> dict[str, Any]:
    """The pipeline's own check, the seven-pair check, and the any-row version."""
    in_pipeline = validate_capture_chronology(clean, strict=False)
    seven_pair = validate_raw_label_chronology(RAW_DIR, strict=False)

    pairs: list[dict[str, Any]] = []
    for source, windows in sorted(CICIDS2017_SCHEDULE.items()):
        if len(windows) < 2:
            continue
        positions = label_positions(raw[source][LABEL])
        n_rows = len(raw[source])
        durations = raw[source]["Flow Duration"].to_numpy(dtype=float) / 1e6
        for i, early in enumerate(windows):
            for j in range(i + 1, len(windows)):
                late = windows[j]
                a, b = positions[early.raw_label], positions[late.raw_label]
                stats = pair_order_stats(a, b)
                entry: dict[str, Any] = {
                    "source_file": source,
                    "earlier": early.raw_label,
                    "later": late.raw_label,
                    "adjacent": j == i + 1,
                    "published_gap_minutes": minutes(late.start) - minutes(early.end),
                    "earlier_rows": int(a.size),
                    "later_rows": int(b.size),
                    "median_earlier": float(position_fraction(np.median(a), n_rows)),
                    "median_later": float(position_fraction(np.median(b), n_rows)),
                    "median_ordered": bool(np.median(a) < np.median(b)),
                    **stats,
                    "earlier_rows_after_first_later_share":
                        stats["earlier_rows_after_first_later"] / int(a.size),
                }
                if stats["earlier_rows_after_first_later"]:
                    # Those rows contradict the published order. Were they flows
                    # that merely stayed open, their durations would say so.
                    out_of_order = a[a > b.min()]
                    entry["out_of_order_rows_flow_duration_s"] = duration_summary(
                        durations[out_of_order])
                pairs.append(entry)

    return {
        "in_pipeline_check": {
            "valid": bool(in_pipeline["valid"]),
            "pairs_checked": int(in_pipeline["checked_pairs"]),
        },
        "seven_pair_check": {
            "valid": bool(seven_pair["valid"]),
            "pairs_checked": int(seven_pair["checked_pairs"]),
            "violations": list(seven_pair["violations"]),
        },
        "chance_of_passing_by_luck": chance_of_passing_by_luck(
            len(windows) for windows in CICIDS2017_SCHEDULE.values()
        ),
        "pairs": pairs,
    }


# ---------------------------------------------------------------------------
# 1. Round trip
# ---------------------------------------------------------------------------
def row_index_integrity(raw: dict[str, pd.DataFrame], clean: pd.DataFrame) -> dict[str, Any]:
    """Compare every cleaned row with the raw row at its ``_row_index``."""
    compared = mismatched = out_of_range = duplicated = 0
    for source, group in clean.groupby(SOURCE, sort=True):
        frame = raw[str(source)]
        idx = group[ORDER].to_numpy()
        duplicated += int(idx.size - np.unique(idx).size)
        in_range = (idx >= 0) & (idx < len(frame))
        out_of_range += int((~in_range).sum())
        idx = idx[in_range]
        group = group.loc[in_range]
        bad = np.zeros(idx.size, dtype=bool)
        for column in CHECK_COLUMNS:
            expected = frame[column].to_numpy()[idx].astype(np.float32)
            bad |= expected != group[column].to_numpy(dtype=np.float32)
        compared += int(idx.size)
        mismatched += int(bad.sum())
    return {
        "columns": list(CHECK_COLUMNS),
        "rows_compared": compared,
        "rows_mismatched": mismatched,
        "rows_out_of_range": out_of_range,
        "duplicate_row_indexes": duplicated,
        "raw_rows_total": int(sum(len(f) for f in raw.values())),
    }


# ---------------------------------------------------------------------------
# 4-5. The 70/30 split
# ---------------------------------------------------------------------------
def attach_raw_labels(clean: pd.DataFrame, raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Add ``raw_label``, ``pos`` (share of the capture) and ``cluster`` to the cleaned rows.

    ``cluster`` numbers the cluster of the row's own sub-label from 1 in file
    order, as in ``position_clusters``, so a trailing stream can be told from
    the block before it.
    """
    out = clean.copy()
    raw_label = np.empty(len(out), dtype=object)
    pos = np.empty(len(out), dtype=float)
    cluster = np.empty(len(out), dtype=int)
    for source, frame in raw.items():
        n_rows = len(frame)
        cluster_of_row = np.zeros(n_rows, dtype=int)
        for rows in label_positions(frame[LABEL]).values():
            for number, chunk in enumerate(split_into_clusters(rows, n_rows), start=1):
                cluster_of_row[chunk] = number
        mask = (out[SOURCE] == source).to_numpy()
        idx = out[ORDER].to_numpy()[mask]
        raw_label[mask] = frame[LABEL].to_numpy()[idx]
        pos[mask] = position_fraction(idx, n_rows)
        cluster[mask] = cluster_of_row[idx]
    out["raw_label"] = raw_label
    out["pos"] = pos
    out["cluster"] = cluster
    return out


def split_tables(
    frame: pd.DataFrame, raw_sizes: dict[str, int]
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reproduce the manifest split and describe each side.

    Returns the summary, the per-group boundaries, the sub-type composition and
    the train and test frames themselves.
    """
    manifest = load_temporal_manifest(MANIFEST)
    train, _calibration, test = temporal_source_split(
        frame,
        label_column=LABEL,
        order_column=ORDER,
        test_size=float(manifest["test_size"]),
        min_test_per_class=int(manifest["min_test_per_class"]),
    )
    verification = verify_split_against_manifest(train, test, manifest, label_column=LABEL)

    groups: list[dict[str, Any]] = []
    for (source, cls), group in frame.groupby([SOURCE, LABEL], sort=True):
        tr = train[(train[SOURCE] == source) & (train[LABEL] == cls)]
        te = test[(test[SOURCE] == source) & (test[LABEL] == cls)]
        windows = [w for w in CICIDS2017_SCHEDULE[str(source)] if w.canonical_label == cls]
        groups.append(
            {
                "source_file": str(source),
                "class": str(cls),
                "n_rows_in_capture": raw_sizes[str(source)],
                "n": int(len(group)),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "first_pos": round(float(group["pos"].min()), 4),
                "train_last_pos": round(float(tr["pos"].max()), 4) if len(tr) else None,
                "test_first_pos": round(float(te["pos"].min()), 4) if len(te) else None,
                "last_pos": round(float(group["pos"].max()), 4),
                "published": "; ".join(f"{w.raw_label} {w.start}-{w.end}" for w in windows),
            }
        )

    composition = (
        pd.concat(
            [
                train.groupby([SOURCE, LABEL, "raw_label", "cluster"]).size().rename("n_train"),
                test.groupby([SOURCE, LABEL, "raw_label", "cluster"]).size().rename("n_test"),
            ],
            axis=1,
        )
        .fillna(0)
        .astype(int)
        .reset_index()
        .rename(columns={SOURCE: "source_file", LABEL: "class"})
    )

    # Is there one time boundary between train and test? Per capture: the share
    # of training rows that come after the capture's first test row.
    overlap: dict[str, Any] = {}
    first_test = test.groupby(SOURCE)[ORDER].min()
    for source, tr in train.groupby(SOURCE):
        if source not in first_test.index:
            overlap[str(source)] = {"first_test_pos": None, "train_rows_after_first_test": 0,
                                    "share_of_train_rows": 0.0}
            continue
        cut = int(first_test[source])
        after = int((tr[ORDER] > cut).sum())
        overlap[str(source)] = {
            "first_test_pos": round(float(position_fraction(cut, raw_sizes[str(source)])), 4),
            "train_rows": int(len(tr)),
            "train_rows_after_first_test": after,
            "share_of_train_rows": after / int(len(tr)),
        }

    summary = {
        "manifest_reproduced": bool(verification["valid"]),
        "mismatches": verification["mismatches"],
        "train_rows": int(verification["actual_train_rows"]),
        "test_rows": int(verification["actual_test_rows"]),
        "overlap_per_capture": overlap,
    }
    return summary, pd.DataFrame(groups), composition, train, test


# ---------------------------------------------------------------------------
# 6. The cross-dataset partition and the ceiling
# ---------------------------------------------------------------------------
def crossdataset_composition(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Sub-label make-up of each side of the 2017 corpus the cross-dataset runs use."""
    from src.crossdataset.loaders import load_ids2017
    from src.crossdataset.splits import chronological_split, random_split

    corpus = load_ids2017()
    group = corpus.group.to_numpy()
    key = corpus.order_key.to_numpy()
    sub = np.empty(len(group), dtype=object)
    for source, frame in raw.items():
        mask = group == source
        sub[mask] = frame[LABEL].to_numpy()[key[mask]]
    base = pd.DataFrame({"class": corpus.y.to_numpy(), "raw_label": sub})

    rows: list[pd.DataFrame] = []
    for mode, part in (
        ("chronological", chronological_split(corpus.y, corpus.order_key, corpus.group)),
        ("random", random_split(corpus.y, seed=RANDOM_STATE)),
    ):
        sides = {"n_train": part.train, "n_test": part.test}
        counts = pd.concat(
            [
                base.iloc[idx].groupby(["class", "raw_label"]).size().rename(name)
                for name, idx in sides.items()
            ],
            axis=1,
        ).fillna(0).astype(int).reset_index()
        counts.insert(0, "mode", mode)
        rows.append(counts)
    return pd.concat(rows, ignore_index=True)


def inflation_by_class(run: str) -> pd.DataFrame | None:
    """How much each class adds to (random - chronological) of the 2017 ceiling.

    Macro-F1 is the mean of the per-class F1 over the classes present in the
    test split, so the gap between two split modes is the mean of the per-class
    gaps and each class's share of it is exact.
    """
    path = CROSS_RESULTS / run / "summary_per_class.csv"
    if not path.is_file():
        return None
    table = pd.read_csv(path)
    f1_cols = [c for c in table.columns if c.startswith("f1__")]
    within = table[(table["train"] == "ids2017") & (table["test"] == "ids2017")]
    chrono = within[within["mode"] == "chronological"].set_index("model")[f1_cols]
    rand = within[within["mode"] == "random"].set_index("model")[f1_cols]
    contribution = (rand - chrono) / len(f1_cols)
    contribution.columns = [c[len("f1__"):] for c in f1_cols]

    # The decomposition is exact, so it must add back up to the inflation the
    # run already published; a mismatch means a class went missing from one side.
    leakage = pd.read_csv(CROSS_RESULTS / run / "summary_leakage.csv")
    published = leakage[leakage["test"] == "ids2017"].set_index("model")["inflation"]
    drift = (contribution.sum(axis=1) - published.reindex(contribution.index)).abs().max()
    if drift > INFLATION_TOLERANCE:
        raise ValueError(
            f"{run}: per-class contributions differ from summary_leakage.csv by {drift:.4f}"
        )
    long = contribution.reset_index().melt(
        id_vars="model", var_name="class", value_name="inflation_contribution"
    )
    long.insert(0, "run", run)
    long["chronological_f1"] = [
        float(chrono.loc[m, f"f1__{c}"]) for m, c in zip(long["model"], long["class"], strict=True)
    ]
    long["random_f1"] = [
        float(rand.loc[m, f"f1__{c}"]) for m, c in zip(long["model"], long["class"], strict=True)
    ]
    return long.round(4)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
TRAIN_COLOR, TEST_COLOR = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 (validated)
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#dcdbd6"
FIGURE_BINS = 100
BAR_HEIGHT = 0.55  # tallest bar of a class row, in row units (one row = 1.0)
#: One figure per group of captures, so each fits on a page. The first holds the
#: three captures whose class holds several sub-attacks, the second the rest.
FIGURES: dict[str, tuple[str, ...]] = {
    "row_order_multi_attack.png": (
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    ),
    "row_order_single_attack.png": (
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    ),
}


def _block_labels(
    source: str, cls: str, captures: dict[str, Any]
) -> list[tuple[float, str]]:
    """What to print above a class's bars: its sub-labels, one tag per sizeable cluster.

    A sub-label that only repeats the class name says nothing, so it is left out.
    One that sits in more than one sizeable cluster is tagged once per cluster
    with the cluster's row count, which is how a trailing stream shows up.
    """
    tags: list[tuple[float, str]] = []
    for name, info in captures[source]["labels"].items():
        if name in (BENIGN, cls) or _canonical(source, name) != cls:
            continue
        sizeable = [c for c in info["clusters"] if c["n"] >= 0.05 * info["n"]]
        short = name.replace("Web Attack ", "").replace("DoS ", "")
        for cluster in sizeable:
            text = f"{short} ({cluster['n']:,})" if len(sizeable) > 1 else short
            tags.append(((cluster["first"] + cluster["last"]) / 2, text))
    return sorted(tags)


def draw_figure(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    captures: dict[str, Any],
    sources: tuple[str, ...],
    path: Path,
) -> None:
    """Where each class's train and test rows sit along the given captures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layout: list[tuple[str, list[str]]] = []
    for source in sources:
        classes = sorted(frame.loc[frame[SOURCE] == source, LABEL].unique(),
                         key=lambda c: (c != BENIGN, c))
        layout.append((source, classes))

    heights = [0.5 + 0.78 * len(classes) for _, classes in layout]
    fig, axes = plt.subplots(
        len(layout), 1, figsize=(8.0, sum(heights) + 0.9), height_ratios=heights,
        facecolor="white",
    )
    edges = np.linspace(0.0, 1.0, FIGURE_BINS + 1)
    width = 1.0 / FIGURE_BINS
    mid = edges[:-1] + width / 2

    for ax, (source, classes) in zip(axes, layout, strict=True):
        n_rows = captures[source]["n_rows"]
        ax.set_facecolor("white")
        for row, cls in enumerate(classes):
            base = len(classes) - 1 - row
            tr = train.loc[(train[SOURCE] == source) & (train[LABEL] == cls), "pos"].to_numpy()
            te = test.loc[(test[SOURCE] == source) & (test[LABEL] == cls), "pos"].to_numpy()
            h_tr, _ = np.histogram(tr, bins=edges)
            h_te, _ = np.histogram(te, bins=edges)
            peak = max(int((h_tr + h_te).max()), 1)
            ax.bar(mid, h_tr / peak * BAR_HEIGHT, width=width, bottom=base, color=TRAIN_COLOR,
                   linewidth=0, align="center")
            ax.bar(mid, h_te / peak * BAR_HEIGHT, width=width, bottom=base, color=TEST_COLOR,
                   linewidth=0, align="center")
            ax.hlines(base, 0, 1, color=GRID, linewidth=0.6, zorder=0)
            # two staggered levels keep neighbouring tags from running together
            for k, (x, text) in enumerate(_block_labels(source, cls, captures)):
                ax.text(x, base + BAR_HEIGHT + 0.04 + 0.17 * (k % 2), text, fontsize=6.3,
                        color=INK_SOFT, ha="center", va="bottom")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.08, len(classes) - 1 + 1.0)
        ax.set_yticks([len(classes) - 1 - r + BAR_HEIGHT / 2 for r in range(len(classes))])
        ax.set_yticklabels(classes, fontsize=7, color=INK)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=6.5, colors=INK_SOFT, length=2)
        ax.set_xticks(np.linspace(0, 1, 11))
        ax.set_title(f"{source.replace('.pcap_ISCX.csv', '')}  ({n_rows:,} flows)",
                     fontsize=8, color=INK, loc="left", pad=3)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

    axes[-1].set_xlabel("position in the raw CSV   (0 = first flow written, 1 = last)",
                        fontsize=7.5, color=INK_SOFT)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=TRAIN_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=TEST_COLOR),
    ]
    fig.legend(handles, ["train: earliest 70% of the class in this capture",
                         "test: latest 30%"], loc="upper right", fontsize=7.5, frameon=False,
               ncol=2, bbox_to_anchor=(0.99, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    logger.info("figure written to %s", path)


def _canonical(source: str, raw_label: str) -> str | None:
    for window in CICIDS2017_SCHEDULE[source]:
        if window.raw_label == raw_label:
            return window.canonical_label
    return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--figure-dir", type=Path, default=None,
                    help="also draw the per-class train/test position figures into this folder")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    for needed in (CLEAN_PARQUET, MANIFEST):
        if not needed.is_file():
            raise SystemExit(f"missing {needed}")

    logger.info("reading %d raw captures", len(CICIDS2017_SCHEDULE))
    raw = {source: read_capture(RAW_DIR, source) for source in sorted(CICIDS2017_SCHEDULE)}
    raw_sizes = {source: len(frame) for source, frame in raw.items()}

    clean = pd.read_parquet(CLEAN_PARQUET, columns=[LABEL, SOURCE, ORDER, *CHECK_COLUMNS])
    clean[LABEL] = clean[LABEL].astype(str)
    clean[SOURCE] = clean[SOURCE].astype(str)
    logger.info("cleaned cache: %s rows", f"{len(clean):,}")

    captures = describe_captures(raw)
    order = published_order(raw, clean)
    integrity = row_index_integrity(raw, clean)

    frame = attach_raw_labels(clean[[LABEL, SOURCE, ORDER]], raw)
    split, groups, composition, train, test = split_tables(frame, raw_sizes)
    cross = crossdataset_composition(raw)
    inflation = [t for t in (inflation_by_class(run) for run in CROSS_RUNS) if t is not None]
    inflation_table = pd.concat(inflation, ignore_index=True) if inflation else pd.DataFrame()

    report = {
        "order_basis": "csv_row_index",
        "gap_fraction": GAP_FRACTION,
        "captures": captures,
        "published_order": order,
        "row_index_integrity": integrity,
        "manifest_split": split,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "row_order_audit.json").write_text(
        json.dumps(to_jsonable(report), indent=2) + "\n", encoding="utf-8")
    groups.to_csv(args.out_dir / "group_boundaries.csv", index=False, lineterminator="\n")
    composition.to_csv(args.out_dir / "subtype_composition.csv", index=False, lineterminator="\n")
    cross.to_csv(args.out_dir / "crossdataset_partition_2017.csv", index=False, lineterminator="\n")
    inflation_table.to_csv(args.out_dir / "inflation_by_class.csv", index=False, lineterminator="\n")

    logger.info("in-pipeline check: %s pair(s), valid=%s",
                order["in_pipeline_check"]["pairs_checked"], order["in_pipeline_check"]["valid"])
    logger.info("seven-pair check: %s pair(s), valid=%s",
                order["seven_pair_check"]["pairs_checked"], order["seven_pair_check"]["valid"])
    logger.info("round trip: %s rows compared, %s mismatched",
                f"{integrity['rows_compared']:,}", integrity["rows_mismatched"])
    logger.info("manifest reproduced: %s (%s train / %s test)", split["manifest_reproduced"],
                f"{split['train_rows']:,}", f"{split['test_rows']:,}")

    if args.figure_dir is not None:
        for name, sources in FIGURES.items():
            draw_figure(frame, train, test, captures, sources, args.figure_dir / name)
    logger.info("wrote %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
