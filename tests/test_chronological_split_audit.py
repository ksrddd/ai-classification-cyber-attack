"""Tests for the helpers behind ``scripts/audit_chronological_split_2017.py``.

The audit exists because two things were easy to believe and are worth checking:
that ``_row_index`` is the raw row position, and that "the medians are in the
published order" means the labels are in the published order. The second does
not follow from the first test being green, so the helpers that replace it are
tested against exactly the case it misses.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_chronological_split_2017.py"


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("audit_chronological_split_2017", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Positions and clusters
# ---------------------------------------------------------------------------
def test_position_fraction_runs_from_zero_to_one(audit) -> None:
    """Same denominator as ``temporal_split._position_stats``: first row 0, last row 1."""
    frac = audit.position_fraction(np.array([0, 5, 10]), 11)

    assert frac.tolist() == [0.0, 0.5, 1.0]


def test_clusters_split_only_at_gaps_wider_than_the_threshold(audit) -> None:
    n_rows = 1000  # a 5% gap is 50 rows
    tight = np.array([10, 11, 12, 40, 41])  # largest step 28 rows: one cluster
    split = np.array([10, 11, 12, 100, 101])  # a 88-row gap: two clusters

    assert [c.size for c in audit.split_into_clusters(tight, n_rows)] == [5]
    assert [c.size for c in audit.split_into_clusters(split, n_rows)] == [3, 2]
    assert audit.split_into_clusters(np.array([], dtype=int), n_rows) == []


def test_clusters_are_reported_in_file_order_whatever_the_input_order(audit) -> None:
    out = audit.position_clusters(np.array([900, 10, 11, 901]), 1000)

    assert [c["n"] for c in out] == [2, 2]
    assert out[0]["first"] < out[1]["first"]


def test_cluster_durations_describe_only_that_cluster(audit) -> None:
    durations = np.zeros(1000)
    durations[900:] = 90.0  # the late rows are the long ones

    out = audit.position_clusters(np.array([1, 2, 900, 901]), 1000, durations_s=durations)

    assert out[0]["flow_duration_s"]["max"] == 0.0
    assert out[1]["flow_duration_s"]["median"] == 90.0


# ---------------------------------------------------------------------------
# The order check
# ---------------------------------------------------------------------------
def test_separated_labels_have_no_inversions(audit) -> None:
    out = audit.pair_order_stats(np.arange(0, 100), np.arange(100, 200))

    assert out["earlier_rows_after_first_later"] == 0
    assert out["later_rows_before_last_earlier"] == 0
    assert out["inverted_pairs_share"] == 0.0


def test_fully_reversed_labels_are_all_inverted(audit) -> None:
    out = audit.pair_order_stats(np.arange(100, 200), np.arange(0, 100))

    assert out["inverted_pairs_share"] == 1.0
    assert out["earlier_rows_after_first_later"] == 100


def test_median_check_passes_a_pair_that_the_any_row_check_fails(audit) -> None:
    """The GoldenEye/Heartbleed case: most of the earlier label comes first, a stream after.

    Medians are in order, so the pipeline's check says "ordered". Yet a large
    share of the earlier label sits after the later label has already started.
    """
    earlier = np.concatenate([np.arange(0, 600), np.arange(1000, 1400)])  # 60% / 40%
    later = np.array([800, 801, 802])

    stats = audit.pair_order_stats(earlier, later)

    assert np.median(earlier) < np.median(later)
    assert stats["earlier_rows_after_first_later"] == 400
    assert stats["later_rows_before_last_earlier"] == 3
    assert 0.0 < stats["inverted_pairs_share"] < 1.0


def test_pair_stats_need_rows_on_both_sides(audit) -> None:
    with pytest.raises(ValueError, match="at least one row"):
        audit.pair_order_stats(np.array([], dtype=int), np.array([1]))


def test_chance_of_passing_by_luck_is_the_product_of_inverse_factorials(audit) -> None:
    # Tuesday 2 windows, Wednesday 5, Thursday 3; single-window captures cannot be
    # ordered and add nothing.
    assert audit.chance_of_passing_by_luck([0, 1, 1, 2, 3, 5]) == pytest.approx(1 / 1440)
    assert audit.chance_of_passing_by_luck([1, 1]) == 1.0


# ---------------------------------------------------------------------------
# Reading and integrity
# ---------------------------------------------------------------------------
def test_read_capture_copes_with_padded_headers_and_the_web_attack_dash(
    audit, tmp_path: Path
) -> None:
    """Raw headers carry leading spaces and one label carries byte 0x96."""
    csv = tmp_path / "toy.csv"
    header = (" Destination Port, Flow Duration, Total Fwd Packets, Other,"
              "Init_Win_bytes_forward, Init_Win_bytes_backward, Label\n")
    rows = (
        "80,5,1,0,29200,235,BENIGN\n"
        "21,7,2,0,8192,-1,Web Attack \x96 XSS\n"
    )
    csv.write_bytes((header + rows).encode("latin-1"))

    frame = audit.read_capture(tmp_path, "toy.csv")

    assert list(frame.columns) == [
        "Destination Port", "Flow Duration", "Total Fwd Packets",
        "Init_Win_bytes_forward", "Init_Win_bytes_backward", "Label",
    ]
    assert frame["Label"].tolist() == ["BENIGN", "Web Attack XSS"]


def _raw_frame(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Destination Port": np.arange(n) % 65535,
            "Flow Duration": np.arange(n) * 3,
            "Total Fwd Packets": np.arange(n) % 7,
            "Label": ["BENIGN"] * n,
        }
    )


def _clean_from(raw: pd.DataFrame, keep: list[int], source: str = "a.csv") -> pd.DataFrame:
    out = raw.iloc[keep][list(raw.columns[:3])].astype(np.float32).reset_index(drop=True)
    out["_row_index"] = keep
    out["source_file"] = source
    out["Label"] = "BENIGN"
    return out


def test_integrity_counts_zero_mismatches_for_a_faithful_cache(audit) -> None:
    raw = {"a.csv": _raw_frame(50)}
    clean = _clean_from(raw["a.csv"], [0, 3, 4, 20, 49])

    out = audit.row_index_integrity(raw, clean)

    assert out["rows_compared"] == 5
    assert out["rows_mismatched"] == 0
    assert out["duplicate_row_indexes"] == 0
    assert out["raw_rows_total"] == 50


def test_integrity_catches_a_row_index_that_points_at_the_wrong_row(audit) -> None:
    """If cleaning had shifted positions, the ordering key would silently be wrong."""
    raw = {"a.csv": _raw_frame(50)}
    clean = _clean_from(raw["a.csv"], [0, 3, 4, 20, 49])
    clean.loc[3, "_row_index"] = 21  # claims to be row 21 but holds row 20's values

    out = audit.row_index_integrity(raw, clean)

    assert out["rows_mismatched"] == 1


def test_integrity_flags_out_of_range_and_repeated_indexes(audit) -> None:
    raw = {"a.csv": _raw_frame(10)}
    clean = _clean_from(raw["a.csv"], [0, 1, 2])
    clean.loc[2, "_row_index"] = 1  # repeats row 1
    clean.loc[0, "_row_index"] = 99  # past the end of the capture

    out = audit.row_index_integrity(raw, clean)

    assert out["rows_out_of_range"] == 1
    assert out["duplicate_row_indexes"] == 1


def test_attach_raw_labels_looks_up_by_position(audit) -> None:
    raw = {"a.csv": _raw_frame(11).assign(Label=list("ABCDEFGHIJK"))}
    clean = pd.DataFrame(
        {"Label": ["x", "y"], "source_file": ["a.csv", "a.csv"], "_row_index": [0, 10]}
    )

    out = audit.attach_raw_labels(clean, raw)

    assert out["raw_label"].tolist() == ["A", "K"]
    assert out["pos"].tolist() == [0.0, 1.0]
    assert out["cluster"].tolist() == [1, 1]


def test_attach_raw_labels_numbers_the_clusters_of_a_trailing_label(audit) -> None:
    """Rows 0-2 and 90-92 are one label in two clusters 87 rows (8.7% of the file) apart."""
    labels = ["X"] * 3 + ["B"] * 87 + ["X"] * 3 + ["B"] * 907
    raw = {"a.csv": _raw_frame(1000).assign(Label=labels)}
    clean = pd.DataFrame(
        {"Label": ["x"] * 3, "source_file": ["a.csv"] * 3, "_row_index": [1, 90, 92]}
    )

    out = audit.attach_raw_labels(clean, raw)

    assert out["cluster"].tolist() == [1, 2, 2]


def test_cluster_signature_reports_the_dominant_port_and_windows(audit) -> None:
    frame = pd.DataFrame(
        {
            "Destination Port": [80, 80, 80, 443],
            "Init_Win_bytes_forward": [29200, 29200, 29200, 8192],
            "Init_Win_bytes_backward": [235, 235, 235, -1],
        }
    )

    sig = audit.cluster_signature(frame, np.array([0, 1, 2, 3]))

    assert sig["top_destination_port"] == 80
    assert sig["top_destination_port_share"] == 0.75
    assert sig["init_win_forward_median"] == 29200
    assert sig["init_win_backward_median"] == 235


# ---------------------------------------------------------------------------
# Splitting the ceiling gap by class
# ---------------------------------------------------------------------------
def _write_cross_run(root: Path, *, inflation: float) -> None:
    run = root / "protocol_x"
    run.mkdir(parents=True)
    rows = []
    for mode, f1s in (("chronological", (0.9, 0.0)), ("random", (0.9, 1.0))):
        rows.append({"mode": mode, "train": "ids2017", "test": "ids2017", "model": "m",
                     "f1__Benign": f1s[0], "f1__Bot": f1s[1]})
    pd.DataFrame(rows).to_csv(run / "summary_per_class.csv", index=False)
    pd.DataFrame(
        [{"test": "ids2017", "model": "m", "chronological": 0.45, "random": 0.95,
          "inflation": inflation}]
    ).to_csv(run / "summary_leakage.csv", index=False)


def test_inflation_is_split_exactly_across_classes(
    audit, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Macro-F1 is a mean over classes, so its gap is the mean of the per-class gaps."""
    _write_cross_run(tmp_path, inflation=0.5)
    monkeypatch.setattr(audit, "CROSS_RESULTS", tmp_path)

    table = audit.inflation_by_class("protocol_x")

    by_class = table.set_index("class")["inflation_contribution"]
    assert by_class["Bot"] == pytest.approx(0.5)
    assert by_class["Benign"] == pytest.approx(0.0)
    assert by_class.sum() == pytest.approx(0.5)


def test_inflation_refuses_a_table_that_does_not_add_up(
    audit, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_cross_run(tmp_path, inflation=0.25)  # the per-class gaps sum to 0.5
    monkeypatch.setattr(audit, "CROSS_RESULTS", tmp_path)

    with pytest.raises(ValueError, match="differ from summary_leakage"):
        audit.inflation_by_class("protocol_x")


def test_inflation_returns_none_for_a_run_that_is_not_there(audit, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(audit, "CROSS_RESULTS", tmp_path)

    assert audit.inflation_by_class("missing") is None
