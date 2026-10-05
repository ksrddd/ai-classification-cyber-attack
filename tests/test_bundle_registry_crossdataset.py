"""Tests for the cross-dataset layout in the bundle registry.

A cross-dataset run sits in the same picker as the training runs but answers a
different question. A training bundle asks "which of these seven models is
best" and has one metric row per model; a cross-dataset run asks "how much of
what a model learned on one corpus survives the move to the other" and its unit
of result is a (train corpus, test corpus) pair. There is no champion to name.

The contract these tests pin is what keeps the shared picker honest:

* the run is discoverable, so it can be selected;
* ``models`` stays empty, so a page written for the other two layouts renders
  nothing rather than a ranking built from whichever of the four combinations
  sorted first;
* the diagonal of the matrix is the ceiling the off-diagonal was measured
  against, not a second reading of it from another file;
* superseded runs are not offered at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.bundles.registry import (
    _discover,
    describe_bundle,
    list_bundles,
    load_bundle,
)

GAP_HEADER = "mode,train,test,model,floor,ceiling,transfer,gap,recovered_pct,f1_macro_all_sd\n"
PER_CLASS_HEADER = "mode,train,test,model,f1__Benign,f1__Bot\n"


def _write_run(root: Path, name: str = "protocol_v1") -> Path:
    """A minimal but complete cross-dataset run: two corpora, one model."""
    path = root / "crossdataset" / name
    path.mkdir(parents=True)

    (path / "results.csv").write_text(
        "mode,seed,train,test,transfer,model,f1_macro_all\n"
        "chronological,42,ids2017,ids2017,False,stacking,0.70\n",
        encoding="utf-8",
    )
    (path / "summary_gap.csv").write_text(
        GAP_HEADER
        # 2018 -> 2017 recovers most of the range; 2017 -> 2018 recovers almost none.
        + "chronological,ids2018,ids2017,stacking,0.1294,0.7329,0.5756,-0.1574,73.9,0.0352\n"
        + "chronological,ids2017,ids2018,stacking,0.132,0.7039,0.1529,-0.5511,3.6,0.0008\n",
        encoding="utf-8",
    )
    (path / "summary_per_class.csv").write_text(
        PER_CLASS_HEADER
        + "chronological,ids2018,ids2017,stacking,0.9507,0.9358\n"
        + "chronological,ids2017,ids2017,stacking,0.9743,0.2650\n",
        encoding="utf-8",
    )
    (path / "summary_leakage.csv").write_text(
        "test,model,chronological,random,inflation\n" "ids2017,stacking,0.7329,0.973,0.2401\n",
        encoding="utf-8",
    )
    (path / "summary_support.csv").write_text(
        "mode,test,n__Benign,n__Bot\nchronological,ids2017,74455,584\n",
        encoding="utf-8",
    )
    (path / "corpora.json").write_text(
        json.dumps(
            {
                "ids2017": {"n_rows": 300000, "n_features": 77, "order_basis": "_row_index"},
                "ids2018": {"n_rows": 262701, "n_features": 77, "order_basis": "Timestamp"},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_run_is_discovered_one_level_down(tmp_path: Path) -> None:
    """Cross-dataset runs nest like the 2018 ones, so both depths must be scanned."""
    _write_run(tmp_path)

    assert "crossdataset/protocol_v1" in _discover(tmp_path)


def test_summary_tables_are_required(tmp_path: Path) -> None:
    """A run that died before analyze_crossdataset.py must not be offered.

    ``results.csv`` alone is what an interrupted run leaves behind. Listing it
    would put an entry in the picker whose every table is empty, which reads as
    "this run found nothing" rather than "this run was never summarised".
    """
    path = tmp_path / "crossdataset" / "half_finished"
    path.mkdir(parents=True)
    (path / "results.csv").write_text("mode,seed\nchronological,42\n", encoding="utf-8")

    assert _discover(tmp_path) == {}


@pytest.mark.parametrize("depth", ["nested", "top"])
def test_underscore_runs_are_never_offered(tmp_path: Path, depth: str) -> None:
    """Superseded runs are skipped at both depths.

    The underscore prefix marks a bundle kept on disk only long enough to diff
    against its successor -- ``_v1_unseeded_models`` reported five seeds that
    were really one run repeated, because the loop seed never reached the
    estimators. Those numbers have been retracted in writing, and a picker
    entry is an invitation to quote one anyway.
    """
    _write_run(tmp_path, name="_v1_unseeded_models" if depth == "nested" else "protocol_v1")
    if depth == "top":
        (tmp_path / "crossdataset").rename(tmp_path / "_crossdataset_old")

    assert _discover(tmp_path) == {}


def test_split_shape_is_read_from_the_run_and_defaults_to_the_legacy_one(tmp_path: Path) -> None:
    """A run without split.json predates the calibration part being dropped."""
    path = _write_run(tmp_path)
    legacy = load_bundle("crossdataset/protocol_v1", tmp_path).run["split_protocol"]
    (path / "split.json").write_text(json.dumps({"shape": "70/30"}), encoding="utf-8")

    current = load_bundle("crossdataset/protocol_v1", tmp_path).run["split_protocol"]

    assert legacy.startswith("60/10/30")
    assert current.startswith("70/30")


def test_models_stays_empty(tmp_path: Path) -> None:
    """The central contract: no per-model metric row is invented.

    Any single score per model would have to be drawn from one of the four
    combinations, and every choice is wrong -- the ceiling flatters, the
    transfer damns, and picking either silently answers a question the run did
    not ask. An empty mapping is recoverable; a plausible wrong number is not.
    """
    _write_run(tmp_path)

    bundle = load_bundle("crossdataset/protocol_v1", tmp_path)

    assert bundle.layout == "crossdataset"
    assert bundle.models == {}
    # The count still reports what ran, so the picker does not read as empty.
    assert bundle.to_summary()["n_models"] == 1


def test_matrix_has_all_four_cells_with_the_diagonal_derived(tmp_path: Path) -> None:
    """Two transfer rows yield four cells, and the diagonal is the ceiling.

    ``summary_gap.csv`` records only the off-diagonal, but each row carries the
    ceiling for the corpus it was tested on -- and that ceiling *is* the
    diagonal cell. Deriving it from there keeps the printed diagonal and the
    recovered-percentage beside it measured against the same number.
    """
    _write_run(tmp_path)

    cells = load_bundle("crossdataset/protocol_v1", tmp_path).crossdataset["cells"]
    by_pair = {(c["train"], c["test"]): c for c in cells}

    assert len(cells) == 4
    assert by_pair[("ids2018", "ids2017")]["f1_macro"] == pytest.approx(0.5756)
    assert by_pair[("ids2017", "ids2018")]["f1_macro"] == pytest.approx(0.1529)
    # Diagonals, taken from the ceiling of the row tested on that corpus.
    assert by_pair[("ids2017", "ids2017")]["f1_macro"] == pytest.approx(0.7329)
    assert by_pair[("ids2018", "ids2018")]["f1_macro"] == pytest.approx(0.7039)


def test_diagonal_cells_report_no_recovered_percentage(tmp_path: Path) -> None:
    """100% there is a tautology, and it would outrank every measurement."""
    _write_run(tmp_path)

    cells = load_bundle("crossdataset/protocol_v1", tmp_path).crossdataset["cells"]

    for cell in cells:
        if cell["self"]:
            assert cell["recovered_pct"] is None
            assert cell["train"] == cell["test"]
        else:
            assert cell["recovered_pct"] is not None


def test_asymmetry_survives_the_round_trip(tmp_path: Path) -> None:
    """The finding itself, pinned: one direction transfers, the other does not.

    This is the number the write-up leads with. If a refactor ever pairs the
    recovered percentages with the wrong direction, the two figures are close
    enough to nothing else in the payload that only an explicit check catches
    it.
    """
    _write_run(tmp_path)

    cells = load_bundle("crossdataset/protocol_v1", tmp_path).crossdataset["cells"]
    recovered = {(c["train"], c["test"]): c["recovered_pct"] for c in cells if not c["self"]}

    assert recovered[("ids2018", "ids2017")] == pytest.approx(73.9)
    assert recovered[("ids2017", "ids2018")] == pytest.approx(3.6)


def test_each_layout_fills_exactly_its_own_payload() -> None:
    """Every layout-specific key is None on the others, so branching is safe.

    Run over whatever is really in ``results/``. The two non-training layouts
    leave ``models`` empty on purpose -- a cross-dataset run has no single
    score per model and an audit trains nothing -- so the same assertion that
    protects a training bundle from losing its metrics would fire on both.
    """
    root = Path(__file__).resolve().parents[1] / "results"
    if not root.is_dir():
        pytest.skip("no results/ in this checkout")

    for summary in list_bundles(root):
        payload = describe_bundle(summary["id"], root)
        layout = summary["layout"]

        assert (payload["crossdataset"] is not None) == (layout == "crossdataset")
        assert (payload["audit"] is not None) == (layout == "audit")

        if layout in {"crossdataset", "audit"}:
            assert payload["models"] == {}
        else:
            assert payload["models"], f"{summary['id']} lost its per-model metrics"


def test_shipped_run_matches_its_own_summary_table() -> None:
    """Cross-check the real run: the payload must not restate the CSV wrongly."""
    root = Path(__file__).resolve().parents[1] / "results"
    path = root / "crossdataset" / "protocol_v1"
    if not (path / "summary_gap.csv").is_file():
        pytest.skip("protocol_v1 is not present in this checkout")

    import pandas as pd

    gap = pd.read_csv(path / "summary_gap.csv")
    cells = load_bundle("crossdataset/protocol_v1", root).crossdataset["cells"]
    transfers = {(c["mode"], c["model"], c["train"], c["test"]): c for c in cells if not c["self"]}

    assert len(transfers) == len(gap)
    for _, row in gap.iterrows():
        cell = transfers[(row["mode"], row["model"], row["train"], row["test"])]
        assert cell["f1_macro"] == pytest.approx(row["transfer"])
        assert cell["ceiling"] == pytest.approx(row["ceiling"])
        assert cell["recovered_pct"] == pytest.approx(row["recovered_pct"])
