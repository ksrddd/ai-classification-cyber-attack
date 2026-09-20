"""Tests for the audit layout in the bundle registry.

An audit sits in the same picker as the training runs and the cross-dataset
runs, and is unlike both: it trains nothing and scores nothing. It asks whether
an existing result can be believed -- do the paired columns still mean the same
thing, can the stacking ensemble see its own test set, does anything in the
feature matrix name the corpus a row came from, does removing those columns
help. There is no accuracy anywhere in it.

The contract these tests pin:

* all four audits are discoverable, so they can be selected;
* each is identified by the files it actually holds, not by its directory name
  -- two of them carry a ``findings.json`` and would otherwise be
  indistinguishable;
* ``models`` stays empty, so a page written for a training layout renders the
  explained mismatch rather than an empty ranking;
* the payload carries ``kind``, which is the only thing the UI may branch on;
* a directory that merely looks audit-shaped is not offered.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.bundles.registry import _discover, describe_bundle, list_bundles, load_bundle


def _write_adversarial(root: Path, name: str = "adversarial_validation") -> Path:
    path = root / "crossdataset" / name
    path.mkdir(parents=True)
    (path / "findings.json").write_text(
        json.dumps(
            {
                "n_features": 60,
                "min_per_side": 500,
                "max_per_side": 20000,
                "seeds": [42, 43],
                "classes": [
                    {
                        "shared_class": "Benign",
                        "n_per_side": 20000,
                        "n_seeds": 2,
                        "auc": 0.9994,
                        "auc_sd": 0.0001,
                        "auc_per_seed": [0.9993, 0.9995],
                        "auc_after_dropping_top_k": {"1": 0.9995, "10": 0.9981},
                        "importance": [
                            {"feature": "min_seg_size_forward", "importance": 0.041, "sd": 0.01},
                            {"feature": "RST Flag Count", "importance": 0.033, "sd": 0.01},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (path / "ranking.csv").write_text(
        "feature,mean,max,count\nmin_seg_size_forward,0.0482,0.2482,6\n",
        encoding="utf-8",
    )
    return path


def _write_diagnostics(path: Path) -> None:
    (path / "diagnostics.json").write_text(
        json.dumps(
            {
                "split": "random stratified 70%/30%, same for every class",
                "preprocessing": "none",
                "seeds": [42, 43],
                "max_per_side": 8000,
                "near_dup_quantile": 0.001,
                "classes": [
                    {
                        "shared_class": "Benign",
                        "n_per_side": 8000,
                        "auc_mean": 0.999,
                        "auc_sd": 0.0002,
                        "best_single_feature": "min_seg_size_forward",
                        "best_single_feature_auc": 0.7659,
                        "best_single_feature_median": {"ids2017": 20.0, "ids2018": 20.0},
                        "disjoint_columns": [],
                        "rows_identical_across_corpora": 13,
                        "rows_identical_across_split": 0,
                        "near_duplicate_test_share": 0.00104,
                        "model_ladder": {"stump": 0.7165, "lightgbm": 0.999},
                        "null_ladder": {"ids2017": {"stump": 0.4996, "lightgbm": 0.4956}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_control(path: Path) -> None:
    (path / "within_dataset_control.json").write_text(
        json.dumps(
            {
                "question": "Does a capture boundary inside one corpus separate?",
                "verdict": "fingerprint confirmed",
                "reason": "every separable class keeps its within-dataset AUC lower",
                "seeds": [42, 43],
                "min_per_side": 300,
                "separable_threshold": 0.99,
                "n_classes_judged": 1,
                "min_gap": 0.24,
                "max_within_auc": 0.76,
                "classes": [
                    {
                        "shared_class": "Benign",
                        "n_per_side": 4000,
                        "cross_auc": 0.999,
                        "within_auc_max": 0.759,
                        "gap": 0.24,
                        "contrasts": {
                            "cross": {
                                "auc": 0.999,
                                "auc_sd": 0.0002,
                                "stump": 0.7165,
                                "kind": "cross",
                                "boundary": "CICIDS2017 against CSE-CIC-IDS2018",
                                "n_available_per_side": 20000,
                            },
                            "time_ids2017": {
                                "auc": 0.759,
                                "auc_sd": 0.003,
                                "stump": 0.61,
                                "kind": "within",
                                "boundary": "earliest against latest half of ids2017",
                                "n_available_per_side": 12000,
                            },
                            "null_ids2017": {
                                "auc": 0.501,
                                "auc_sd": 0.004,
                                "stump": 0.5,
                                "kind": "null",
                                "boundary": "two random halves of ids2017",
                                "n_available_per_side": 12000,
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_tool_matched(path: Path) -> None:
    (path / "tool_matched.json").write_text(
        json.dumps(
            {
                "question": "With the attack tool held fixed, do the corpora separate?",
                "seeds": [42, 43],
                "tool_pairs": [
                    {
                        "tool": "Hulk",
                        "shared_class": "DoS",
                        "ids2017_label": "DoS Hulk",
                        "ids2018_label": "DoS attacks-Hulk",
                        "identity_from_documentation": False,
                        "n_per_side": 8000,
                        "n_available": {"ids2017": 40000, "ids2018": 40000},
                        "same_tool_across_corpora": {
                            "auc": 0.9999,
                            "auc_sd": 0.0001,
                            "stump": 0.9529,
                        },
                        "whole_class_across_corpora": {
                            "auc": 1.0,
                            "auc_sd": 0.0,
                            "stump": 0.8935,
                        },
                        "delta": 0.0001,
                    }
                ],
                "different_tools_within_ids2017": [
                    {
                        "shared_class": "Web Attack",
                        "tools": ["Web Attack - Brute Force", "Web Attack - XSS"],
                        "n_per_side": 652,
                        "auc": 0.6768,
                        "stump": 0.5811,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_feature_mapping(root: Path) -> Path:
    path = root / "crossdataset" / "feature_mapping_audit"
    path.mkdir(parents=True)
    (path / "findings.json").write_text(
        json.dumps(
            {
                "n_rows_ids2017": 384889,
                "n_rows_ids2018": 399623,
                "n_features": 77,
                "n_distinct_measurements": 61,
                "invariants": [
                    {
                        "invariant": "variance_eq_std_squared",
                        "rate_ids2017": 0.0,
                        "rate_ids2018": 0.0,
                        "verdict": "PASS",
                        "columns": ["Packet Length Variance", "Packet Length Std"],
                        "note": "variance is the square of the standard deviation",
                    }
                ],
                "duplicate_groups": [["Avg Bwd Segment Size", "Bwd Packet Length Mean"]],
                "scale_audit": [
                    {
                        "feature": "Flow Duration",
                        "flag": "SCALE",
                        "reason": "median differs by 3 decades",
                        "ids2017": {"median": 3.0},
                        "ids2018": {"median": 4000.0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (path / "report.md").write_text("# mapping audit\n", encoding="utf-8")
    return path


def _write_stacking_leakage(root: Path) -> Path:
    path = root / "crossdataset" / "stacking_leakage_audit"
    path.mkdir(parents=True)
    for name, distance in (("ids2017", 0.052408), ("ids2018", 0.270866)):
        (path / f"{name}.json").write_text(
            json.dumps(
                {
                    "dataset": name,
                    "run_utc": "2026-09-12T12:03:36+00:00",
                    "audit_rows": 40000,
                    "seed": 42,
                    "passed": True,
                    "n_checks": 2,
                    "n_failed": 0,
                    "checks": [
                        {
                            "section": "1. train/test disjointness",
                            "check": "no duplicate flow across split",
                            "passed": True,
                            "detail": "0 distinct flow(s) appear in both",
                        },
                        {
                            "section": "2. meta-features are out-of-fold",
                            "check": "meta-learner was fitted on OOF",
                            "passed": True,
                            "detail": f"out-of-fold 0.000000, in-sample {distance:f}",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
    return path


def _write_ablation(root: Path) -> Path:
    path = root / "crossdataset" / "ablation"
    path.mkdir(parents=True)
    (path / "summary.csv").write_text(
        "k,mode,ceiling,transfer,gap,recovered_pct\n"
        "0,chronological,0.7392,0.2462,-0.493,22.5202\n"
        "20,chronological,0.6547,0.1844,-0.4704,12.2\n",
        encoding="utf-8",
    )
    (path / "per_model.csv").write_text(
        "model,0,5,10,20,change\nlightgbm,26.2,9.9,8.3,6.1,-20.1\n",
        encoding="utf-8",
    )
    (path / "PREREGISTRATION.md").write_text(
        "# Ablation pre-registration\n\nWritten before the runs.\n", encoding="utf-8"
    )
    return path


@pytest.fixture
def results(tmp_path: Path) -> Path:
    """One of each audit, side by side, as they sit on disk."""
    adversarial = _write_adversarial(tmp_path)
    _write_diagnostics(adversarial)
    _write_control(adversarial)
    _write_tool_matched(adversarial)
    _write_feature_mapping(tmp_path)
    _write_stacking_leakage(tmp_path)
    _write_ablation(tmp_path)
    return tmp_path


def test_every_audit_is_discoverable(results: Path) -> None:
    assert set(_discover(results)) == {
        "crossdataset/adversarial_validation",
        "crossdataset/feature_mapping_audit",
        "crossdataset/stacking_leakage_audit",
        "crossdataset/ablation",
    }


def test_kind_comes_from_the_files_not_the_directory_name(tmp_path: Path) -> None:
    """Two audits carry a findings.json; only their companions tell them apart.

    Renaming a directory must not change what the registry thinks it holds --
    a run copied aside under another name would otherwise be mis-rendered.
    """
    _write_adversarial(tmp_path, name="some_other_name")
    bundle = load_bundle("crossdataset/some_other_name", tmp_path)
    assert bundle.audit is not None
    assert bundle.audit["kind"] == "adversarial_validation"


def test_audit_reports_no_models(results: Path) -> None:
    """A page written for a training layout must find nothing to rank here."""
    for bundle_id in _discover(results):
        bundle = load_bundle(bundle_id, results)
        assert bundle.layout == "audit"
        assert bundle.models == {}
        assert bundle.run["n_models"] == 0
        assert bundle.run["majority_baseline_acc"] is None


def test_summary_names_the_question(results: Path) -> None:
    kinds = {b["id"]: b["audit_kind"] for b in list_bundles(results)}
    assert kinds["crossdataset/ablation"] == "ablation"
    assert kinds["crossdataset/stacking_leakage_audit"] == "stacking_leakage"


def test_adversarial_payload_keeps_the_ablation_curve(results: Path) -> None:
    audit = describe_bundle("crossdataset/adversarial_validation", results)["audit"]
    assert audit["ablation_k"] == [1, 10]
    klass = audit["classes"][0]
    assert klass["shared_class"] == "Benign"
    assert klass["ablation"] == {"1": 0.9995, "10": 0.9981}
    assert audit["ranking"][0]["feature"] == "min_seg_size_forward"


def test_diagnostics_absent_reads_as_not_run(results: Path, tmp_path: Path) -> None:
    """Missing diagnostics must be None, never an empty pass.

    An AUC of 1.0 that was never stress-tested and one that survived every
    check are different claims, and rendering the first as the second is the
    one mistake this payload must not permit.
    """
    assert (
        describe_bundle("crossdataset/adversarial_validation", results)["audit"]["diagnostics"]
        is not None
    )

    bare = tmp_path / "bare"
    _write_adversarial(bare)
    assert (
        describe_bundle("crossdataset/adversarial_validation", bare)["audit"]["diagnostics"] is None
    )


def test_stacking_leakage_stays_per_corpus(results: Path) -> None:
    """The corpora are reported separately because only one of them is at risk.

    The raw 2018 corpus is 29% duplicate flows, so it is the side where the
    order of deduplication and splitting matters. Merging the verdicts would
    hide which corpus proved it.
    """
    audit = describe_bundle("crossdataset/stacking_leakage_audit", results)["audit"]
    assert [c["dataset"] for c in audit["corpora"]] == ["ids2017", "ids2018"]
    assert [c["label"] for c in audit["corpora"]] == ["CICIDS2017", "CSE-CIC-IDS2018"]
    assert all(c["passed"] for c in audit["corpora"])


def test_ablation_keeps_per_model_rows_wide(results: Path) -> None:
    """One column per K, so a row can be read across.

    Whether a model falls monotonically as more columns are dropped is what
    separates a real effect from one K that happened to look good, and long
    form would take that away.
    """
    audit = describe_bundle("crossdataset/ablation", results)["audit"]
    assert audit["ks"] == [0, 5, 10, 20]
    row = audit["per_model"][0]
    assert row["recovered_pct"]["0"] == 26.2
    assert row["change"] == -20.1
    assert audit["preregistration"].startswith("# Ablation pre-registration")


def test_a_directory_that_only_looks_audit_shaped_is_not_offered(tmp_path: Path) -> None:
    """``findings.json`` alone is ambiguous and must not resolve to a kind."""
    stray = tmp_path / "crossdataset" / "half_finished"
    stray.mkdir(parents=True)
    (stray / "findings.json").write_text("{}", encoding="utf-8")
    assert _discover(tmp_path) == {}


def test_control_keeps_each_boundary_separate(results: Path) -> None:
    """A per-contrast AUC must not be flattened into one "within-dataset" number.

    A time split and a capture-file split of the same class are different
    claims and can disagree, and the argument the control makes is a comparison
    between named boundaries -- collapsing them would leave a number nobody can
    check.
    """
    control = describe_bundle("crossdataset/adversarial_validation", results)["audit"]["control"]
    assert control["verdict"] == "fingerprint confirmed"
    contrasts = control["classes"][0]["contrasts"]
    assert {v["kind"] for v in contrasts.values()} == {"cross", "within", "null"}
    assert contrasts["time_ids2017"]["boundary"].startswith("earliest against latest")


def test_control_absent_reads_as_not_run(tmp_path: Path) -> None:
    """Never a silent pass: not running the control is not the same as passing it."""
    _write_diagnostics(_write_adversarial(tmp_path))
    audit = describe_bundle("crossdataset/adversarial_validation", tmp_path)["audit"]
    assert audit["diagnostics"] is not None
    assert audit["control"] is None


def test_tool_matched_keeps_its_like_for_like_baseline(results: Path) -> None:
    """A tool-matched AUC without its class-level twin is not readable.

    The point of the contrast is the comparison: if the same program across
    corpora separates as well as the whole mixed class does, tool composition
    was never what the classifier used. Dropping the baseline to save a field
    would leave a number with nothing to be measured against.
    """
    matched = describe_bundle("crossdataset/adversarial_validation", results)["audit"]["control"][
        "tool_matched"
    ]
    pair = matched["tool_pairs"][0]
    assert pair["same_tool"]["auc"] == 0.9999
    assert pair["whole_class"]["auc"] == 1.0
    assert pair["same_tool"]["stump"] == 0.9529
    assert pair["identity_from_documentation"] is False


def test_tool_matched_absent_reads_as_not_run(results: Path, tmp_path: Path) -> None:
    """Present here, absent in a run that predates the script -- never a pass."""
    assert (
        describe_bundle("crossdataset/adversarial_validation", results)["audit"]["control"][
            "tool_matched"
        ]
        is not None
    )

    older = tmp_path / "older"
    adversarial = _write_adversarial(older)
    _write_control(adversarial)
    assert (
        describe_bundle("crossdataset/adversarial_validation", older)["audit"]["control"][
            "tool_matched"
        ]
        is None
    )
