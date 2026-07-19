from __future__ import annotations

from scripts.dgx_preflight import PROJECT_ROOT, run_static_preflight


def test_static_dgx_preflight_passes_for_delivery_tree() -> None:
    report = run_static_preflight(
        PROJECT_ROOT,
        PROJECT_ROOT / "configs" / "splits" / "source_holdout_v3_full_70_30.json",
    )
    assert report["passed"], report["checks"]
    assert report["checks"]["gpu_pipeline_configuration"]["passed"]
    ratio = report["checks"]["source_held_ratio"]
    assert ratio["passed"]
    assert ratio["detail"]["train_rows"] == 9_529_802
    assert ratio["detail"]["test_rows"] == 4_084_201
    assert "does not prove DGX" in report["claim"]
