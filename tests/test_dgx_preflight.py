from __future__ import annotations

from scripts.dgx_preflight import PROJECT_ROOT, run_static_preflight


def test_static_dgx_preflight_passes_for_delivery_tree() -> None:
    report = run_static_preflight(
        PROJECT_ROOT,
        PROJECT_ROOT / "configs" / "splits" / "source_holdout_v2_70_30.json",
    )
    assert report["passed"], report["checks"]
    assert report["checks"]["gpu_pipeline_configuration"]["passed"]
    assert "does not prove DGX" in report["claim"]
