"""Tests for src.data.eda."""

from __future__ import annotations

from pathlib import Path

from src.data import eda


def test_describe_dataset_keys(cleaned_df) -> None:
    summary = eda.describe_dataset(cleaned_df)
    for key in (
        "shape", "n_features", "missing_total", "infinite_total",
        "label_distribution", "duplicate_row_count",
    ):
        assert key in summary, f"summary missing key: {key}"


def test_describe_dataset_label_distribution_sums_to_row_count(cleaned_df) -> None:
    summary = eda.describe_dataset(cleaned_df)
    assert sum(summary["label_distribution"].values()) == cleaned_df.shape[0]


def test_plot_class_distribution_writes_png(cleaned_df, tmp_path: Path) -> None:
    out = eda.plot_class_distribution(cleaned_df, save_to=tmp_path / "cd.png")
    assert out.exists() and out.stat().st_size > 0


def test_run_eda_writes_all_four_figures(cleaned_df, tmp_path: Path) -> None:
    summary = eda.run_eda(cleaned_df, output_dir=tmp_path)
    figs = summary["figures"]
    assert set(figs.keys()) == {
        "class_distribution",
        "missing_value_audit",
        "correlation_heatmap",
        "feature_distributions",
    }
    for path in figs.values():
        assert Path(path).exists()
