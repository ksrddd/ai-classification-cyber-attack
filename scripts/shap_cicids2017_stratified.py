"""Class-stratified SHAP for a CICIDS2017 run bundle.

``main.py --stage explain`` draws 1,000 test rows at random, so on this corpus
nearly all of them are BENIGN and the per-class ranking of a rare class (Web
Attack, Infiltration, Heartbleed) rests on a handful of rows. This script takes
up to ``--per-class`` rows of every class from the run's locked test partition,
the same stratified mode ``scripts/shap_ids2018.py`` uses for IDS2018 bundles.

Usage:
    python scripts/shap_cicids2017_stratified.py --run-name cicids2017_temporal_v1_shap \
        --model catboost --per-class 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.artifacts.paths import result_run_dir  # noqa: E402
from src.config.constants import RESULTS_DIR  # noqa: E402
from src.data.temporal_split import load_temporal_manifest, temporal_source_split  # noqa: E402
from src.explainability.shap_analyzer import analyze_model  # noqa: E402
from src.pipelines.explain import CLEAN_CACHE_PATH, ROW_INDEX_COLUMN, SPLIT_MANIFEST_PATH  # noqa: E402

logger = logging.getLogger("shap_cicids2017_stratified")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default="catboost")
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_dir = result_run_dir(args.run_name, results_root=RESULTS_DIR)
    feature_cols = json.loads((run_dir / "feature_columns.json").read_text(encoding="utf-8"))
    encoder = joblib.load(run_dir / "label_encoder.joblib")
    class_names = [str(c) for c in encoder.classes_]

    manifest = load_temporal_manifest(SPLIT_MANIFEST_PATH)
    split_keys = ["Label", "source_file", ROW_INDEX_COLUMN]
    frame = pd.read_parquet(
        CLEAN_CACHE_PATH, columns=split_keys + [c for c in feature_cols if c not in split_keys]
    )
    _, _, test_df = temporal_source_split(
        frame,
        order_column=ROW_INDEX_COLUMN,
        test_size=float(manifest["test_size"]),
        min_test_per_class=int(manifest["min_test_per_class"]),
    )
    del frame
    logger.info("Rebuilt locked test partition: %d rows", len(test_df))

    X_test = test_df[feature_cols].astype(np.float32).reset_index(drop=True)
    y_test = encoder.transform(test_df["Label"].astype(str)).astype(int)

    pipeline = joblib.load(run_dir / f"{args.model}.joblib")
    result = analyze_model(
        pipeline,
        X_test,
        class_names=class_names,
        model_name=args.model,
        top_k=args.top_k,
        y=y_test,
        per_class_samples=args.per_class,
        save_dir=run_dir / f"shap_per{args.per_class}" / args.model,
    )
    print(json.dumps({"overall": result.top_features_overall[:5],
                      "per_class_n": result.per_class_n}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
