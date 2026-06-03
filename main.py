"""Command-line entry point for the cyber-attack classification pipeline.

Usage
-----
    python main.py --stage eda
    python main.py --stage preprocess
    python main.py --stage train    --model rf
    python main.py --stage evaluate
    python main.py --stage explain
    python main.py --stage predict  --input my_traffic.csv
    python main.py --stage all

Stages call into ``src.pipelines.*`` — those modules contain the real logic.
This file is intentionally thin: parse args, configure logging, dispatch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import configure_logging, default_log_file  # noqa: E402

STAGES = ("eda", "preprocess", "train", "evaluate", "explain", "predict", "all")
MODELS = ("lr", "rf", "mlp", "all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cyberml",
        description="AI-Based Cyber Attack Classification — pipeline driver",
    )
    parser.add_argument("--stage", choices=STAGES, required=True,
                        help="Which pipeline stage to run")
    parser.add_argument("--model", choices=MODELS, default="all",
                        help="Which model to train/evaluate (default: all)")
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "src" / "config" / "config.yaml",
                        help="Path to YAML config (default: src/config/config.yaml)")
    parser.add_argument("--input", type=Path, default=None,
                        help="CSV file for --stage predict")
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="Override config's raw data directory "
                             "(useful for running against data/sample/ instead of data/raw/)")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(level=args.log_level, log_file=default_log_file())

    if args.stage in ("eda", "all"):
        from src.pipelines.eda import run as run_eda
        run_eda(args.config, raw_dir_override=args.raw_dir)

    if args.stage in ("preprocess", "all"):
        from src.pipelines.preprocess import run as run_preprocess
        run_preprocess(args.config, raw_dir_override=args.raw_dir)

    if args.stage in ("train", "all"):
        # TODO(phase 6): from src.pipelines.train import run; run(args.config, args.model)
        if args.stage != "all":
            raise NotImplementedError("Phase 6 will wire up the train stage.")

    if args.stage in ("evaluate", "all"):
        if args.stage != "all":
            raise NotImplementedError("Phase 8 will wire up the evaluate stage.")

    if args.stage in ("explain", "all"):
        if args.stage != "all":
            raise NotImplementedError("Phase 9 will wire up the explain stage.")

    if args.stage == "predict":
        if args.input is None:
            raise SystemExit("--stage predict requires --input PATH.csv")
        raise NotImplementedError("Phase 12 will wire up the predict stage.")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
