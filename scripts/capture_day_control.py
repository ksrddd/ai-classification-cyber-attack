"""Separate two capture days of the *same* attack from two different attacks.

``scripts/within_dataset_control.py`` asked whether a boundary inside one
corpus separates as well as the boundary between corpora, and answered
"clearly not" in four classes out of six. In DoS and DDoS it answered "yes,
perfectly" -- and that pair of rows turns out not to mean what it looks like.

The 2018 corpus puts each attack tool on its own capture day, and the shared
seven-class schema collapses several tools into one class. So the DDoS group
contrast, labelled "2018-02-20 against 2018-02-21", is really LOIC-HTTP against
HOIC; the DoS one is Hulk against GoldenEye. Those reach AUC 1.0 because they
are different attacks, which says nothing about capture sessions.

This script separates the two explanations by running contrasts where only one
thing varies:

**same tool, different day** -- the sub-label is held fixed and the capture day
changes. This is the control the previous script meant to run. Only a few
sub-labels span more than one day in 2018, and they are the only rows in the
corpus that can answer the question.

**different tool, same class** -- two sub-labels the schema collapses together.
This is the confound itself, measured rather than argued about. If it lands at
1.0 while the first lands far lower, the DoS and DDoS results are explained and
the within-dataset control stands.

Run on 2018 only, because every 2017 attack sub-label lives in exactly one
capture file and neither contrast can be drawn there.

The feature list is read from the adversarial-validation findings rather than
recomputed, so all three scripts measure over the identical 60 columns.

Run::

    python scripts/capture_day_control.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from within_dataset_control import MIN_PER_SIDE, SEEDS, score_contrast  # noqa: E402

from src.crossdataset.labels import map_to_shared  # noqa: E402
from src.crossdataset.loaders import IDS2018_PARQUET, LABEL_COL  # noqa: E402
from src.crossdataset.schema import to_canonical  # noqa: E402

logger = logging.getLogger("capture_day_control")

AV_DIR = PROJECT_ROOT / "results" / "crossdataset" / "adversarial_validation"
OUT = AV_DIR / "capture_day_control.json"

#: Cap per side. The same ceiling the within-dataset control uses, so the two
#: sets of numbers sit on the same scale.
MAX_PER_SIDE = 8_000


def features_from_findings() -> list[str]:
    """The exact 60 columns the adversarial run measured over.

    Recomputing them here would risk a different answer -- a column that is
    constant in 2018 alone is not constant across both corpora -- and the whole
    point is that these contrasts are comparable to that run.
    """
    findings = AV_DIR / "findings.json"
    if not findings.is_file():
        raise FileNotFoundError(
            f"{findings} not found -- run scripts/adversarial_validation.py first."
        )
    return list(json.loads(findings.read_text(encoding="utf-8"))["features"])


def load_2018_with_subtypes(features: list[str]) -> pd.DataFrame:
    """The 2018 corpus with its raw sub-label and capture day kept.

    The protocol's loader collapses the label to the shared seven, which is
    exactly the information these contrasts need, so this reads the parquet
    directly instead. No sampling either: these contrasts are small and drawing
    from the whole corpus avoids a second sampling step to reason about.
    """
    raw = pd.read_parquet(IDS2018_PARQUET)
    frame = to_canonical(raw, "ids2018", keep_dst_port=True)[features].copy()
    frame["_subtype"] = raw[LABEL_COL].astype(str).to_numpy()
    frame["_class"] = map_to_shared(raw[LABEL_COL].astype(str), "ids2018").to_numpy()
    ts = pd.to_datetime(raw["Timestamp"], format="mixed", dayfirst=True, errors="coerce")
    frame["_day"] = ts.dt.date.astype(str).to_numpy()
    return frame.dropna(subset=["_class"]).reset_index(drop=True)


def two_largest(counts: pd.Series) -> tuple[str, str] | None:
    """The two biggest buckets, or None if fewer than two clear the floor."""
    usable = counts[counts >= MIN_PER_SIDE]
    if len(usable) < 2:
        return None
    top = usable.sort_values(ascending=False).index[:2]
    return str(top[0]), str(top[1])


def run_pair(
    label: str,
    kind: str,
    a: pd.DataFrame,
    b: pd.DataFrame,
    features: list[str],
    note: str,
) -> dict:
    n = min(len(a), len(b), MAX_PER_SIDE)
    scored = score_contrast(a, b, features, n)
    logger.info(
        "  %-28s %-18s n=%-6s AUC %.4f ±%.4f  stump %.4f",
        label,
        kind,
        f"{n:,}",
        scored["auc"],
        scored["auc_sd"],
        scored["stump"],
    )
    logger.info("  %-28s %s", "", note)
    return {
        "label": label,
        "kind": kind,
        "n_per_side": int(n),
        "boundary": note,
        **scored,
    }


def same_tool_different_day(frame: pd.DataFrame, features: list[str]) -> list[dict]:
    """Hold the attack tool fixed and change the capture day."""
    results = []
    for subtype, rows in frame.groupby("_subtype", sort=False):
        days = two_largest(rows["_day"].value_counts())
        if days is None:
            continue
        first, second = days
        results.append(
            run_pair(
                subtype,
                "same tool, other day",
                rows[rows["_day"] == first],
                rows[rows["_day"] == second],
                features,
                f"{first} against {second}, identical sub-label",
            )
        )
    return results


def different_tool_same_class(frame: pd.DataFrame, features: list[str]) -> list[dict]:
    """Hold the collapsed class fixed and change the attack tool."""
    results = []
    for klass, rows in frame.groupby("_class", sort=False):
        tools = two_largest(rows["_subtype"].value_counts())
        if tools is None:
            continue
        first, second = tools
        results.append(
            run_pair(
                str(klass),
                "other tool, same class",
                rows[rows["_subtype"] == first],
                rows[rows["_subtype"] == second],
                features,
                f"{first} against {second}, both collapsed to {klass}",
            )
        )
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    features = features_from_findings()
    logger.info("loading the 2018 corpus with sub-labels kept")
    frame = load_2018_with_subtypes(features)
    logger.info(
        "%d rows, %d sub-label(s), %d capture day(s), %d features",
        len(frame),
        frame["_subtype"].nunique(),
        frame["_day"].nunique(),
        len(features),
    )

    logger.info("")
    logger.info("A. same attack tool, different capture day")
    same_day = same_tool_different_day(frame, features)
    if not same_day:
        logger.info("  none: no sub-label spans two days with enough rows")

    logger.info("")
    logger.info("B. different attack tool, same collapsed class")
    same_class = different_tool_same_class(frame, features)

    day_aucs = [r["auc"] for r in same_day]
    tool_aucs = [r["auc"] for r in same_class]

    logger.info("")
    logger.info("=" * 78)
    if day_aucs and tool_aucs:
        logger.info("same tool, other day : %.4f to %.4f", min(day_aucs), max(day_aucs))
        logger.info("other tool, same class: %.4f to %.4f", min(tool_aucs), max(tool_aucs))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "question": (
                    "Is the perfect within-2018 group separation a capture-day "
                    "effect, or two different attack tools the schema collapsed "
                    "into one class?"
                ),
                "corpus": "ids2018",
                "seeds": list(SEEDS),
                "min_per_side": MIN_PER_SIDE,
                "max_per_side": MAX_PER_SIDE,
                "n_features": len(features),
                "same_tool_other_day": same_day,
                "other_tool_same_class": same_class,
                "same_tool_other_day_range": (
                    [round(min(day_aucs), 4), round(max(day_aucs), 4)] if day_aucs else None
                ),
                "other_tool_same_class_range": (
                    [round(min(tool_aucs), 4), round(max(tool_aucs), 4)] if tool_aucs else None
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
