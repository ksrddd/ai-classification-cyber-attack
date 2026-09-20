"""The same attack tool, run in two different labs, two years apart.

Adversarial validation separates CICIDS2017 from CSE-CIC-IDS2018 at AUC
0.9994-1.0000 inside every shared class, and ``within_dataset_control.py``
showed no capture boundary inside either corpus comes close. One explanation
was left standing, and it was this script's own sibling result that raised it:
inside CSE-CIC-IDS2018, two *different attack tools* collapsed into one class
separate at AUC 1.0000.

The shared seven-class schema does that on both sides. "DoS" in 2017 is Hulk,
GoldenEye, slowloris and Slowhttptest mixed together; "DoS" in 2018 is the same
four names in different proportions. So part of the cross-dataset AUC could be
tool composition rather than anything about the labs -- the model separating
"mostly Hulk" from "mostly GoldenEye" and being credited with finding a dataset
fingerprint.

Both corpora name the tool in their raw label column, and six tools appear on
both sides with enough rows to draw from. Comparing Hulk against Hulk holds the
attack program fixed and changes only the lab and the year, which is the
strongest control this data can support:

* **still near 1.0** -- tool composition is not the explanation, and the
  fingerprint is a fact about how each corpus was produced.
* **substantially lower** -- a share of the headline result was the schema
  putting different attacks under one name, and why transfer fails needs
  restating.

Three contrasts per tool, all at the same rows per side
-------------------------------------------------------
``same tool, across corpora``  Hulk 2017 against Hulk 2018 -- the question.
``whole class, across corpora``  DoS 2017 against DoS 2018, re-run at the same
n so the tool-matched number is compared against a like-for-like baseline
rather than against the headline figure from a different sample size.
``different tools, within 2017``  Hulk against GoldenEye inside one corpus --
the 2018 finding repeated on the other side, so it is a property of the schema
rather than a quirk of one dataset.

2017 sub-labels are recovered by joining the raw captures back onto the cleaned
parquet through ``source_file`` and ``_row_index``; the cleaning step collapses
the label column, and the row position it keeps is exactly what makes the
original recoverable.

Run::

    python scripts/tool_matched_av.py
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
from src.crossdataset.loaders import (  # noqa: E402
    IDS2017_PARQUET,
    IDS2018_PARQUET,
    LABEL_COL,
)
from src.crossdataset.schema import to_canonical  # noqa: E402

logger = logging.getLogger("tool_matched_av")

AV_DIR = PROJECT_ROOT / "results" / "crossdataset" / "adversarial_validation"
OUT = AV_DIR / "tool_matched.json"
RAW_2017 = PROJECT_ROOT / "data" / "raw"

MAX_PER_SIDE = 8_000
#: Rows kept per sub-label after loading, before any contrast runs. Comfortably
#: above MAX_PER_SIDE so no draw is constrained by it, and small enough that
#: both corpora fit in memory at once.
KEEP_PER_SUBTYPE = 40_000

#: 2017 label -> 2018 label, for tools that are the same program on both sides.
#:
#: The pairs are by tool name as each corpus wrote it. Two documented families
#: are deliberately absent: FTP-Patator/SSH-Patator against
#: FTP-BruteForce/SSH-Bruteforce, where the attack type matches but the two
#: corpora do not state that the same program produced it, and pairing on type
#: alone would put the confound back in under a different name.
TOOL_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("Hulk", "DoS Hulk", "DoS attacks-Hulk"),
    ("GoldenEye", "DoS GoldenEye", "DoS attacks-GoldenEye"),
    ("Slowloris", "DoS slowloris", "DoS attacks-Slowloris"),
    ("SlowHTTPTest", "DoS Slowhttptest", "DoS attacks-SlowHTTPTest"),
    ("Bot", "Bot", "Bot"),
    # CIC documents the 2017 DDoS capture as LOIC, which is the tool the 2018
    # LOIC-HTTP day used. Kept, and flagged in the output, because it is the
    # one pair resting on the documentation rather than on both labels saying
    # the same word.
    ("LOIC-HTTP", "DDoS", "DDoS attacks-LOIC-HTTP"),
    ("Web brute force", "Web Attack - Brute Force", "Brute Force -Web"),
    ("XSS", "Web Attack - XSS", "Brute Force -XSS"),
)
#: Pairs whose identity rests on CIC's documentation rather than on the label.
DOCUMENTED_ONLY = {"LOIC-HTTP"}


def normalise(label: str) -> str:
    """Fold the 2017 captures' non-ASCII dash so a pair table can be written.

    The Web Attack labels carry a latin-1 en dash that survives into the CSV,
    so ``Web Attack \\x96 Brute Force`` and ``Web Attack - Brute Force`` are the
    same class written two ways.
    """
    out = "".join(c if c.isascii() else "-" for c in str(label))
    # The dash was already corrupted before the CSV was written: the bytes are
    # the UTF-8 encoding of the replacement character, which decodes to *three*
    # non-ASCII characters rather than one. Substituting each separately gives
    # "Web Attack --- Brute Force", which silently matches nothing -- the first
    # version of this function did that and both Web Attack pairs were skipped
    # as if 2017 held no such rows. No real label in either corpus contains a
    # doubled dash, so collapsing runs is safe.
    while "--" in out:
        out = out.replace("--", "-")
    return " ".join(out.split())


def subtypes_2017() -> pd.Series:
    """Recover the original 2017 sub-label for every surviving cleaned row.

    Cleaning collapses ``Label`` to nine classes and keeps ``_row_index``, the
    row's position in its own capture file. Reading each raw capture's label
    column and indexing it by that position puts the tool name back, without
    re-running any of the cleaning.
    """
    keys = pd.read_parquet(IDS2017_PARQUET, columns=["_row_index", "source_file"])
    out = pd.Series(index=keys.index, dtype=object)

    for name, rows in keys.groupby("source_file", sort=False):
        path = RAW_2017 / str(name)
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found -- the 2017 sub-labels live only in the raw "
                "captures, which this script needs to recover the tool names."
            )
        raw = pd.read_csv(
            path,
            usecols=lambda c: c.strip() == "Label",
            encoding="latin-1",
            low_memory=False,
        )
        labels = raw.iloc[:, 0].astype(str).map(normalise).to_numpy()
        out.loc[rows.index] = labels[rows["_row_index"].to_numpy()]
        logger.info("  %-58s %7d row(s)", str(name)[:58], len(rows))

    return out


def load_corpus(
    parquet: Path, dataset: str, subtypes: pd.Series, wanted: set[str], seed: int = 42
) -> pd.DataFrame:
    """Canonical features for the wanted sub-labels only, thinned per sub-label.

    Only the rows a contrast can use are kept, and each sub-label is thinned to
    ``KEEP_PER_SUBTYPE``. Both corpora have to be resident at the same time and
    the 2018 one is 11.4M rows; carrying every row of it to draw 8,000 would
    cost gigabytes for nothing.
    """
    raw = pd.read_parquet(parquet)
    keep = subtypes.isin(wanted).to_numpy()
    raw = raw[keep]
    subtypes = subtypes[keep]

    frame = to_canonical(raw, dataset, keep_dst_port=True)
    frame["_subtype"] = subtypes.to_numpy()
    frame["_class"] = map_to_shared(raw[LABEL_COL].astype(str), dataset).to_numpy()
    frame = frame.dropna(subset=["_class"]).reset_index(drop=True)

    shuffled = frame.sample(frac=1.0, random_state=seed)
    return shuffled.groupby("_subtype", sort=False).head(KEEP_PER_SUBTYPE).reset_index(drop=True)


def features_from_findings() -> list[str]:
    """The exact 60 columns every other script in this audit measured over."""
    findings = AV_DIR / "findings.json"
    if not findings.is_file():
        raise FileNotFoundError(
            f"{findings} not found -- run scripts/adversarial_validation.py first."
        )
    return list(json.loads(findings.read_text(encoding="utf-8"))["features"])


def report(label: str, kind: str, scored: dict, note: str) -> None:
    logger.info(
        "  %-16s %-26s n=%-6s AUC %.4f ±%.4f  stump %.4f   %s",
        label,
        kind,
        f"{scored['n_per_side']:,}",
        scored["auc"],
        scored["auc_sd"],
        scored["stump"],
        note,
    )


def run_pairs(c17: pd.DataFrame, c18: pd.DataFrame, features: list[str]) -> list[dict]:
    """The question, and its like-for-like baseline, for each matched tool."""
    results = []
    for tool, label17, label18 in TOOL_PAIRS:
        a = c17[c17["_subtype"] == label17]
        b = c18[c18["_subtype"] == label18]
        if len(a) < MIN_PER_SIDE or len(b) < MIN_PER_SIDE:
            logger.info(
                "  %-16s skipped: %d / %d row(s), below %d on one side",
                tool,
                len(a),
                len(b),
                MIN_PER_SIDE,
            )
            continue

        klass = str(a["_class"].iloc[0])
        whole_a = c17[c17["_class"] == klass]
        whole_b = c18[c18["_class"] == klass]
        # The same n for both contrasts: the tool-matched number has to be read
        # against a class-level number drawn the same way, not against the
        # headline figure computed on a different sample size.
        n = min(len(a), len(b), len(whole_a), len(whole_b), MAX_PER_SIDE)

        same_tool = score_contrast(a, b, features, n)
        whole_class = score_contrast(whole_a, whole_b, features, n)
        for row in (same_tool, whole_class):
            row["n_per_side"] = int(n)

        report(tool, "same tool, across corpora", same_tool, f"{label17} / {label18}")
        report(tool, "whole class, across corpora", whole_class, klass)

        results.append(
            {
                "tool": tool,
                "shared_class": klass,
                "ids2017_label": label17,
                "ids2018_label": label18,
                "identity_from_documentation": tool in DOCUMENTED_ONLY,
                "n_per_side": int(n),
                "n_available": {"ids2017": int(len(a)), "ids2018": int(len(b))},
                "same_tool_across_corpora": same_tool,
                "whole_class_across_corpora": whole_class,
                "delta": round(whole_class["auc"] - same_tool["auc"], 4),
            }
        )
    return results


def run_within_2017(c17: pd.DataFrame, features: list[str]) -> list[dict]:
    """Two tools the schema calls one class, inside CICIDS2017.

    The 2018 half of this was already measured and reached 1.0000 twice. If
    2017 agrees, collapsing tools into a class is a property of the schema
    rather than of one corpus -- which is what makes it a finding about the
    protocol instead of a quirk to note in passing.
    """
    results = []
    for klass, rows in c17.groupby("_class", sort=False):
        counts = rows["_subtype"].value_counts()
        usable = counts[counts >= MIN_PER_SIDE]
        if len(usable) < 2:
            continue
        first, second = usable.index[:2]
        a = rows[rows["_subtype"] == first]
        b = rows[rows["_subtype"] == second]
        n = min(len(a), len(b), MAX_PER_SIDE)
        scored = score_contrast(a, b, features, n)
        scored["n_per_side"] = int(n)
        report(str(klass), "other tool, same class", scored, f"{first} / {second}")
        results.append(
            {
                "shared_class": str(klass),
                "tools": [str(first), str(second)],
                "n_per_side": int(n),
                **scored,
            }
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

    logger.info("recovering 2017 sub-labels from the raw captures")
    subs17 = subtypes_2017()

    # Everything a contrast might use: the paired tools plus every tool sharing
    # a class with one of them, since the class-level baseline needs the rest
    # of the class too.
    #
    # The class comes from the parquet's own label column rather than from the
    # sub-label: 2017's mapper takes the nine-class label the cleaning step
    # already produced, and handing it "DoS Hulk" returns NaN for every row.
    subs17 = subs17.map(normalise)
    collapsed17 = map_to_shared(
        pd.read_parquet(IDS2017_PARQUET, columns=[LABEL_COL])[LABEL_COL].astype(str),
        "ids2017",
    )
    paired = {normalise(x) for _, x, _ in TOOL_PAIRS}
    classes = set(collapsed17[subs17.isin(paired)].dropna().unique())
    wanted17 = set(subs17[collapsed17.isin(classes)].unique())

    logger.info("loading CICIDS2017")
    c17 = load_corpus(IDS2017_PARQUET, "ids2017", subs17, wanted17)

    logger.info("loading CSE-CIC-IDS2018")
    subs18 = (
        pd.read_parquet(IDS2018_PARQUET, columns=[LABEL_COL])[LABEL_COL].astype(str).map(normalise)
    )
    # Narrowed the same way as 2017. Keeping every sub-label would canonicalise
    # all 11.4M rows with both corpora already resident, which costs gigabytes
    # to then draw 8,000 from.
    paired18 = {normalise(x) for _, _, x in TOOL_PAIRS}
    cls18 = map_to_shared(subs18, "ids2018")
    classes18 = set(cls18[subs18.isin(paired18)].dropna().unique())
    wanted18 = set(subs18[cls18.isin(classes18)].unique())
    c18 = load_corpus(IDS2018_PARQUET, "ids2018", subs18, wanted18)

    logger.info(
        "2017: %d row(s), %d sub-label(s) | 2018: %d row(s), %d sub-label(s)",
        len(c17),
        c17["_subtype"].nunique(),
        len(c18),
        c18["_subtype"].nunique(),
    )

    logger.info("")
    logger.info("same tool across corpora, against the whole class at the same n")
    pairs = run_pairs(c17, c18, features)

    logger.info("")
    logger.info("different tools inside CICIDS2017")
    within = run_within_2017(c17, features)

    logger.info("")
    logger.info("=" * 78)
    if pairs:
        matched = [p["same_tool_across_corpora"]["auc"] for p in pairs]
        whole = [p["whole_class_across_corpora"]["auc"] for p in pairs]
        logger.info("same tool, across corpora : %.4f to %.4f", min(matched), max(matched))
        logger.info("whole class, across corpora: %.4f to %.4f", min(whole), max(whole))
    if within:
        w = [r["auc"] for r in within]
        logger.info("other tool, inside 2017   : %.4f to %.4f", min(w), max(w))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "question": (
                    "With the attack tool held fixed, do the two corpora still " "separate?"
                ),
                "seeds": list(SEEDS),
                "min_per_side": MIN_PER_SIDE,
                "max_per_side": MAX_PER_SIDE,
                "n_features": len(features),
                "tool_pairs": pairs,
                "different_tools_within_ids2017": within,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
