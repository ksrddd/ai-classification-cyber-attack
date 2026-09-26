"""What the clean-before-sampling fix changed in the cross-dataset result.

``protocol_v1`` sampled the 2018 corpus to 300,000 rows *before* cleaning it;
``protocol_v2`` rebuilt that corpus from the raw captures and cleans before
sampling, as the 2017 side always did. Same seven models, same two split
modes, same five seeds, same shared schema -- the preparation order of one
corpus is the only thing that moved. So every difference between the two runs
is attributable to it, and the two headline claims can be re-checked directly:

**The asymmetry.** Training on 2018 and testing on 2017 recovered far more of
the achievable range than the reverse. Does that survive, and by how much?

**The split mode.** A random split inflated the 2017 ceiling by about 0.25 and
the 2018 one by very little, and under a random split the two transfer
directions looked symmetric. Does the split mode still erase the asymmetry?

A built-in control: the 2017 corpus was not touched. Its ceilings should
therefore reproduce exactly for every model whose fit is deterministic, and
any drift on that side measures run-to-run nondeterminism rather than the fix.

Run::

    python scripts/compare_protocols.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results" / "crossdataset"
OUT_DIR = RESULTS / "protocol_comparison"

logger = logging.getLogger("compare_protocols")

VERSIONS = ("v1", "v2")
FORWARD, BACKWARD = "2017->2018", "2018->2017"
#: A 2017 ceiling that moved by less than this between runs is taken as
#: reproduced. The protocol reports four decimals, so anything smaller is
#: rounding rather than a change.
REPRODUCED = 0.0001


def load(version: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = RESULTS / f"protocol_{version}"
    gap = pd.read_csv(path / "summary_gap.csv")
    gap["direction"] = [FORWARD if t == "ids2017" else BACKWARD for t in gap["train"]]
    return gap, pd.read_csv(path / "summary_leakage.csv")


def recovered_by_model(gap: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Recovered percent per model, one column per direction, plus the asymmetry."""
    t = gap[gap["mode"] == mode].pivot(index="model", columns="direction", values="recovered_pct")
    t["asymmetry"] = t[BACKWARD] - t[FORWARD]
    return t


def summarise(t: pd.DataFrame) -> dict:
    return {
        "mean": {c: round(float(t[c].mean()), 1) for c in t.columns},
        "median": {c: round(float(t[c].median()), 1) for c in t.columns},
        "models_favouring_2018_to_2017": int((t["asymmetry"] > 0).sum()),
        "n_models": int(len(t)),
        "best_2018_to_2017": {
            "model": str(t[BACKWARD].idxmax()),
            "recovered_pct": round(float(t[BACKWARD].max()), 1),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    runs = {v: load(v) for v in VERSIONS}
    report: dict = {"versions": list(VERSIONS)}
    per_model_rows = []

    # 1. The asymmetry, under each split mode.
    report["asymmetry"] = {}
    for mode in ("chronological", "random"):
        report["asymmetry"][mode] = {}
        for v in VERSIONS:
            t = recovered_by_model(runs[v][0], mode)
            report["asymmetry"][mode][v] = summarise(t)
            for model, r in t.iterrows():
                per_model_rows.append(
                    {
                        "version": v,
                        "mode": mode,
                        "model": model,
                        "recovered_2017_to_2018": r[FORWARD],
                        "recovered_2018_to_2017": r[BACKWARD],
                        "asymmetry": round(float(r["asymmetry"]), 1),
                    }
                )

    # 2. The split mode: how far a random split moves recovered percent, per
    #    direction. Recovered percent is measured against the ceiling, so a
    #    split that inflates one corpus's ceiling shrinks every transfer into
    #    that corpus even when the transfer itself did not change.
    report["split_mode_effect"] = {}
    for v in VERSIONS:
        gap = runs[v][0]
        p = gap.pivot_table(index=["model", "direction"], columns="mode", values="recovered_pct")
        delta = (p["random"] - p["chronological"]).groupby(level="direction")
        report["split_mode_effect"][v] = {
            d: {
                "mean": round(float(s.mean()), 1),
                "min": round(float(s.min()), 1),
                "max": round(float(s.max()), 1),
            }
            for d, s in delta
        }

    # 3. Within-dataset inflation of the ceiling, and the ceilings themselves.
    report["ceiling_inflation"] = {
        v: {
            corpus: round(float(s.mean()), 4)
            for corpus, s in runs[v][1].groupby("test")["inflation"]
        }
        for v in VERSIONS
    }
    report["ceilings"] = {
        v: {
            f"{mode}/{corpus}": round(float(s.mean()), 4)
            for (mode, corpus), s in runs[v][0].groupby(["mode", "test"])["ceiling"]
        }
        for v in VERSIONS
    }

    # 4. The control: 2017 was not touched, so its chronological ceilings
    #    should reproduce for every deterministic model.
    lk = {v: runs[v][1].set_index(["test", "model"]) for v in VERSIONS}
    c17 = pd.DataFrame({v: lk[v].loc["ids2017", "chronological"] for v in VERSIONS})
    c17["delta"] = (c17["v2"] - c17["v1"]).round(4)
    report["control_2017_ceiling"] = {
        "reproduced": sorted(c17.index[c17["delta"].abs() < REPRODUCED]),
        "drifted": {m: float(d) for m, d in c17["delta"].items() if abs(d) >= REPRODUCED},
    }

    logger.info(
        "asymmetry (chronological): v1 mean %.1f, v2 mean %.1f points",
        report["asymmetry"]["chronological"]["v1"]["mean"]["asymmetry"],
        report["asymmetry"]["chronological"]["v2"]["mean"]["asymmetry"],
    )
    logger.info(
        "asymmetry (random):        v1 mean %.1f, v2 mean %.1f points",
        report["asymmetry"]["random"]["v1"]["mean"]["asymmetry"],
        report["asymmetry"]["random"]["v2"]["mean"]["asymmetry"],
    )
    logger.info("2017 ceiling reproduced for: %s", report["control_2017_ceiling"]["reproduced"])
    logger.info("2017 ceiling drifted for:    %s", report["control_2017_ceiling"]["drifted"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(per_model_rows).to_csv(args.out_dir / "per_model_recovered.csv", index=False)
    logger.info("wrote %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
