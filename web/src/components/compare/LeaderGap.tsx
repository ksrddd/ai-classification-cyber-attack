"use client";

/**
 * How far each model sits behind the leader, with the interval on that gap.
 *
 * The obvious chart — every model's macro-F1 with its own confidence interval
 * on a shared axis — is wrong here, and wrong in a way that flatters the worst
 * model. Each model's marginal interval on this test set is wide (roughly
 * 0.62–0.80) because macro-F1 itself swings from resample to resample when
 * three classes have one to three test flows. Drawn that way, all seven
 * intervals overlap, including two models that disagree on 1,955 of 90,000
 * flows. Overlapping marginal intervals do not mean two estimates are
 * indistinguishable; that inference is a well-known error.
 *
 * The models are scored on the *same* resamples, so the interval on their
 * pairwise difference is far tighter and is the quantity that actually answers
 * "is this gap real". That is what this chart draws: the leader pinned at
 * zero, every rival as a difference with its paired interval. A bar touching
 * the zero line is a gap the data cannot confirm.
 */

import { clsx } from "clsx";
import type { BundleSignificance } from "@/lib/bundles";
import { isAbsent } from "@/lib/bundles";

export type GapRow = {
  name: string;
  delta: number;
  low: number;
  high: number;
  p: number | null;
  disagree: number | null;
  separable: boolean | null;
};

export function buildGaps(
  significance: Record<string, BundleSignificance>,
  knownModels: string[],
): GapRow[] {
  return Object.entries(significance)
    .filter(
      ([name, s]) =>
        knownModels.includes(name) &&
        !isAbsent(s.delta_f1_macro) &&
        !isAbsent(s.delta_ci_low) &&
        !isAbsent(s.delta_ci_high),
    )
    .map(([name, s]) => ({
      name,
      delta: s.delta_f1_macro as number,
      low: s.delta_ci_low as number,
      high: s.delta_ci_high as number,
      p: s.mcnemar_p,
      disagree: s.disagreeing_flows,
      separable: s.separable_from_leader,
    }))
    .sort((a, b) => a.delta - b.delta);
}

export function LeaderGap({ leader, rows }: { leader: string; rows: GapRow[] }) {
  if (!rows.length) return null;

  const rawMax = Math.max(...rows.map((r) => r.high), 0);
  const rawMin = Math.min(...rows.map((r) => r.low), 0);
  const pad = (rawMax - rawMin) * 0.06;
  const min = rawMin - pad;
  const max = rawMax + pad;
  const pct = (v: number) => ((v - min) / (max - min)) * 100;
  const zero = pct(0);

  return (
    <figure className="m-0">
      <div className="grid grid-cols-[10.5rem_1fr_5.5rem] items-center gap-3 pb-2">
        <span className="text-[11.5px] font-medium text-info">{leader}</span>
        <div className="relative h-3">
          <div
            className="absolute top-0 bottom-0 w-px bg-info/60"
            style={{ left: `${zero}%` }}
          />
          <span
            className="absolute top-1/2 -translate-y-1/2 text-[9.5px] font-mono text-info whitespace-nowrap"
            style={{ left: `calc(${zero}% + 6px)` }}
          >
            leader
          </span>
        </div>
        <span className="text-right text-[10.5px] font-mono text-ink-3">—</span>
      </div>

      <div className="space-y-px">
        {rows.map((r) => {
          const crossesZero = r.low <= 0 && r.high >= 0;
          const confirmed = r.separable === true;
          return (
            <div
              key={r.name}
              className="grid grid-cols-[10.5rem_1fr_5.5rem] items-center gap-3 py-1.5"
              title={
                r.p == null
                  ? undefined
                  : `McNemar p=${r.p.toFixed(4)} · ${r.disagree?.toLocaleString()} disagreeing flows`
              }
            >
              <span
                className={clsx(
                  "text-[11.5px] truncate",
                  confirmed ? "text-ink-1" : "text-ink-2",
                )}
              >
                {r.name}
              </span>

              <div className="relative h-4">
                <div className="absolute inset-x-0 top-1/2 h-px -translate-y-1/2 bg-line-subtle" />
                <div
                  className="absolute top-0 bottom-0 w-px bg-info/30"
                  style={{ left: `${zero}%` }}
                />
                <div
                  className={clsx(
                    "absolute top-1/2 h-[7px] -translate-y-1/2 rounded-[1px] transition-[left,width] duration-200 ease-out",
                    confirmed ? "bg-ink-1/50" : "bg-warn/30",
                  )}
                  style={{
                    left: `${pct(r.low)}%`,
                    width: `${Math.max(pct(r.high) - pct(r.low), 0.4)}%`,
                  }}
                />
                <div
                  className={clsx(
                    "absolute top-1/2 h-[11px] w-[2.5px] -translate-x-1/2 -translate-y-1/2 rounded-[1px] transition-[left] duration-200 ease-out",
                    confirmed ? "bg-ink-0" : "bg-warn",
                  )}
                  style={{ left: `${pct(r.delta)}%` }}
                  aria-hidden
                />
              </div>

              <span
                className={clsx(
                  "text-right text-[11px] font-mono tabular-nums",
                  confirmed ? "text-ink-1" : "text-warn",
                )}
              >
                {crossesZero ? "not sep." : `−${r.delta.toFixed(4)}`}
              </span>
            </div>
          );
        })}
      </div>

      <figcaption className="mt-3 text-[11px] text-ink-2 leading-relaxed max-w-[62ch]">
        Distance behind <span className="text-info">{leader}</span> in macro-F1, with
        the 95% interval on that difference. Bars that touch the leader line are gaps
        this test set cannot confirm — those models are not measurably worse. Hover a
        row for its McNemar p-value and the number of flows the two models disagree on.
      </figcaption>
    </figure>
  );
}
