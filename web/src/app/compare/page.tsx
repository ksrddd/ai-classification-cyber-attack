"use client";

/**
 * Comparison — the page the whole dashboard exists for.
 *
 * The earlier version was a numbered 1–7 table inside a bordered panel under a
 * dataset eyebrow. That was scaffolding, not design, and worse: the numbering
 * asserted a ranking the data does not support. On the 2018 bundle the top
 * four models sit within 0.003 macro-F1 and no paired test separates them.
 *
 * So the page states the finding in a sentence, draws the gaps behind the
 * leader with the interval on each gap, and only then offers the full table.
 * Nothing is wrapped in a box it does not need.
 */

import { useEffect, useState } from "react";
import { AppShell } from "@/components/shell/AppShell";
import { Nil } from "@/components/ui/Nil";
import { BundleGate } from "@/components/bundle/BundleGate";
import { LeaderGap, buildGaps } from "@/components/compare/LeaderGap";
import {
  type BundleDetail,
  type BundleMetrics,
  count,
  duration,
  isAbsent,
  percent,
  rankBy,
  score,
} from "@/lib/bundles";

const COLUMNS: {
  key: keyof BundleMetrics;
  label: string;
  fmt: (v: never) => string;
  sortKey?: boolean;
}[] = [
  { key: "f1_macro", label: "F1 macro", fmt: (v) => score(v, 4), sortKey: true },
  { key: "accuracy", label: "Accuracy", fmt: (v) => score(v, 4) },
  { key: "recall_macro", label: "Recall macro", fmt: (v) => score(v, 4) },
  { key: "mcc", label: "MCC", fmt: (v) => score(v, 4) },
  { key: "binary_fpr", label: "FPR", fmt: (v) => percent(v, 3) },
  { key: "false_alarms_fp", label: "False alarms", fmt: (v) => count(v) },
  { key: "missed_attacks_fn", label: "Missed", fmt: (v) => count(v) },
  { key: "train_seconds", label: "Train", fmt: (v) => duration(v) },
];

export default function ComparePage() {
  return (
    <AppShell title="Comparison">
      <BundleGate what="the comparison">{(data) => <Body data={data} />}</BundleGate>
    </AppShell>
  );
}

function Body({ data }: { data: BundleDetail }) {
  const { ranked, unmeasured } = rankBy(data.models, "f1_macro");
  const leader = ranked[0]?.[0];
  const gaps = buildGaps(data.significance ?? {}, Object.keys(data.models));
  const inconclusive = gaps.filter((g) => g.separable !== true).length;

  const emptyColumns = COLUMNS.filter(({ key }) =>
    Object.values(data.models).every((m) => isAbsent(m[key])),
  );

  return (
    <div className="max-w-[68rem] space-y-9">
      <Lede data={data} leader={leader} gaps={gaps.length} inconclusive={inconclusive} />

      {gaps.length > 0 && leader && (
        <section>
          <h2 className="text-[12.5px] font-semibold text-ink-0">
            Distance behind the leader
          </h2>
          <div className="mt-3">
            <LeaderGap leader={leader} rows={gaps} />
          </div>
        </section>
      )}

      <section>
        <div className="flex items-baseline justify-between gap-4 border-b border-line-base pb-1.5">
          <h2 className="text-[12.5px] font-semibold text-ink-0">Every recorded metric</h2>
          <span className="text-[10.5px] font-mono text-ink-3">
            {ranked.length + unmeasured.length} models
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-[11.5px]">
            <caption className="sr-only">
              Model metrics for bundle {data.id}. Dashes mark values this bundle did not
              record.
            </caption>
            <thead>
              <tr className="text-ink-3">
                <th scope="col" className="text-left font-medium py-2 pr-3">
                  Model
                </th>
                {COLUMNS.map((c) => (
                  <th
                    key={String(c.key)}
                    scope="col"
                    className={
                      "text-right font-medium py-2 pl-3 whitespace-nowrap " +
                      (c.sortKey ? "text-ink-1" : "")
                    }
                  >
                    {c.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ranked.map(([name, m]) => (
                <tr
                  key={name}
                  className="border-t border-line-subtle hover:bg-surface-raised/60"
                >
                  <th
                    scope="row"
                    className={
                      "text-left font-medium py-2 pr-3 whitespace-nowrap " +
                      (name === leader ? "text-info" : "text-ink-2")
                    }
                  >
                    {name}
                  </th>
                  {COLUMNS.map((c) => {
                    const raw = m[c.key];
                    return (
                      <td
                        key={String(c.key)}
                        className="py-2 pl-3 text-right font-mono tabular-nums text-ink-1"
                      >
                        {isAbsent(raw) ? (
                          <Nil reason={`${data.id} does not record ${String(c.key)}`} />
                        ) : (
                          (c.fmt as (v: unknown) => string)(raw)
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {unmeasured.length > 0 && (
          <p className="mt-3 text-[11px] leading-relaxed text-ink-2 max-w-[62ch]">
            <span className="font-mono text-ink-1">{unmeasured.join(", ")}</span>{" "}
            {unmeasured.length > 1 ? "are" : "is"} absent from the sort: this bundle
            records no macro-F1 for {unmeasured.length > 1 ? "them" : "it"}. Excluded
            rather than placed last, because an unmeasured model is not a bad one.
          </p>
        )}

        {emptyColumns.length > 0 && (
          <p className="mt-2 text-[11px] leading-relaxed text-ink-2 max-w-[62ch]">
            Empty for every model here:{" "}
            <span className="font-mono text-ink-1">
              {emptyColumns.map((c) => c.label).join(", ")}
            </span>
            . The other pipeline records {emptyColumns.length > 1 ? "these" : "this"};
            this one never did.
          </p>
        )}
      </section>
    </div>
  );
}

/**
 * The finding, in a sentence. Replaces the decorative eyebrow the old layout
 * carried: the dataset name is already in the rail, and a label is not a
 * finding.
 */
function Lede({
  data,
  leader,
  gaps,
  inconclusive,
}: {
  data: BundleDetail;
  leader?: string;
  gaps: number;
  inconclusive: number;
}) {
  const baseline = data.run.majority_baseline_acc;
  const random = data.run.split_protocol?.includes("random");

  return (
    <section className="max-w-[62ch] space-y-2">
      <p className="text-[15px] leading-snug text-ink-0 text-balance">
        {gaps === 0 ? (
          <>
            <strong className="font-semibold">
              This run never tested whether the gaps are real.
            </strong>{" "}
            The order below is point estimates alone — the distance between first and
            second may be noise, and nothing here can tell you.
          </>
        ) : inconclusive > 0 ? (
          <>
            <strong className="font-semibold">
              {inconclusive} of {gaps} models cannot be separated from {leader}.
            </strong>{" "}
            Their gaps are inside the margin of error, so the order below is a grouping,
            not a ranking.
          </>
        ) : (
          <>
            <strong className="font-semibold">{leader} leads measurably.</strong> Every
            other model sits outside the margin of error.
          </>
        )}
      </p>

      <p className="text-[11.5px] leading-relaxed text-ink-2">
        {data.dataset} · {data.run.split_protocol ?? "split protocol not recorded"} ·{" "}
        {count(data.run.n_test)} test flows · {data.run.n_classes ?? "?"} classes
        {!isAbsent(baseline) && (
          <> · predicting only the largest class scores {score(baseline, 4)}</>
        )}
      </p>

      {random && (
        <p className="text-[11.5px] leading-relaxed text-warn max-w-[62ch]">
          Split is random, not chronological: flows from one attack burst can land on
          both sides of it, so every score here is an upper bound.
        </p>
      )}
    </section>
  );
}
