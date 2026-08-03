"use client";

/**
 * Model comparison for whichever bundle is selected.
 *
 * Rewritten against the bundle API so it renders CICIDS2017 and
 * CSE-CIC-IDS2018 with the same code. Two rules from the redesign mockup are
 * load-bearing here:
 *
 *   1. A metric the bundle never recorded shows as absent, never as 0. The
 *      two pipelines record different subsets, so any column can be empty.
 *   2. Models missing the sort metric are listed separately rather than sunk
 *      to the bottom -- "not measured" is not "worst".
 */

import { useEffect, useState } from "react";
import { AppShell } from "@/components/shell/AppShell";
import { Panel, PanelHeader } from "@/components/ui/Panel";
import { Nil } from "@/components/ui/Nil";
import { useBundle } from "@/components/bundle/BundleProvider";
import {
  type BundleDetail,
  type BundleMetrics,
  count,
  duration,
  getBundle,
  isAbsent,
  percent,
  rankBy,
  score,
} from "@/lib/bundles";

const COLUMNS: {
  key: keyof BundleMetrics;
  label: string;
  fmt: (v: never) => string;
}[] = [
  { key: "accuracy", label: "Accuracy", fmt: (v) => score(v, 4) },
  { key: "f1_macro", label: "F1 macro", fmt: (v) => score(v, 4) },
  { key: "f1_weighted", label: "F1 weighted", fmt: (v) => score(v, 4) },
  { key: "recall_macro", label: "Recall macro", fmt: (v) => score(v, 4) },
  { key: "mcc", label: "MCC", fmt: (v) => score(v, 4) },
  { key: "binary_fpr", label: "Binary FPR", fmt: (v) => percent(v, 3) },
  { key: "false_alarms_fp", label: "False alarms", fmt: (v) => count(v) },
  { key: "missed_attacks_fn", label: "Missed attacks", fmt: (v) => count(v) },
  { key: "train_seconds", label: "Train", fmt: (v) => duration(v) },
];

export default function ComparePage() {
  const { activeId, active } = useBundle();
  const [data, setData] = useState<BundleDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!activeId) return;
    let cancelled = false;
    setData(null);
    setError(null);
    getBundle(activeId)
      .then((d) => !cancelled && setData(d))
      .catch((e: Error) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [activeId]);

  return (
    <AppShell title="Comparison">
      {!activeId && !error && <Empty title="No results bundle is loaded" />}
      {error && <Empty title="Could not read this bundle" detail={error} />}
      {activeId && !data && !error && <Skeleton />}
      {data && <CompareBody data={data} datasetLabel={active?.dataset ?? data.dataset} />}
    </AppShell>
  );
}

function CompareBody({
  data,
  datasetLabel,
}: {
  data: BundleDetail;
  datasetLabel: string;
}) {
  const { ranked, unmeasured } = rankBy(data.models, "f1_macro");
  const baseline = data.run.majority_baseline_acc;

  // Columns this bundle has nothing for at all -- worth saying once at the
  // top instead of leaving the reader to infer it from a wall of dashes.
  const emptyColumns = COLUMNS.filter(({ key }) =>
    Object.values(data.models).every((m) => isAbsent(m[key])),
  );

  return (
    <div className="space-y-4">
      <Panel>
        <PanelHeader
          eyebrow={datasetLabel}
          title={`${ranked.length + unmeasured.length} models · ${
            data.run.n_classes ?? "?"
          } classes`}
          sub={
            <>
              {data.run.split_protocol ?? "split protocol not recorded"} ·{" "}
              {count(data.run.n_test)} test flows
              {!isAbsent(baseline) && <> · majority baseline {score(baseline, 4)}</>}
            </>
          }
        />

        <div className="overflow-x-auto">
          <table className="w-full text-[11.5px]">
            <thead>
              <tr className="text-ink-3 border-b border-line-base">
                <th className="text-left font-medium px-3 py-2">Model</th>
                {COLUMNS.map((c) => (
                  <th
                    key={String(c.key)}
                    className="text-right font-medium px-3 py-2 whitespace-nowrap"
                  >
                    {c.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ranked.map(([name, m], i) => (
                <tr key={name} className="border-b border-line-subtle hover:bg-surface-hover">
                  <td className="px-3 py-2 text-ink-0 font-medium whitespace-nowrap">
                    <span className="text-ink-3 font-mono mr-2">{i + 1}</span>
                    {name}
                  </td>
                  {COLUMNS.map((c) => {
                    const raw = m[c.key];
                    return (
                      <td
                        key={String(c.key)}
                        className="px-3 py-2 text-right font-mono tabular-nums text-ink-1"
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
          <div className="px-3 py-2.5 border-t border-line-base text-[11px] text-ink-2 leading-relaxed">
            Not ranked — this bundle records no F1 macro for{" "}
            <span className="font-mono text-ink-1">{unmeasured.join(", ")}</span>. They are
            excluded rather than placed last, because an unmeasured model is not a bad one.
          </div>
        )}
      </Panel>

      {emptyColumns.length > 0 && (
        <Panel>
          <PanelHeader
            title="Columns this bundle cannot fill"
            sub="Recorded by the other pipeline, absent here. Shown as dashes, never as zeros."
          />
          <div className="px-3 pb-3 flex flex-wrap gap-1.5">
            {emptyColumns.map((c) => (
              <span
                key={String(c.key)}
                className="px-2 py-1 rounded-sm text-[10.5px] font-mono border border-line-base bg-surface text-ink-2"
              >
                {c.label}
              </span>
            ))}
          </div>
        </Panel>
      )}
    </div>
  );
}

function Skeleton() {
  return (
    <div className="space-y-2">
      {[...Array(8)].map((_, i) => (
        <div key={i} className="h-8 rounded-sm bg-surface-raised animate-pulse" />
      ))}
    </div>
  );
}

function Empty({ title, detail }: { title: string; detail?: string }) {
  return (
    <Panel className="p-6">
      <h3 className="text-[13px] font-semibold text-ink-0">{title}</h3>
      <p className="mt-1.5 text-[11.5px] text-ink-2 leading-relaxed max-w-prose">
        {detail ??
          "Train a run, or pick a different bundle in the rail. The dashboard reads whatever is under results/."}
      </p>
    </Panel>
  );
}
