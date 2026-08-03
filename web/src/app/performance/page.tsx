"use client";

/**
 * Model detail — one model from the selected bundle.
 *
 * Shows the per-class table and, where the bundle stores one, the confusion
 * matrix. The matrix is row-normalised because raw counts make every row
 * except the majority class invisible: on CSE-CIC-IDS2018 the benign class is
 * 83% of the test split.
 */

import { useEffect, useState } from "react";
import { AppShell } from "@/components/shell/AppShell";
import { Panel, PanelHeader } from "@/components/ui/Panel";
import { KpiCard } from "@/components/ui/KpiCard";
import { Nil } from "@/components/ui/Nil";
import { BundleGate, Empty } from "@/components/bundle/BundleGate";
import { useBundle } from "@/components/bundle/BundleProvider";
import {
  type BundleDetail,
  type PerClassRow,
  count,
  duration,
  getBundleConfusion,
  getBundleReport,
  isAbsent,
  percent,
  score,
} from "@/lib/bundles";

/** Below this many test flows, a per-class score carries no information. */
const UNRELIABLE_SUPPORT = 30;

export default function PerformancePage() {
  return (
    <AppShell title="Model detail">
      <BundleGate what="model detail">{(data) => <Body data={data} />}</BundleGate>
    </AppShell>
  );
}

function Body({ data }: { data: BundleDetail }) {
  const { activeId } = useBundle();
  const names = Object.keys(data.models).sort();
  const [selected, setSelected] = useState(names[0] ?? "");
  const [report, setReport] = useState<PerClassRow[] | null>(null);
  const [confusion, setConfusion] = useState<{ labels: string[]; rows: number[][] } | null>(
    null,
  );
  const [noConfusion, setNoConfusion] = useState(false);

  // Reset when the bundle changes — the previously selected model may not
  // exist in the new one.
  useEffect(() => {
    if (!names.includes(selected)) setSelected(names[0] ?? "");
  }, [names, selected]);

  useEffect(() => {
    if (!selected || !activeId) return;
    let cancelled = false;
    setReport(null);
    setConfusion(null);
    setNoConfusion(false);
    getBundleReport(selected, activeId)
      .then((r) => !cancelled && setReport(r.rows))
      .catch(() => !cancelled && setReport([]));
    getBundleConfusion(selected, activeId)
      .then((c) => !cancelled && setConfusion({ labels: c.labels, rows: c.rows }))
      .catch(() => !cancelled && setNoConfusion(true));
    return () => {
      cancelled = true;
    };
  }, [selected, activeId]);

  if (!names.length) {
    return <Empty title="This bundle contains no models">Nothing to show yet.</Empty>;
  }

  const m = data.models[selected];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-1.5">
        {names.map((n) => (
          <button
            key={n}
            onClick={() => setSelected(n)}
            className={
              "px-2.5 py-1 rounded-sm text-[11px] font-mono border transition-colors " +
              (n === selected
                ? "bg-surface-elevated text-ink-0 border-info"
                : "bg-surface text-ink-2 border-line-base hover:text-ink-1")
            }
          >
            {n}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <KpiCard label="Accuracy" value={score(m?.accuracy, 4)} />
        <KpiCard label="F1 macro" value={score(m?.f1_macro, 4)} />
        <KpiCard
          label="Binary FPR"
          value={isAbsent(m?.binary_fpr) ? <Nil /> : percent(m.binary_fpr, 3)}
          sub={
            isAbsent(m?.false_alarms_fp)
              ? undefined
              : `${count(m.false_alarms_fp)} false alarms`
          }
        />
        <KpiCard
          label="Train time"
          value={isAbsent(m?.train_seconds) ? <Nil /> : duration(m.train_seconds)}
          sub={m?.accelerator ?? undefined}
        />
      </div>

      <Panel>
        <PanelHeader
          eyebrow={selected}
          title="Per-class results"
          sub="Support sits next to every score, because a score computed on a handful of flows is not a measurement."
        />
        {report === null ? (
          <div className="p-4 space-y-1.5">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="h-6 rounded-sm bg-surface animate-pulse" />
            ))}
          </div>
        ) : report.length === 0 ? (
          <p className="px-3 pb-3 text-[11.5px] text-ink-2">
            This bundle stores no per-class report for {selected}.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-[11.5px]">
              <thead>
                <tr className="text-ink-3 border-b border-line-base">
                  <th className="text-left font-medium px-3 py-2">Class</th>
                  <th className="text-right font-medium px-3 py-2">Precision</th>
                  <th className="text-right font-medium px-3 py-2">Recall</th>
                  <th className="text-right font-medium px-3 py-2">F1</th>
                  <th className="text-right font-medium px-3 py-2">Support</th>
                </tr>
              </thead>
              <tbody>
                {report.map((r) => {
                  const weak = r.support < UNRELIABLE_SUPPORT;
                  return (
                    <tr key={r.class} className="border-b border-line-subtle">
                      <td className="px-3 py-1.5 text-ink-0">{r.class}</td>
                      <td className="px-3 py-1.5 text-right font-mono tabular-nums text-ink-1">
                        {score(r.precision, 4)}
                      </td>
                      <td className="px-3 py-1.5 text-right font-mono tabular-nums text-ink-1">
                        {score(r.recall, 4)}
                      </td>
                      <td className="px-3 py-1.5 text-right font-mono tabular-nums text-ink-1">
                        {score(r["f1-score"], 4)}
                      </td>
                      <td
                        className={
                          "px-3 py-1.5 text-right font-mono tabular-nums " +
                          (weak ? "text-warn" : "text-ink-2")
                        }
                        title={
                          weak
                            ? "Too few test flows for this score to mean anything"
                            : undefined
                        }
                      >
                        {count(r.support)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <Panel>
        <PanelHeader
          title="Confusion matrix"
          sub="Row-normalised: each row shows where that class went, as a percentage of its own support."
        />
        {noConfusion ? (
          <p className="px-3 pb-3 text-[11.5px] text-ink-2 max-w-prose leading-relaxed">
            Bundle <span className="font-mono text-ink-1">{data.id}</span> does not store a
            confusion matrix as data. The CICIDS2017 pipeline saves one as a PNG figure
            instead, so it cannot be re-rendered here.
          </p>
        ) : confusion === null ? (
          <div className="h-40 m-3 rounded-sm bg-surface animate-pulse" />
        ) : (
          <ConfusionGrid labels={confusion.labels} rows={confusion.rows} />
        )}
      </Panel>
    </div>
  );
}

function ConfusionGrid({ labels, rows }: { labels: string[]; rows: number[][] }) {
  return (
    <div className="overflow-x-auto px-3 pb-3">
      <table className="text-[10px] border-collapse">
        <thead>
          <tr>
            <th className="p-1" />
            {labels.map((l) => (
              <th key={l} className="p-1 text-ink-3 font-normal align-bottom h-24" title={l}>
                <div className="[writing-mode:vertical-rl] rotate-180 whitespace-nowrap">
                  {l}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => {
            const total = row.reduce((a, b) => a + b, 0);
            return (
              <tr key={labels[i]}>
                <td className="p-1 pr-2 text-ink-2 whitespace-nowrap text-right">
                  {labels[i]}
                </td>
                {row.map((cell, j) => {
                  const frac = total ? cell / total : 0;
                  return (
                    <td
                      key={j}
                      title={`${labels[i]} to ${labels[j]}: ${cell.toLocaleString()} (${(
                        frac * 100
                      ).toFixed(1)}%)`}
                      className="w-7 h-7 text-center font-mono border border-line-subtle"
                      style={{
                        background: `rgba(42,184,216,${frac.toFixed(3)})`,
                        color: frac > 0.5 ? "#080D17" : undefined,
                      }}
                    >
                      {frac >= 0.005 ? (frac * 100).toFixed(0) : ""}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
