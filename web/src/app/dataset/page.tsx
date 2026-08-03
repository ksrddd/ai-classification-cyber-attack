"use client";

/**
 * Dataset — the class distribution of the selected bundle's test split.
 *
 * The per-class support is the single most load-bearing number on both
 * datasets: three CSE-CIC-IDS2018 classes have 1-3 test flows, and two
 * CICIDS2017 classes have 4 and 11, and no F1 computed on those means
 * anything. Support is therefore shown next to every class, and classes too
 * small to support inference are flagged rather than silently listed.
 */

import { AppShell } from "@/components/shell/AppShell";
import { Panel, PanelHeader } from "@/components/ui/Panel";
import { KpiCard } from "@/components/ui/KpiCard";
import { Nil } from "@/components/ui/Nil";
import { BundleGate } from "@/components/bundle/BundleGate";
import { classColor } from "@/lib/colors";
import { type BundleDetail, count, isAbsent, score } from "@/lib/bundles";

/** Below this many test flows, a per-class score carries no information. */
const UNRELIABLE_SUPPORT = 30;

export default function DatasetPage() {
  return (
    <AppShell title="Dataset">
      <BundleGate what="the dataset view">{(data) => <Body data={data} />}</BundleGate>
    </AppShell>
  );
}

function Body({ data }: { data: BundleDetail }) {
  const supports = data.run.per_class_n_test ?? {};
  const total = Object.values(supports).reduce((a, b) => a + b, 0);
  const rows = data.classes
    .map((name) => ({ name, support: supports[name] ?? null }))
    .sort((a, b) => (b.support ?? -1) - (a.support ?? -1));
  const unreliable = rows.filter(
    (r) => r.support != null && r.support < UNRELIABLE_SUPPORT,
  );

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <KpiCard label="Classes" value={data.classes.length} sub={data.dataset} />
        <KpiCard label="Test flows" value={count(total || data.run.n_test)} />
        <KpiCard label="Train flows" value={count(data.run.n_train)} />
        <KpiCard
          label="Below reliable support"
          value={unreliable.length}
          sub={`fewer than ${UNRELIABLE_SUPPORT} test flows`}
          color={unreliable.length ? "#E0A020" : undefined}
        />
      </div>

      {unreliable.length > 0 && (
        <Panel className="p-4">
          <h3 className="text-[12.5px] font-semibold text-warn">
            {unreliable.length} class{unreliable.length > 1 ? "es" : ""} cannot support
            inference
          </h3>
          <p className="mt-1.5 text-[11.5px] text-ink-2 leading-relaxed max-w-prose">
            <span className="font-mono text-ink-1">
              {unreliable.map((r) => `${r.name} (${r.support})`).join(", ")}
            </span>
            . A single prediction moves F1 by a large fraction on these, so their scores
            describe the sample, not the model. Report them as insufficient support rather
            than as a result.
          </p>
        </Panel>
      )}

      <Panel>
        <PanelHeader
          eyebrow={data.id}
          title="Class distribution — test split"
          sub={
            data.run.split_protocol ?? "this bundle does not record a split protocol"
          }
        />
        <div className="px-3 pb-3 space-y-1">
          {rows.map(({ name, support }) => {
            const share = support != null && total ? support / total : null;
            const weak = support != null && support < UNRELIABLE_SUPPORT;
            return (
              <div key={name} className="flex items-center gap-3 text-[11.5px]">
                <span
                  className="h-2 w-2 rounded-full flex-shrink-0"
                  style={{ background: classColor(name) }}
                />
                <span className="w-52 truncate text-ink-0">{name}</span>
                <div className="flex-1 h-1.5 rounded-sm bg-surface overflow-hidden">
                  {share != null && (
                    <div
                      className="h-full rounded-sm"
                      style={{
                        width: `${Math.max(share * 100, 0.4)}%`,
                        background: classColor(name),
                      }}
                    />
                  )}
                </div>
                <span className="w-20 text-right font-mono tabular-nums text-ink-1">
                  {isAbsent(support) ? <Nil /> : count(support)}
                </span>
                <span className="w-16 text-right font-mono text-[10.5px] text-ink-3">
                  {share == null ? <Nil /> : `${(share * 100).toFixed(2)}%`}
                </span>
                {weak && (
                  <span className="text-[9.5px] font-mono uppercase text-warn">low</span>
                )}
              </div>
            );
          })}
        </div>
      </Panel>

      {(data.run.dropped_columns?.length ?? 0) > 0 && (
        <Panel>
          <PanelHeader
            title="Columns removed before training"
            sub="Leakage, identifiers, and features with no variance on the training split."
          />
          <div className="px-3 pb-3 flex flex-wrap gap-1.5">
            {data.run.dropped_columns!.map((c) => (
              <span
                key={c}
                className="px-2 py-1 rounded-sm text-[10.5px] font-mono border border-line-base bg-surface text-ink-2"
              >
                {c}
              </span>
            ))}
          </div>
        </Panel>
      )}
    </div>
  );
}
