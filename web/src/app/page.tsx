"use client";

/**
 * Overview — the headline numbers for whichever bundle is selected.
 *
 * Leads with macro F1 and the majority-class baseline rather than accuracy,
 * because on both datasets the majority class is over 80% of the test split
 * and accuracy alone cannot distinguish a working detector from a constant.
 */

import { AppShell } from "@/components/shell/AppShell";
import { KpiCard } from "@/components/ui/KpiCard";
import { Panel, PanelHeader } from "@/components/ui/Panel";
import { Nil } from "@/components/ui/Nil";
import { BundleGate } from "@/components/bundle/BundleGate";
import { modelColor } from "@/lib/colors";
import {
  type BundleDetail,
  count,
  isAbsent,
  percent,
  rankBy,
  score,
} from "@/lib/bundles";

export default function OverviewPage() {
  return (
    <AppShell title="Overview">
      <BundleGate what="the overview">{(data) => <Body data={data} />}</BundleGate>
    </AppShell>
  );
}

function Body({ data }: { data: BundleDetail }) {
  const { ranked } = rankBy(data.models, "f1_macro");
  const best = ranked[0];
  const baseline = data.run.majority_baseline_acc;
  const lowestFpr = rankBy(data.models, "binary_fpr", false).ranked[0];

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <KpiCard
          label="Dataset"
          value={<span className="text-[15px]">{data.dataset}</span>}
          sub={`${data.run.n_classes ?? "?"} classes · ${data.run.n_features ?? "?"} features`}
        />
        <KpiCard
          label="Best F1 macro"
          value={best ? score(best[1].f1_macro, 4) : <Nil />}
          sub={best ? best[0] : "no model records F1 macro"}
          color={best ? modelColor(best[0]) : undefined}
        />
        <KpiCard
          label="Majority baseline"
          value={isAbsent(baseline) ? <Nil /> : score(baseline, 4)}
          sub="accuracy of always predicting the largest class"
        />
        <KpiCard
          label="Lowest false-alarm rate"
          value={lowestFpr ? percent(lowestFpr[1].binary_fpr, 3) : <Nil />}
          sub={lowestFpr ? lowestFpr[0] : "not recorded in this bundle"}
        />
      </div>

      <Panel>
        <PanelHeader
          eyebrow={data.id}
          title="How this run was produced"
          sub="Provenance decides how the numbers may be read, so it sits above them."
        />
        <dl className="grid sm:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-2 px-3 pb-3 text-[11.5px]">
          <Row k="Split protocol" v={data.run.split_protocol} />
          <Row k="Train rows" v={count(data.run.n_train)} />
          <Row k="Test rows" v={count(data.run.n_test)} />
          <Row k="Random seed" v={data.run.random_state} />
          <Row k="Class weighting" v={data.run.class_weighting} />
          <Row
            k="Hyperparameter search"
            v={data.run.hp_tuned == null ? null : data.run.hp_tuned ? "yes" : "no"}
          />
        </dl>
        {data.run.split_protocol?.includes("random") && (
          <p className="px-3 pb-3 text-[11px] text-warn leading-relaxed max-w-prose">
            This run used a random stratified split. Flows from the same attack burst can
            land on both sides of it, so every score below should be read as an upper bound
            rather than as generalisation to future traffic.
          </p>
        )}
      </Panel>

      <Panel>
        <PanelHeader
          title="Models in this bundle"
          sub="Ranked by F1 macro. Models the bundle never scored are not ranked."
        />
        <div className="px-3 pb-3 space-y-1.5">
          {ranked.map(([name, m]) => (
            <div key={name} className="flex items-center gap-3 text-[11.5px]">
              <span
                className="h-2 w-2 rounded-full flex-shrink-0"
                style={{ background: modelColor(name) }}
              />
              <span className="w-40 text-ink-0 truncate">{name}</span>
              <span className="font-mono tabular-nums text-ink-1">
                {score(m.f1_macro, 4)}
              </span>
              <span className="text-ink-3 font-mono text-[10.5px]">
                acc {score(m.accuracy, 4)}
              </span>
              {!isAbsent(m.f1_macro_ci_low) && (
                <span className="text-ink-3 font-mono text-[10.5px]">
                  95% CI [{score(m.f1_macro_ci_low, 4)}, {score(m.f1_macro_ci_high, 4)}]
                </span>
              )}
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-3 border-b border-line-subtle py-1">
      <dt className="text-ink-3">{k}</dt>
      <dd className="font-mono text-ink-1">{isAbsent(v) || v === "" ? <Nil /> : v}</dd>
    </div>
  );
}
