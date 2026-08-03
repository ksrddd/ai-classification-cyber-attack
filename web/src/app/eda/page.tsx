"use client";

/**
 * Distributions — train vs test balance for the selected bundle.
 *
 * The EDA figures under results/figures/ were produced by the CICIDS2017
 * pipeline only. Rather than show 2017 plots while a 2018 run is selected —
 * which would be actively misleading — the figure panel is gated on the
 * bundle, and the class balance table below is computed from whichever
 * bundle is active.
 */

import { AppShell } from "@/components/shell/AppShell";
import { Panel, PanelHeader } from "@/components/ui/Panel";
import { FigureImg } from "@/components/ui/FigureImg";
import { Nil } from "@/components/ui/Nil";
import { BundleGate, Code } from "@/components/bundle/BundleGate";
import { classColor } from "@/lib/colors";
import { figureUrl } from "@/lib/api";
import { type BundleDetail, count, isAbsent } from "@/lib/bundles";

const FIGURES = [
  { key: "class_distribution", label: "Class distribution" },
  { key: "correlation_heatmap", label: "Feature correlation" },
  { key: "feature_importance", label: "Feature importance" },
];

export default function EdaPage() {
  return (
    <AppShell title="Distributions">
      <BundleGate what="distributions">{(data) => <Body data={data} />}</BundleGate>
    </AppShell>
  );
}

function Body({ data }: { data: BundleDetail }) {
  const train = data.run.per_class_n_train ?? {};
  const test = data.run.per_class_n_test ?? {};
  const hasTrain = Object.keys(train).length > 0;

  const rows = data.classes.map((name) => ({
    name,
    train: train[name] ?? null,
    test: test[name] ?? null,
  }));
  const trainTotal = Object.values(train).reduce((a, b) => a + b, 0);
  const testTotal = Object.values(test).reduce((a, b) => a + b, 0);

  return (
    <div className="space-y-4">
      <Panel>
        <PanelHeader
          eyebrow={data.id}
          title="Class balance across the split"
          sub={
            hasTrain
              ? "Per-class counts on both sides, as recorded by the run."
              : "This bundle records per-class counts for the test split only."
          }
        />
        <div className="overflow-x-auto">
          <table className="w-full text-[11.5px]">
            <thead>
              <tr className="text-ink-3 border-b border-line-base">
                <th className="text-left font-medium px-3 py-2">Class</th>
                <th className="text-right font-medium px-3 py-2">Train</th>
                <th className="text-right font-medium px-3 py-2">Train %</th>
                <th className="text-right font-medium px-3 py-2">Test</th>
                <th className="text-right font-medium px-3 py-2">Test %</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.name} className="border-b border-line-subtle">
                  <td className="px-3 py-1.5 text-ink-0">
                    <span
                      className="inline-block h-2 w-2 rounded-full mr-2 align-middle"
                      style={{ background: classColor(r.name) }}
                    />
                    {r.name}
                  </td>
                  <Cell value={r.train} total={trainTotal} />
                  <Cell value={r.train} total={trainTotal} asPercent />
                  <Cell value={r.test} total={testTotal} />
                  <Cell value={r.test} total={testTotal} asPercent />
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <Panel>
        <PanelHeader
          title="EDA figures"
          sub="Static plots written by the exploratory stage."
        />
        {data.layout === "cicids2017" ? (
          <div className="grid md:grid-cols-2 gap-3 px-3 pb-3">
            {FIGURES.map((f) => (
              <FigureImg
                key={f.key}
                src={figureUrl(`${f.key}.png`)}
                alt={f.label}
                maxH="280px"
              />
            ))}
          </div>
        ) : (
          <p className="px-3 pb-3 text-[11.5px] text-ink-2 max-w-prose leading-relaxed">
            No EDA figures were generated for{" "}
            <span className="font-mono text-ink-1">{data.id}</span>. The plots under{" "}
            <Code>results/figures/</Code> belong to the CICIDS2017 pipeline and describe a
            different corpus with a different class taxonomy, so they are not shown here —
            rendering them against a 2018 run would attach 2017 evidence to 2018 numbers.
          </p>
        )}
      </Panel>
    </div>
  );
}

function Cell({
  value,
  total,
  asPercent = false,
}: {
  value: number | null;
  total: number;
  asPercent?: boolean;
}) {
  if (isAbsent(value)) {
    return (
      <td className="px-3 py-1.5 text-right">
        <Nil />
      </td>
    );
  }
  return (
    <td className="px-3 py-1.5 text-right font-mono tabular-nums text-ink-1">
      {asPercent
        ? total
          ? `${((value / total) * 100).toFixed(2)}%`
          : "—"
        : count(value)}
    </td>
  );
}
