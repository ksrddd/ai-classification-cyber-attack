import { AppShell } from "@/components/shell/AppShell";
import { KpiCard } from "@/components/ui/KpiCard";
import { Panel, PanelHeader } from "@/components/ui/Panel";
import { Pill } from "@/components/ui/Pill";
import { Donut } from "@/components/ui/Donut";
import { getOverview, getModels, getCompare } from "@/lib/api";
import { classColor, modelColor, modelLabel, modelShort } from "@/lib/colors";

export const revalidate = 60;

/* Pre-defined sparkline patterns — decorative trend lines for KPIs */
const SPARK_RECORDS  = [162, 165, 168, 170, 172, 174, 176, 178, 180, 181, 182];
const SPARK_F1       = [0.94, 0.96, 0.968, 0.974, 0.978, 0.982, 0.985, 0.988, 0.990, 0.991, 0.9912];
const SPARK_CLASSES  = [4, 5, 5, 6, 6, 6, 7, 7, 7, 7, 7];
const SPARK_FEATURES = [65, 68, 71, 73, 75, 76, 77, 77, 78, 78, 78];

const MEDAL = ["1st", "2nd", "3rd", "4th", "5th", "6th"];

export default async function OverviewPage() {
  const [overview, modelsData, compare] = await Promise.all([
    getOverview().catch(() => null),
    getModels().catch(() => ({ models: [] as string[] })),
    getCompare().catch(() => null),
  ]);

  const models = modelsData.models;
  const dist = overview?.label_distribution ?? {};
  const totalFlows = Object.values(dist).reduce((a: number, b: number) => a + b, 0);

  /* Champion model */
  let bestModel = "";
  let bestF1 = 0;
  const bestMetrics: Record<string, number> = {};
  if (compare) {
    for (const [name, m] of Object.entries(compare)) {
      const f1 = m.f1_weighted ?? 0;
      if (f1 > bestF1) {
        bestF1 = f1;
        bestModel = name;
        Object.assign(bestMetrics, m);
      }
    }
  }

  /* Class distribution for donut */
  const distEntries = Object.entries(dist).sort(([, a], [, b]) => (b as number) - (a as number));
  const donutData = distEntries.map(([cls, count]) => ({
    name: cls,
    pct: totalFlows > 0 ? ((count as number) / totalFlows) * 100 : 0,
    color: classColor(cls),
  }));

  /* Ranked models */
  const ranked = compare
    ? Object.entries(compare).sort(
        ([, a], [, b]) => (b.f1_weighted ?? 0) - (a.f1_weighted ?? 0),
      )
    : [];

  const bColor = modelColor(bestModel);

  return (
    <AppShell title="Overview">
      <div className="flex flex-col gap-5 animate-fadeIn w-full min-h-full">

        {/* ── Page header ── */}
        <div className="flex items-end justify-between flex-wrap gap-3">
          <div>
            <div className="text-[10.5px] uppercase tracking-[.18em] text-ink-3 font-semibold mb-1">
              KMITL · IT · Senior Project 2569
            </div>
            <h1 className="text-[26px] font-semibold tracking-tight leading-tight text-ink-0">
              Operations overview
            </h1>
            <div className="text-[12px] text-ink-2 mt-1.5 inline-flex items-center gap-3 flex-wrap">
              <span className="inline-flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 rounded-sm bg-ok" />
                {models.length} models trained
              </span>
              <span className="text-ink-3">·</span>
              <span>{overview?.mode?.toUpperCase() ?? "—"} classification</span>
              <span className="text-ink-3">·</span>
              <span className="tabular-nums">{totalFlows.toLocaleString()} flow records</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Pill tone="ok">CICIDS2017/CSE-CIC-IDS2018 ready</Pill>
            <Pill tone="brand">multiclass</Pill>
          </div>
        </div>

        {/* ── Champion strip ── */}
        {bestModel && (
          <div className="border border-line-base rounded bg-surface-raised"
            style={{ borderLeftColor: bColor, borderLeftWidth: "3px" }}>
            <div className="grid grid-cols-1 md:grid-cols-3 divide-y md:divide-y-0 md:divide-x divide-line-subtle">
              {/* Model */}
              <div className="px-4 py-3 flex items-center gap-3">
                <div
                  className="h-9 w-9 flex-shrink-0 border rounded-sm grid place-items-center font-mono text-[11px] font-semibold"
                  style={{ borderColor: `${bColor}55`, color: bColor, background: `${bColor}12` }}
                >
                  {modelShort(bestModel)}
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-[13.5px] font-semibold text-ink-0">{modelLabel(bestModel)}</span>
                    <Pill tone="brand" size="sm">champion</Pill>
                  </div>
                  <div className="text-[10.5px] text-ink-3 font-mono mt-0.5">best on holdout test set</div>
                </div>
              </div>

              {/* Metrics */}
              <div className="px-4 py-3">
                <div className="text-[9.5px] uppercase tracking-[.16em] text-ink-3 mb-2 font-semibold">Performance</div>
                <div className="grid grid-cols-3 gap-4">
                  {[
                    { k: "f1_weighted",       l: "F1 (W)" },
                    { k: "accuracy",          l: "Accuracy" },
                    { k: "matthews_corrcoef", l: "MCC" },
                  ].map(({ k, l }) => (
                    <div key={k}>
                      <div className="text-[9.5px] text-ink-3 font-mono uppercase">{l}</div>
                      <div className="tabular-nums text-[15px] font-semibold font-mono text-ink-0 mt-0.5">
                        {(bestMetrics[k] ?? 0).toFixed(4)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Classes + rows */}
              <div className="px-4 py-3">
                <div className="text-[9.5px] uppercase tracking-[.16em] text-ink-3 mb-2 font-semibold">Attack classes</div>
                <div className="flex flex-wrap gap-1 mb-3">
                  {(overview?.labels ?? []).map((lb) => (
                    <span
                      key={lb}
                      className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-sm text-[10.5px] font-medium border"
                      style={{
                        borderColor: `${classColor(lb)}40`,
                        color: classColor(lb),
                        background: `${classColor(lb)}10`,
                      }}
                    >
                      {lb}
                    </span>
                  ))}
                </div>
                <div className="flex gap-5 text-[10.5px] font-mono">
                  <div>
                    <span className="text-ink-3">train </span>
                    <span className="text-ink-0 tabular-nums">{overview?.row_counts?.train?.toLocaleString() ?? "—"}</span>
                  </div>
                  <div>
                    <span className="text-ink-3">test </span>
                    <span className="text-ink-0 tabular-nums">{overview?.row_counts?.test?.toLocaleString() ?? "—"}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── KPI tiles ── */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <KpiCard
            label="Total flow records"
            value={totalFlows > 0 ? totalFlows.toLocaleString() : "—"}
            color="#38BDF8"
            spark={SPARK_RECORDS}
            sub="CICIDS2017 + CSE-CIC-IDS2018 dataset"
          />
          <KpiCard
            label="Best F1 score"
            value={bestF1 > 0 ? bestF1.toFixed(4) : "—"}
            color={bColor}
            spark={SPARK_F1}
            sub={bestModel ? modelLabel(bestModel) : undefined}
          />
          <KpiCard
            label="Attack classes"
            value={overview?.labels?.length ?? "—"}
            color="#F43F5E"
            spark={SPARK_CLASSES}
            sub={`${models.length} / 6 models trained`}
          />
          <KpiCard
            label="Features"
            value={overview?.n_features ?? "—"}
            color="#10B981"
            spark={SPARK_FEATURES}
            sub="Network flow features"
          />
        </div>

        {/* ── Main panels ── */}
        <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">

          {/* Class distribution */}
          <Panel className="lg:col-span-5 flex flex-col">
            <PanelHeader
              eyebrow="Dataset"
              title="Label distribution"
              sub={`${distEntries.length} classes · ${totalFlows.toLocaleString()} total records`}
            />
            <div className="px-5 pt-4 pb-5 flex-1 overflow-y-auto">
              {donutData.length > 0 && (
                <div className="relative aspect-square w-full max-w-[200px] mx-auto mb-4">
                  <Donut data={donutData} size={200} thickness={18} />
                  <div className="absolute inset-0 grid place-items-center pointer-events-none">
                    <div className="text-center">
                      <div className="text-[10px] uppercase tracking-[.16em] text-ink-2">Total</div>
                      <div className="tabular-nums text-[22px] font-semibold leading-none mt-0.5 text-ink-0">
                        {totalFlows > 0 ? (totalFlows / 1000).toFixed(0) + "K" : "—"}
                      </div>
                      <div className="text-[10.5px] text-ink-2 mt-0.5">flows</div>
                    </div>
                  </div>
                </div>
              )}

              <div className="space-y-2.5">
                {distEntries.map(([cls, count]) => {
                  const pct = totalFlows > 0 ? ((count as number) / totalFlows) * 100 : 0;
                  const color = classColor(cls);
                  return (
                    <div key={cls} className="space-y-1">
                      <div className="flex items-center justify-between text-[11.5px]">
                        <span className="flex items-center gap-1.5">
                          <span className="h-2 w-2 rounded-sm flex-shrink-0" style={{ background: color }} />
                          <span className="text-ink-0">{cls}</span>
                        </span>
                        <span className="tabular-nums text-ink-1">
                          {(count as number).toLocaleString()}{" "}
                          <span className="text-ink-3">({pct.toFixed(1)}%)</span>
                        </span>
                      </div>
                      {/* bg-surface-elevated is theme-aware; replaces dark-only bg-white/[.04] */}
                      <div className="h-1 rounded-full bg-surface-elevated overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{ width: `${pct}%`, background: color }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </Panel>

          {/* Model leaderboard */}
          <Panel className="lg:col-span-7 flex flex-col">
            <PanelHeader
              eyebrow="Models"
              title="Leaderboard"
              sub="Sorted by F1 weighted (descending)"
              right={
                ranked.length > 0 ? (
                  <Pill tone="brand" size="sm">{ranked.length} models</Pill>
                ) : undefined
              }
            />

            {ranked.length === 0 ? (
              <div className="flex-1 px-5 py-8 text-center text-[12px] text-ink-2">
                Run{" "}
                <code className="font-mono text-[10.5px] py-0.5 px-[5px] rounded bg-surface-elevated ring-1 ring-line-base text-ink-1 inline-flex items-center">
                  --stage evaluate
                </code>{" "}
                to generate comparison data.
              </div>
            ) : (
              <div className="flex flex-col flex-1 min-h-0">
                {/* Header */}
                <div className="grid grid-cols-[40px_1fr_80px_80px_80px_80px] gap-3 px-4 h-8 items-center text-[10px] uppercase tracking-[.14em] text-ink-2 border-b border-line-subtle flex-shrink-0">
                  <span>Rank</span>
                  <span>Model</span>
                  <span className="text-right">F1 (W)</span>
                  <span className="text-right">Accuracy</span>
                  <span className="text-right">Precision</span>
                  <span className="text-right">Recall</span>
                </div>

                <div className="flex-1 overflow-y-auto divide-y divide-line-subtle">
                  {ranked.map(([name, m], i) => {
                    const color = modelColor(name);
                    const isChamp = i === 0;
                    return (
                      <div
                        key={name}
                        className={`grid grid-cols-[40px_1fr_80px_80px_80px_80px] gap-3 px-4 py-3 items-center transition hover:bg-surface-hover ${isChamp ? "bg-brand-blue/[.04]" : ""}`}
                      >
                        <span
                          className="text-[10.5px] font-semibold tabular-nums px-1.5 py-0.5 rounded font-mono text-center"
                          style={{ background: `${color}22`, color }}
                        >
                          {MEDAL[i] ?? `${i + 1}th`}
                        </span>

                        <div className="flex items-center gap-2 min-w-0">
                          <div
                            className="h-7 w-7 rounded flex-shrink-0 grid place-items-center font-mono text-[10.5px] font-semibold"
                            style={{ background: `${color}1F`, color }}
                          >
                            {modelShort(name)}
                          </div>
                          <div className="min-w-0">
                            <div className="text-[12.5px] font-medium text-ink-0 truncate">
                              {modelLabel(name)}
                            </div>
                            {isChamp && <Pill tone="brand" size="sm">★ best</Pill>}
                          </div>
                        </div>

                        <div className="tabular-nums text-[12.5px] font-semibold font-mono text-ink-0 text-right">
                          {(m.f1_weighted ?? 0).toFixed(4)}
                        </div>
                        <div className="tabular-nums text-[12px] text-ink-1 text-right">
                          {(m.accuracy ?? 0).toFixed(4)}
                        </div>
                        <div className="tabular-nums text-[12px] text-ink-1 text-right">
                          {(m.precision_weighted ?? 0).toFixed(4)}
                        </div>
                        <div className="tabular-nums text-[12px] text-ink-1 text-right">
                          {(m.recall_weighted ?? 0).toFixed(4)}
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Data quality footer */}
                <div className="border-t border-line-subtle px-4 py-3 flex-shrink-0">
                  <div className="flex items-center gap-4 text-[11px] text-ink-2">
                    <span className="flex items-center gap-1.5">
                      <span className="h-1.5 w-1.5 rounded-sm bg-ok" />
                      {overview?.missing_total ?? 0} missing
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span className="h-1.5 w-1.5 rounded-sm bg-ok" />
                      {overview?.infinite_total ?? 0} ±Inf
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span className="h-1.5 w-1.5 rounded-sm bg-ok" />
                      {overview?.duplicate_row_count?.toLocaleString() ?? 0} dupes
                    </span>
                    <span className="ml-auto tabular-nums text-ink-3">
                      {overview?.n_features ?? "—"} features · holdout test set
                    </span>
                  </div>
                </div>
              </div>
            )}
          </Panel>
        </div>
      </div>
    </AppShell>
  );
}
