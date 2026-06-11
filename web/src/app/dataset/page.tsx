import { AppShell } from "@/components/shell/AppShell";
import { Panel, PanelHeader } from "@/components/ui/Panel";
import { KpiCard } from "@/components/ui/KpiCard";
import { Pill } from "@/components/ui/Pill";
import { Donut } from "@/components/ui/Donut";
import { getOverview } from "@/lib/api";
import { classColor } from "@/lib/colors";

export const dynamic = "force-dynamic";

export default async function DatasetPage() {
  const overview = await getOverview().catch(() => null);

  const dist = overview?.label_distribution ?? {};
  const total = Object.values(dist).reduce((a: number, b: number) => a + b, 0);
  const entries = Object.entries(dist).sort(([, a], [, b]) => (b as number) - (a as number));

  const donutData = entries.map(([cls, count]) => ({
    name: cls,
    pct: total > 0 ? ((count as number) / total) * 100 : 0,
    color: classColor(cls),
  }));

  return (
    <AppShell title="Dataset Overview">
      <div className="space-y-5 animate-fadeIn">
        <div>
          <h1 className="text-[26px] font-semibold tracking-tight text-ink-0">Dataset Overview</h1>
          <p className="text-[12px] text-ink-2 mt-1">
            CICIDS2017 · Canadian Institute for Cybersecurity · 8 CSVs · ~2.8M flow records · 78 features
          </p>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <KpiCard label="Mode"           value={overview?.mode?.toUpperCase() ?? "—"} color="#22D3EE" />
          <KpiCard label="Target classes" value={overview?.labels?.length ?? "—"}      color="#3B82F6" />
          <KpiCard label="Train rows"     value={overview?.row_counts?.train?.toLocaleString() ?? "—"} color="#10B981" />
          <KpiCard label="Test rows"      value={overview?.row_counts?.test?.toLocaleString()  ?? "—"} color="#F59E0B" />
        </div>

        {/* Status pills */}
        <div className="flex flex-wrap gap-2">
          <Pill tone="ok">CICIDS2017 ready</Pill>
          <Pill tone="ok">{overview?.n_features ?? 78} features</Pill>
          <Pill tone={overview?.missing_total ? "warn" : "ok"}>
            {overview?.missing_total ?? 0} missing cells
          </Pill>
          <Pill tone={overview?.infinite_total ? "warn" : "ok"}>
            {overview?.infinite_total ?? 0} ±Inf values
          </Pill>
          <Pill tone="brand">
            mode: {overview?.classification_mode ?? overview?.mode ?? "—"}
          </Pill>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Label distribution with donut */}
          <Panel>
            <PanelHeader
              eyebrow="Post-mapping"
              title="Label distribution"
              sub={`${total.toLocaleString()} total records`}
            />
            <div className="px-5 py-4">
              {donutData.length > 0 && (
                <div className="relative aspect-square w-full max-w-[180px] mx-auto mb-5">
                  <Donut data={donutData} size={180} thickness={16} />
                  <div className="absolute inset-0 grid place-items-center pointer-events-none">
                    <div className="text-center">
                      <div className="text-[10px] uppercase tracking-[.16em] text-ink-2">Total</div>
                      <div className="tabular-nums text-[20px] font-semibold leading-none mt-0.5 text-ink-0">
                        {total > 0 ? (total / 1000).toFixed(0) + "K" : "—"}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div className="space-y-2.5">
                {entries.map(([cls, count]) => {
                  const pct = total > 0 ? ((count as number) / total) * 100 : 0;
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
                      <div className="h-1 rounded-full bg-white/[.04] overflow-hidden">
                        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </Panel>

          {/* Dataset stats panel */}
          <Panel>
            <PanelHeader
              eyebrow="Label scheme"
              title="Active classes"
              sub={`${overview?.mode ?? "—"} classification`}
            />
            <div className="px-5 py-4">
              <div className="flex flex-wrap gap-2 mb-6">
                {(overview?.labels ?? []).map((lb) => {
                  const color = classColor(lb);
                  return (
                    <span
                      key={lb}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[12px] font-semibold"
                      style={{
                        background: `${color}18`,
                        color,
                        boxShadow: `inset 0 0 0 1px ${color}35`,
                      }}
                    >
                      <span className="h-1.5 w-1.5 rounded-full" style={{ background: color }} />
                      {lb}
                    </span>
                  );
                })}
              </div>

              <div className="space-y-0 divide-y divide-line-subtle text-[12px]">
                {[
                  ["Total records", total.toLocaleString()],
                  ["Features", (overview?.n_features ?? 78).toString()],
                  ["Train rows", overview?.row_counts?.train?.toLocaleString() ?? "—"],
                  ["Test rows", overview?.row_counts?.test?.toLocaleString() ?? "—"],
                  ["Missing cells", (overview?.missing_total ?? 0).toString()],
                  ["±Inf values", (overview?.infinite_total ?? 0).toString()],
                  ["Duplicate rows", (overview?.duplicate_row_count ?? 0).toLocaleString()],
                ].map(([label, value]) => (
                  <div key={label} className="flex items-center justify-between py-2.5">
                    <span className="text-ink-2">{label}</span>
                    <span className="tabular-nums font-mono text-ink-0">{value}</span>
                  </div>
                ))}
              </div>
            </div>
          </Panel>
        </div>
      </div>
    </AppShell>
  );
}
