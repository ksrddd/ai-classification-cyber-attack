import { AppShell } from "@/components/shell/AppShell";
import { Panel, PanelHeader } from "@/components/ui/Panel";
import { KpiCard } from "@/components/ui/KpiCard";
import { Pill } from "@/components/ui/Pill";
import { getCompare } from "@/lib/api";
import { modelColor, modelLabel } from "@/lib/colors";

export const dynamic = "force-dynamic";

const METRIC_COLS = [
  { k: "accuracy",            l: "Accuracy"   },
  { k: "f1_weighted",         l: "F1 (W)"     },
  { k: "f1_macro",            l: "F1 (M)"     },
  { k: "precision_weighted",  l: "Precision"  },
  { k: "recall_weighted",     l: "Recall"     },
  { k: "roc_auc",             l: "ROC-AUC"    },
  { k: "matthews_corrcoef",   l: "MCC"        },
];

const BAR_METRICS = [
  { k: "f1_weighted",   l: "F1 weighted",  color: "#22D3EE" },
  { k: "f1_macro",      l: "F1 macro",     color: "#6366F1" },
  { k: "accuracy",      l: "Accuracy",     color: "#3B82F6" },
  { k: "roc_auc",       l: "ROC-AUC",      color: "#10B981" },
];

function modelShort(name: string): string {
  const map: Record<string, string> = {
    random_forest: "RF", xgboost: "XGB", lightgbm: "LGB",
    catboost: "CB", mlp: "MLP", logistic_regression: "LR",
  };
  return map[name] ?? name.slice(0, 3).toUpperCase();
}

export default async function ComparePage() {
  const compare = await getCompare().catch(() => null);

  if (!compare) {
    return (
      <AppShell title="Model Comparison">
        <div className="py-16 text-center text-ink-2 text-[13px]">
          No comparison data. Run{" "}
          <code className="font-mono text-[10.5px] py-0.5 px-[5px] rounded bg-white/[.06] ring-1 ring-white/10 text-ink-1 inline-flex items-center">
            python main.py --stage evaluate
          </code>
        </div>
      </AppShell>
    );
  }

  const ranked = Object.entries(compare).sort(
    ([, a], [, b]) =>
      ((b as Record<string, number>).f1_weighted ?? 0) -
      ((a as Record<string, number>).f1_weighted ?? 0),
  );

  const [bestName, bestMetrics] = ranked[0];
  const bestColor = modelColor(bestName);
  const medals = ["1st", "2nd", "3rd", "4th", "5th", "6th"];

  return (
    <AppShell title="Model Comparison">
      <div className="space-y-5 animate-fadeIn">
        <div>
          <h1 className="text-[26px] font-semibold tracking-tight text-ink-0">Model Comparison</h1>
          <p className="text-[12px] text-ink-2 mt-1">
            Cross-model leaderboard · holdout test set · sorted by F1 weighted
          </p>
        </div>

        {/* KPI row */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <KpiCard label="Best model"      value={modelLabel(bestName)}  color={bestColor} />
          <KpiCard label="Best F1 (W)"     value={(bestMetrics as Record<string,number>).f1_weighted?.toFixed(4) ?? "—"} color={bestColor} />
          <KpiCard label="Models compared" value={ranked.length}         color="#6366F1" />
        </div>

        {/* Hero best-model strip */}
        <div
          className="relative overflow-hidden rounded-xl p-5 ring-1"
          style={{ background: `${bestColor}08`, borderColor: `${bestColor}30` }}
        >
          <div
            className="absolute -right-8 -top-8 h-32 w-32 rounded-full blur-2xl opacity-20 pointer-events-none"
            style={{ background: bestColor }}
          />
          <div className="relative flex items-center gap-4 flex-wrap">
            <div
              className="h-12 w-12 rounded-xl grid place-items-center font-semibold text-[14px] flex-shrink-0"
              style={{ background: `${bestColor}1F`, color: bestColor, boxShadow: `inset 0 0 0 1px ${bestColor}55` }}
            >
              {modelShort(bestName)}
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-[18px] font-semibold text-ink-0">{modelLabel(bestName)}</span>
                <Pill tone="brand" size="sm">★ champion</Pill>
              </div>
              <div className="mt-2 flex items-center gap-4 flex-wrap text-[11px] tabular-nums">
                {["f1_weighted", "accuracy", "precision_weighted", "recall_weighted", "roc_auc"].map((k) => (
                  <span key={k} className="inline-flex items-center gap-1.5 text-ink-1">
                    <span className="h-1 w-1 rounded-full" style={{ background: bestColor }} />
                    <span className="text-ink-3">{k.replace(/_/g, " ")}</span>
                    <span className="text-ink-0 font-medium">
                      {((bestMetrics as Record<string, number>)[k] ?? 0).toFixed(4)}
                    </span>
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Leaderboard table */}
        <Panel>
          <PanelHeader eyebrow="Leaderboard" title="All models ranked" right={<Pill tone="muted" size="sm">holdout test set</Pill>} />
          <div className="overflow-x-auto">
            <table className="w-full text-[12px]">
              <thead>
                <tr className="text-left border-b border-line-subtle">
                  <th className="px-4 py-3 text-[10px] uppercase tracking-[.14em] text-ink-2 font-medium w-16">Rank</th>
                  <th className="px-4 py-3 text-[10px] uppercase tracking-[.14em] text-ink-2 font-medium">Model</th>
                  {METRIC_COLS.map((c) => (
                    <th key={c.k} className="px-4 py-3 text-[10px] uppercase tracking-[.14em] text-ink-2 font-medium text-right">
                      {c.l}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {ranked.map(([name, m], i) => {
                  const metrics = m as Record<string, number>;
                  const color = modelColor(name);
                  const isFirst = i === 0;
                  return (
                    <tr
                      key={name}
                      className={`border-b border-line-subtle last:border-0 hover:bg-white/[.02] transition ${isFirst ? "bg-brand-blue/[.03]" : ""}`}
                    >
                      <td className="px-4 py-3">
                        <span
                          className="text-[10.5px] font-semibold px-2 py-0.5 rounded font-mono"
                          style={{ background: `${color}22`, color }}
                        >
                          {medals[i] ?? `${i + 1}th`}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <span className="flex items-center gap-2">
                          <span
                            className="h-7 w-7 rounded grid place-items-center font-mono text-[10.5px] font-semibold flex-shrink-0"
                            style={{ background: `${color}1F`, color }}
                          >
                            {modelShort(name)}
                          </span>
                          <span className="font-medium text-ink-0">{modelLabel(name)}</span>
                          {isFirst && <Pill tone="brand" size="sm">★ best</Pill>}
                        </span>
                      </td>
                      {METRIC_COLS.map((col) => (
                        <td key={col.k} className="px-4 py-3 text-right tabular-nums"
                            style={{ color: isFirst ? "#10B981" : "#A8AFC0" }}>
                          {(metrics[col.k] ?? 0).toFixed(4)}
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>

        {/* Visual comparison */}
        <Panel>
          <PanelHeader eyebrow="Visual" title="Side-by-side metric comparison" />
          <div className="px-5 py-4 space-y-6">
            {BAR_METRICS.map(({ k, l, color: barColor }) => (
              <div key={k}>
                <div className="text-[10px] uppercase tracking-[.14em] text-ink-2 mb-2 flex items-center gap-1.5">
                  <span className="h-1 w-1 rounded-full" style={{ background: barColor }} />
                  {l}
                </div>
                <div className="space-y-2">
                  {ranked.map(([name, m]) => {
                    const val = (m as Record<string, number>)[k] ?? 0;
                    const color = modelColor(name);
                    return (
                      <div key={name} className="grid grid-cols-[140px_1fr_64px] gap-3 items-center">
                        <span className="text-[11.5px] text-ink-1 truncate">{modelLabel(name)}</span>
                        <div className="h-1.5 rounded-full bg-white/[.04] overflow-hidden">
                          <div
                            className="h-full rounded-full"
                            style={{ width: `${val * 100}%`, background: color }}
                          />
                        </div>
                        <span className="tabular-nums text-[11.5px] text-right" style={{ color }}>
                          {val.toFixed(4)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </AppShell>
  );
}
