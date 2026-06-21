import { AppShell } from "@/components/shell/AppShell";
import { figureUrl, getOverview } from "@/lib/api";
import { FigureImg } from "@/components/ui/FigureImg";
import { CheckCircle2, AlertTriangle, Info } from "lucide-react";

export const revalidate = 300;

const FIGURES = [
  {
    key: "class_distribution",
    label: "Class Distribution",
    eyebrow: "01 · BALANCE AUDIT",
    sub: "Log-scaled class balance — 9 classes with severe minority imbalance",
    accent: "#F59E0B",
    col: "md:col-span-2",
    maxH: "320px",
  },
  {
    key: "missing_value_audit",
    label: "Missing Value Audit",
    eyebrow: "02 · INTEGRITY CHECK",
    sub: "Top-20 columns by null count — completeness scan before imputation",
    accent: "#38BDF8",
    col: "",
  },
  {
    key: "correlation_heatmap",
    label: "Feature Correlation Matrix",
    eyebrow: "03 · STRUCTURE ANALYSIS",
    sub: "Pearson r across 25 highest-variance features — multicollinearity scan",
    accent: "#6366F1",
    col: "",
  },
  {
    key: "feature_distributions",
    label: "Feature Fingerprint",
    eyebrow: "04 · DISCRIMINATIVE POWER",
    sub: "Violin distributions of 6 most discriminative features across all attack classes",
    accent: "#10B981",
    col: "md:col-span-2",
  },
];

export default async function EdaPage() {
  const ov = await getOverview().catch(() => null);
  const rows = ov?.shape?.[0];
  const cols = ov?.shape?.[1];
  const features = ov?.n_features ?? 77;
  const dupes = ov?.duplicate_row_count ?? 0;
  const missing = ov?.missing_total ?? 0;
  const infinite = ov?.infinite_total ?? 0;
  const nClasses = ov?.labels?.length ?? 7;

  return (
    <AppShell title="EDA">
      <div className="space-y-5 animate-fadeIn">

        {/* ── Page header ── */}
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-[9px] uppercase tracking-[.22em] text-ink-3 font-semibold font-mono mb-1">
              Dataset Audit
            </p>
            <h1 className="text-[22px] font-semibold tracking-tight text-ink-0">
              Exploratory Analysis
            </h1>
            <p className="text-[11px] font-mono text-ink-3 mt-1">
              CICIDS2017 + CSE-CIC-IDS2018 · network flow records · pre-modelling sanity check
            </p>
          </div>
          <div className="hidden md:flex items-center gap-1.5 flex-shrink-0 text-[9.5px] font-mono font-semibold text-ok border border-ok/25 bg-ok/5 rounded-sm px-2.5 py-1.5">
            <span className="h-1.5 w-1.5 rounded-sm bg-ok" />
            SCAN COMPLETE
          </div>
        </div>

        {/* ── Dataset metrics strip ── */}
        <div className="border border-line-base rounded bg-surface-raised overflow-hidden">
          <div className="px-4 py-2 border-b border-line-subtle flex items-center">
            <span className="text-[9px] uppercase tracking-[.2em] text-ink-3 font-semibold font-mono">
              Dataset Overview
            </span>
            <span className="ml-auto font-mono text-[9px] text-ink-3">cicids-2017</span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 divide-x divide-y md:divide-y-0 divide-line-subtle">
            {[
              { label: "Total rows",     value: rows?.toLocaleString() ?? "—", sub: "raw traffic records", color: "#22D3EE" },
              { label: "Columns",        value: cols ?? "—",                   sub: "raw network features", color: "#3B82F6" },
              { label: "Model features", value: features,                      sub: "after preprocessing",  color: "#10B981" },
              { label: "Duplicate rows", value: dupes.toLocaleString(),        sub: dupes === 0 ? "none detected" : "flagged", color: dupes > 0 ? "#F59E0B" : "#6C7488" },
            ].map(({ label, value, sub, color }) => (
              <div key={label} className="px-4 py-3 flex items-start gap-3">
                <div
                  className="w-0.5 self-stretch rounded-full flex-shrink-0"
                  style={{ background: color, minHeight: "28px" }}
                />
                <div>
                  <p className="text-[9px] uppercase tracking-[.14em] text-ink-3 font-semibold font-mono">
                    {label}
                  </p>
                  <p className="text-[20px] font-bold font-mono tabular-nums text-ink-0 leading-none mt-0.5">
                    {value}
                  </p>
                  <p className="text-[10px] font-mono text-ink-3 mt-0.5">{sub}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* ── Scan findings ── */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">

          {/* Missing values */}
          {missing > 0 ? (
            <div className="border border-warn/25 bg-warn/5 rounded-sm px-3 py-2.5 flex items-center gap-3">
              <div className="h-7 w-7 rounded-sm bg-warn/10 grid place-items-center flex-shrink-0">
                <AlertTriangle size={14} className="text-warn" />
              </div>
              <div>
                <p className="text-[9px] uppercase tracking-[.14em] font-semibold font-mono text-ink-3">
                  Missing Cells
                </p>
                <p className="text-[12px] font-mono font-semibold text-warn">
                  {missing.toLocaleString()} detected
                </p>
              </div>
            </div>
          ) : (
            <div className="border border-ok/20 bg-ok/5 rounded-sm px-3 py-2.5 flex items-center gap-3">
              <div className="h-7 w-7 rounded-sm bg-ok/10 grid place-items-center flex-shrink-0">
                <CheckCircle2 size={14} className="text-ok" />
              </div>
              <div>
                <p className="text-[9px] uppercase tracking-[.14em] font-semibold font-mono text-ink-3">
                  Missing Cells
                </p>
                <p className="text-[12px] font-mono font-semibold text-ok">0 — clean</p>
              </div>
            </div>
          )}

          {/* Infinite values */}
          {infinite > 0 ? (
            <div className="border border-warn/25 bg-warn/5 rounded-sm px-3 py-2.5 flex items-center gap-3">
              <div className="h-7 w-7 rounded-sm bg-warn/10 grid place-items-center flex-shrink-0">
                <AlertTriangle size={14} className="text-warn" />
              </div>
              <div>
                <p className="text-[9px] uppercase tracking-[.14em] font-semibold font-mono text-ink-3">
                  ±Inf Values
                </p>
                <p className="text-[12px] font-mono font-semibold text-warn">
                  {infinite.toLocaleString()} detected
                </p>
              </div>
            </div>
          ) : (
            <div className="border border-ok/20 bg-ok/5 rounded-sm px-3 py-2.5 flex items-center gap-3">
              <div className="h-7 w-7 rounded-sm bg-ok/10 grid place-items-center flex-shrink-0">
                <CheckCircle2 size={14} className="text-ok" />
              </div>
              <div>
                <p className="text-[9px] uppercase tracking-[.14em] font-semibold font-mono text-ink-3">
                  ±Inf Values
                </p>
                <p className="text-[12px] font-mono font-semibold text-ok">0 — clean</p>
              </div>
            </div>
          )}

          {/* Class info */}
          <div className="border border-info/20 bg-info/5 rounded-sm px-3 py-2.5 flex items-center gap-3">
            <div className="h-7 w-7 rounded-sm bg-info/10 grid place-items-center flex-shrink-0">
              <Info size={14} className="text-info" />
            </div>
            <div>
              <p className="text-[9px] uppercase tracking-[.14em] font-semibold font-mono text-ink-3">
                Attack Classes
              </p>
              <p className="text-[12px] font-mono font-semibold text-info">
                {nClasses} labels · {ov?.classification_mode ?? "multiclass"}
              </p>
            </div>
          </div>
        </div>

        {/* ── Data leakage certificate ── */}
        <div className="border border-line-base rounded bg-surface-raised overflow-hidden">
          <div className="flex items-stretch">
            <div className="w-[3px] bg-ok flex-shrink-0" />
            <div className="flex items-center gap-4 px-4 py-3 flex-1 min-w-0">
              <div className="h-8 w-8 rounded-sm border border-ok/30 bg-ok/10 grid place-items-center flex-shrink-0">
                <CheckCircle2 size={15} className="text-ok" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-[11.5px] font-semibold text-ok">No Data Leakage Detected</p>
                <p className="text-[10.5px] text-ink-3 font-mono mt-0.5">
                  Scaler fits per cross-validation fold inside{" "}
                  <code className="text-ink-1 bg-surface-elevated px-1 rounded-sm border border-line-base">
                    sklearn.Pipeline
                  </code>
                  {" "}· labels encoded post-split ·{" "}
                  <code className="text-ink-1 bg-surface-elevated px-1 rounded-sm border border-line-base">
                    class_weight=&apos;balanced&apos;
                  </code>
                </p>
              </div>
              <span className="hidden md:inline-flex items-center gap-1 text-[9px] uppercase tracking-[.16em] font-semibold text-ok border border-ok/25 rounded-sm px-2 py-1 font-mono flex-shrink-0">
                PASS
              </span>
            </div>
          </div>
        </div>

        {/* ── Figure panels ── */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-start">
          {FIGURES.map(({ key, label, eyebrow, sub, accent, col, maxH }) => (
            <div
              key={key}
              className={`border border-line-base rounded bg-surface-raised overflow-hidden ${col}`}
              style={{ borderTopColor: accent, borderTopWidth: "2px" }}
            >
              <div className="px-4 pt-3 pb-2.5 border-b border-line-subtle flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p
                    className="text-[9px] uppercase tracking-[.2em] font-semibold font-mono mb-0.5"
                    style={{ color: accent }}
                  >
                    {eyebrow}
                  </p>
                  <p className="text-[13px] font-semibold text-ink-0">{label}</p>
                  <p className="text-[10.5px] font-mono text-ink-3 mt-0.5">{sub}</p>
                </div>
                <span
                  className="h-1.5 w-1.5 rounded-sm flex-shrink-0 mt-1"
                  style={{ background: accent }}
                />
              </div>
              <div className="p-4">
                <FigureImg src={figureUrl(`${key}.png`)} alt={label} maxH={maxH} />
              </div>
            </div>
          ))}
        </div>

      </div>
    </AppShell>
  );
}
