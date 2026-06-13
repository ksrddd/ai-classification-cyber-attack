"use client";

import { useEffect, useRef, useState } from "react";
import { AppShell } from "@/components/shell/AppShell";
import { getModels, predictCsv } from "@/lib/api";
import { classColor, modelColor, modelLabel, modelShort } from "@/lib/colors";
import { Upload, FileText, AlertTriangle, ChevronRight } from "lucide-react";

type PredictResult = {
  n_rows: number;
  class_counts: Record<string, number>;
  preview: Record<string, unknown>[];
  columns: string[];
  usedModel: string;
};

const MAX_FILE_BYTES = 50 * 1024 * 1024; // 50 MB

/* ── tiny helpers ──────────────────────────────────────────────────────── */

function ConfidenceBadge({ value }: { value: number }) {
  const pct = value * 100;
  const color = pct >= 90 ? "#10B981" : pct >= 70 ? "#F59E0B" : "#EF4444";
  return (
    <span className="tabular-nums font-mono font-semibold text-[11px]" style={{ color }}>
      {pct.toFixed(1)}%
    </span>
  );
}

function MiniBar({ value, color, label }: { value: number; color: string; label: string }) {
  return (
    <div className="space-y-0.5">
      <div className="flex justify-between items-center">
        <span className="text-[9px] uppercase tracking-[.1em] text-ink-3 font-mono truncate pr-1">{label}</span>
        <span className="text-[9.5px] tabular-nums font-mono text-ink-2">{value.toFixed(3)}</span>
      </div>
      <div className="h-0.5 rounded-full bg-surface-elevated overflow-hidden">
        <div className="h-full rounded-full transition-all" style={{ width: `${value * 100}%`, background: color }} />
      </div>
    </div>
  );
}

/* ── main page ─────────────────────────────────────────────────────────── */

export default function PredictPage() {
  const [models, setModels] = useState<string[]>([]);
  const [selected, setSelected] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let cancelled = false;
    getModels().catch(() => ({ models: [] as string[] })).then((d) => {
      if (cancelled) return;
      setModels(d.models);
      if (d.models[0]) setSelected(d.models[0]);
    });
    return () => { cancelled = true; };
  }, []);

  const handleFile = (f: File) => {
    if (f.size > MAX_FILE_BYTES) { setError(`File too large (max 50 MB)`); return; }
    setFile(f); setResult(null); setError("");
  };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f?.name.endsWith(".csv")) handleFile(f);
  };
  const handlePredict = async () => {
    if (!file || !selected) return;
    setLoading(true); setError("");
    try {
      const res = await predictCsv(selected, file);
      setResult({ ...res, usedModel: selected });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Prediction failed");
    } finally { setLoading(false); }
  };

  const total = result ? Object.values(result.class_counts).reduce((a, b) => a + b, 0) : 0;
  const modelChanged = result && result.usedModel !== selected;
  const availableProbs = result ? result.columns.filter((c) => c.startsWith("proba_")) : [];
  const sortedClasses = result
    ? Object.entries(result.class_counts).sort(([, a], [, b]) => b - a)
    : [];
  const topClass = sortedClasses[0]?.[0] ?? "";
  const mc = modelColor(selected);

  return (
    <AppShell title="Predict CSV">
      <div className="animate-fadeIn space-y-6">

        {/* ── page title ── */}
        <div className="flex items-end justify-between">
          <div>
            <p className="text-[9.5px] uppercase tracking-[.2em] text-ink-3 font-semibold mb-1">Inference</p>
            <h1 className="text-[22px] font-semibold tracking-tight text-ink-0">Predict New Traffic</h1>
            <p className="text-[11px] text-ink-3 font-mono mt-1">
              CICIDS2017 · 77 features · 7 attack classes · per-row probabilities
            </p>
          </div>
        </div>

        {/* ── step 1+2 in one row ── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

          {/* Model selector */}
          <div className="border border-line-base rounded bg-surface-raised"
            style={{ borderTopColor: mc, borderTopWidth: "2px" }}>
            <div className="px-4 pt-3 pb-2 border-b border-line-subtle flex items-center justify-between">
              <div>
                <p className="text-[9px] uppercase tracking-[.2em] text-ink-3 font-semibold">Step 1</p>
                <p className="text-[13px] font-semibold text-ink-0 mt-0.5">Select model</p>
              </div>
              <span className="font-mono text-[10px] text-ink-3 bg-surface-elevated border border-line-base rounded-sm px-2 py-0.5">
                {modelShort(selected)}
              </span>
            </div>
            <div className="p-4 space-y-2">
              {models.map((name) => {
                const isActive = name === selected;
                const c = modelColor(name);
                return (
                  <button
                    key={name}
                    onClick={() => setSelected(name)}
                    className="w-full flex items-center gap-3 h-9 px-3 rounded-sm border text-left transition-colors"
                    style={{
                      borderColor: isActive ? `${c}50` : "var(--border-subtle)",
                      background: isActive ? `${c}0E` : "transparent",
                    }}
                  >
                    <span
                      className="h-5 w-5 rounded-sm grid place-items-center font-mono text-[9px] font-bold flex-shrink-0"
                      style={{ background: `${c}20`, color: c }}
                    >
                      {modelShort(name)}
                    </span>
                    <span className="flex-1 text-[12px] font-medium" style={{ color: isActive ? c : "var(--text-secondary)" }}>
                      {modelLabel(name)}
                    </span>
                    {isActive && <ChevronRight size={12} style={{ color: c }} />}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Upload */}
          <div className="border border-line-base rounded bg-surface-raised flex flex-col">
            <div className="px-4 pt-3 pb-2 border-b border-line-subtle">
              <p className="text-[9px] uppercase tracking-[.2em] text-ink-3 font-semibold">Step 2</p>
              <p className="text-[13px] font-semibold text-ink-0 mt-0.5">Upload CSV</p>
            </div>
            <div className="p-4 flex-1 flex flex-col gap-3">
              <div
                onDrop={handleDrop}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onClick={() => inputRef.current?.click()}
                className="flex-1 min-h-[140px] border border-dashed rounded-sm flex flex-col items-center justify-center cursor-pointer transition-colors"
                style={{
                  borderColor: dragOver ? "#22D3EE" : file ? "#10B981" : "var(--border-base)",
                  background: dragOver ? "rgba(34,211,238,0.04)" : file ? "rgba(16,185,129,0.04)" : "transparent",
                }}
              >
                <input ref={inputRef} type="file" accept=".csv" className="hidden"
                  onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} />
                {file ? (
                  <div className="text-center space-y-2">
                    <FileText size={28} className="mx-auto text-ok" />
                    <div>
                      <p className="text-[13px] font-semibold text-ink-0">{file.name}</p>
                      <p className="text-[10.5px] font-mono text-ink-3 mt-0.5">
                        {(file.size / 1024).toFixed(1)} KB · ready
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="text-center space-y-2">
                    <Upload size={28} className="mx-auto text-ink-3" />
                    <div>
                      <p className="text-[12.5px] text-ink-2 font-medium">Drop CSV here or click to browse</p>
                      <p className="text-[10.5px] font-mono text-ink-3 mt-0.5">CICIDS-format · .csv only</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Run button inside upload panel */}
              <button
                onClick={handlePredict}
                disabled={!file || !selected || loading}
                className="w-full h-10 rounded-sm font-semibold text-[12.5px] disabled:opacity-30 transition-colors flex items-center justify-center gap-2"
                style={{ background: mc, color: "#07090E" }}
              >
                {loading ? (
                  <><span className="h-1.5 w-1.5 rounded-sm bg-canvas/50 animate-pulseDot" />Classifying…</>
                ) : (
                  <><Upload size={14} />Run — {modelLabel(selected)}</>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Warning: model changed after run */}
        {modelChanged && (
          <div className="flex items-center gap-2 text-[11px] font-mono text-warn border border-warn/20 bg-warn/5 rounded-sm px-3 py-2">
            <AlertTriangle size={13} className="flex-shrink-0" />
            Showing results from <strong>{modelLabel(result!.usedModel)}</strong>.
            Click Run to re-classify with {modelLabel(selected)}.
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="border border-critical/25 bg-critical/5 rounded-sm px-4 py-3 text-[11.5px] text-critical font-mono">
            {error}
          </div>
        )}

        {/* ── Results ── */}
        {result && (
          <div className="space-y-4 animate-fadeIn">

            {/* Result header */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                { label: "Model", value: modelShort(result.usedModel), sub: modelLabel(result.usedModel), color: modelColor(result.usedModel) },
                { label: "Rows classified", value: result.n_rows.toLocaleString(), sub: file?.name ?? "", color: "#38BDF8" },
                { label: "Classes detected", value: Object.keys(result.class_counts).length, sub: `of 7 total`, color: "#6366F1" },
                { label: "Top class", value: topClass, sub: `${((result.class_counts[topClass] ?? 0) / total * 100).toFixed(1)}% of rows`, color: classColor(topClass) },
              ].map(({ label, value, sub, color }) => (
                <div key={label}
                  className="border border-line-base rounded bg-surface-raised p-4"
                  style={{ borderLeftColor: color, borderLeftWidth: "2px" }}>
                  <p className="text-[9.5px] uppercase tracking-[.16em] text-ink-3 font-semibold">{label}</p>
                  <p className="text-[20px] font-bold font-mono tabular-nums text-ink-0 mt-1 leading-none">{value}</p>
                  <p className="text-[10px] font-mono text-ink-3 mt-1.5 truncate">{sub}</p>
                </div>
              ))}
            </div>

            {/* Main result panels */}
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">

              {/* Class breakdown */}
              <div className="lg:col-span-2 border border-line-base rounded bg-surface-raised">
                <div className="px-4 pt-3 pb-2 border-b border-line-subtle">
                  <p className="text-[9px] uppercase tracking-[.2em] text-ink-3 font-semibold">Distribution</p>
                  <p className="text-[13px] font-semibold text-ink-0 mt-0.5">Predicted classes</p>
                  <p className="text-[10.5px] font-mono text-ink-3 mt-0.5">{total.toLocaleString()} total rows</p>
                </div>
                <div className="p-4 space-y-3">
                  {sortedClasses.map(([cls, count]) => {
                    const pct = total > 0 ? (count / total) * 100 : 0;
                    const color = classColor(cls);
                    const isThreat = cls !== "BENIGN";
                    return (
                      <div key={cls} className="space-y-1.5">
                        <div className="flex items-center justify-between">
                          <span className="flex items-center gap-2 text-[11.5px]">
                            <span className="h-2 w-2 rounded-sm flex-shrink-0" style={{ background: color }} />
                            <span className={isThreat ? "font-semibold text-ink-0" : "text-ink-1"}>{cls}</span>
                          </span>
                          <span className="flex items-center gap-2 text-[11px] font-mono">
                            <span className="tabular-nums text-ink-2">{count.toLocaleString()}</span>
                            <span className="tabular-nums text-ink-3 w-10 text-right">{pct.toFixed(1)}%</span>
                          </span>
                        </div>
                        <div className="h-1 rounded-full bg-surface-elevated overflow-hidden">
                          <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Per-row probability table */}
              <div className="lg:col-span-3 border border-line-base rounded bg-surface-raised overflow-hidden">
                <div className="px-4 pt-3 pb-2 border-b border-line-subtle flex items-start justify-between">
                  <div>
                    <p className="text-[9px] uppercase tracking-[.2em] text-ink-3 font-semibold">Detail</p>
                    <p className="text-[13px] font-semibold text-ink-0 mt-0.5">Per-row probabilities</p>
                    <p className="text-[10.5px] font-mono text-ink-3 mt-0.5">
                      first 10 rows · {modelLabel(result.usedModel)}
                    </p>
                  </div>
                  <span className="text-[9.5px] font-mono text-ink-3 border border-line-base rounded-sm px-2 py-0.5 mt-0.5">
                    {modelShort(result.usedModel)}
                  </span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-line-subtle">
                        <th className="px-3 py-2 text-left text-[9px] uppercase tracking-[.14em] text-ink-3 font-semibold w-8">#</th>
                        <th className="px-3 py-2 text-left text-[9px] uppercase tracking-[.14em] text-ink-3 font-semibold">Prediction</th>
                        <th className="px-3 py-2 text-right text-[9px] uppercase tracking-[.14em] text-ink-3 font-semibold">Conf.</th>
                        <th className="px-3 py-2 text-left text-[9px] uppercase tracking-[.14em] text-ink-3 font-semibold min-w-[160px]">
                          Class probabilities
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.preview.slice(0, 10).map((row, i) => {
                        const label = String(row["predicted_label"] ?? "—");
                        const maxP = availableProbs.length > 0
                          ? Math.max(...availableProbs.map((c) => Number(row[c] ?? 0)))
                          : (typeof row["max_proba"] === "number" ? row["max_proba"] : 0);
                        const isThreat = label !== "BENIGN";
                        const dominantProb = availableProbs.find((c) => c === `proba_${label}`);
                        const topProbs = availableProbs
                          .map((c) => ({ cls: c.replace("proba_", ""), val: Number(row[c] ?? 0) }))
                          .filter((x) => x.val > 0.01)
                          .sort((a, b) => b.val - a.val)
                          .slice(0, 3);

                        return (
                          <tr key={i}
                            className="border-b border-line-subtle last:border-0 hover:bg-surface-elevated transition-colors"
                          >
                            <td className="px-3 py-2.5 tabular-nums font-mono text-[10px] text-ink-3">{i + 1}</td>
                            <td className="px-3 py-2.5">
                              <span className="flex items-center gap-1.5">
                                <span className="h-1.5 w-1.5 rounded-sm flex-shrink-0"
                                  style={{ background: classColor(label) }} />
                                <span
                                  className="text-[11.5px] font-semibold"
                                  style={{ color: isThreat ? classColor(label) : "var(--text-primary)" }}
                                >
                                  {label}
                                </span>
                              </span>
                            </td>
                            <td className="px-3 py-2.5 text-right">
                              <ConfidenceBadge value={maxP} />
                            </td>
                            <td className="px-3 py-2.5">
                              <div className="space-y-1 min-w-[150px]">
                                {topProbs.map(({ cls, val }) => (
                                  <MiniBar key={cls} value={val} color={classColor(cls)} label={cls} />
                                ))}
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Footer note */}
            <p className="text-[10px] font-mono text-ink-3">
              probabilities are calibrated outputs from{" "}
              <span className="text-ink-1">{modelLabel(result.usedModel)}</span> ·
              switch model and re-run to compare · results shown for first 10 rows only
            </p>
          </div>
        )}
      </div>
    </AppShell>
  );
}
