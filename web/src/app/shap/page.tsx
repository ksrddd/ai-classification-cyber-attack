"use client";

import { useState, useEffect } from "react";
import { AppShell } from "@/components/shell/AppShell";
import { Panel, PanelHeader } from "@/components/ui/Panel";
import { KpiCard } from "@/components/ui/KpiCard";
import { BarRow } from "@/components/ui/BarRow";
import { Pill } from "@/components/ui/Pill";
import { shapFigureUrl, getModels, getShap } from "@/lib/api";
import { classColor, modelColor, modelLabel } from "@/lib/colors";

export default function ShapPage() {
  const [models, setModels] = useState<string[]>([]);
  const [selected, setSelected] = useState("");
  const [shap, setShap] = useState<{
    overall: [string, number][];
    per_class: Record<string, [string, number][]>;
  } | null>(null);
  const [loadingShap, setLoadingShap] = useState(false);
  const [shapError, setShapError] = useState(false);
  const [cls, setCls] = useState("");

  useEffect(() => {
    let cancelled = false;
    getModels().catch(() => ({ models: [] as string[] })).then((d) => {
      if (cancelled) return;
      setModels(d.models);
      if (d.models[0]) setSelected(d.models[0]);
    });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!selected) return;
    setShap(null);
    setLoadingShap(true);
    setShapError(false);
    getShap(selected)
      .then((s) => {
        setShap(s);
        setLoadingShap(false);
        const classes = Object.keys(s.per_class ?? {});
        if (classes[0]) setCls(classes[0]);
      })
      .catch(() => {
        setLoadingShap(false);
        setShapError(true);
      });
  }, [selected]);

  const color = modelColor(selected);
  const overall = shap?.overall ?? [];
  const perClass = shap?.per_class ?? {};
  const classes = Object.keys(perClass);
  const classItems = perClass[cls] ?? [];
  const maxOverall = overall[0]?.[1] ?? 1;
  const maxClass = classItems[0]?.[1] ?? 1;
  const explainer = ["mlp", "logistic_regression"].includes(selected)
    ? "KernelExplainer"
    : "TreeExplainer";

  return (
    <AppShell title="SHAP Explainability">
      <div className="space-y-5 animate-fadeIn">

        {/* Header */}
        <div className="flex items-end justify-between flex-wrap gap-3">
          <div>
            <h1 className="text-[26px] font-semibold tracking-tight text-ink-0">SHAP Explainability</h1>
            <p className="text-[12px] text-ink-2 mt-1">
              TreeExplainer (RF/XGB/LGB/CatBoost) · KernelExplainer (MLP/LR) · why did the model decide that?
            </p>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {models.map((name) => {
              const c = modelColor(name);
              const isActive = name === selected;
              return (
                <button
                  key={name}
                  onClick={() => setSelected(name)}
                  className="px-3 h-8 rounded-lg text-[12px] font-medium transition ring-1"
                  style={{
                    background: isActive ? `${c}22` : "transparent",
                    color: isActive ? c : "var(--text-secondary)",
                    borderColor: isActive ? `${c}55` : "var(--border-base)",
                    boxShadow: isActive ? `0 0 12px ${c}33` : "none",
                  }}
                >
                  {modelLabel(name)}
                </button>
              );
            })}
          </div>
        </div>

        {/* KPIs */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <KpiCard
            label="Top-1 feature"
            value={overall[0]?.[0] ?? "—"}
            color={color}
            sub={overall[0] ? `mean |SHAP| = ${overall[0][1].toFixed(4)}` : undefined}
          />
          <KpiCard label="Classes explained" value={classes.length || "—"} color="#6366F1" />
          <KpiCard label="Explainer"         value={explainer}             color="#F59E0B" />
        </div>

        {shapError ? (
          <Panel>
            <div className="py-16 text-center text-ink-2 text-[12px]">
              Failed to load SHAP data for{" "}
              <span className="text-ink-1 font-medium">{modelLabel(selected)}</span>.
              Run{" "}
              <code className="font-mono text-[10.5px] py-0.5 px-[5px] rounded bg-surface-elevated ring-1 ring-line-base text-ink-1 inline-flex items-center">
                python main.py --stage shap
              </code>{" "}
              first.
            </div>
          </Panel>
        ) : !shap ? (
          <Panel>
            <div className="py-16 text-center text-ink-2 text-[12px]">
              <span className="inline-flex items-center gap-2">
                {loadingShap
                  ? <><span className="h-1.5 w-1.5 rounded-full bg-info animate-pulseDot" /> Loading SHAP data…</>
                  : "Select a model above"
                }
              </span>
            </div>
          </Panel>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Overall importance */}
              <Panel>
                <PanelHeader
                  eyebrow="Global"
                  title="Feature importance"
                  sub="Mean |SHAP| across all classes — higher = more influential"
                />
                <div className="px-5 py-4 space-y-2.5">
                  {[...overall].reverse().map(([feat, val]) => (
                    <BarRow
                      key={feat}
                      label={feat}
                      value={val}
                      max={maxOverall}
                      color={color}
                      suffix=""
                      right={
                        <span className="tabular-nums text-[10.5px] text-ink-2 w-12 text-right font-mono">
                          {val.toFixed(4)}
                        </span>
                      }
                    />
                  ))}
                </div>
              </Panel>

              {/* Per-class */}
              <Panel>
                <PanelHeader
                  eyebrow="Per-class"
                  title="Class drill-down"
                  right={
                    <div className="flex flex-wrap gap-1">
                      {classes.map((c) => (
                        <button
                          key={c}
                          onClick={() => setCls(c)}
                          className="px-2 py-0.5 rounded text-[10.5px] font-medium transition"
                          style={{
                            background: c === cls ? `${classColor(c)}22` : "transparent",
                            color: c === cls ? classColor(c) : "var(--text-secondary)",
                            boxShadow:
                              c === cls ? `inset 0 0 0 1px ${classColor(c)}44` : "none",
                          }}
                        >
                          {c}
                        </button>
                      ))}
                    </div>
                  }
                />
                <div className="px-5 py-4 space-y-2.5">
                  {classItems.length === 0 ? (
                    <p className="text-[12px] text-ink-2">No data for this class.</p>
                  ) : (
                    [...classItems].reverse().map(([feat, val]) => (
                      <BarRow
                        key={feat}
                        label={feat}
                        value={val}
                        max={maxClass}
                        color={classColor(cls)}
                        suffix=""
                        right={
                          <span className="tabular-nums text-[10.5px] text-ink-2 w-12 text-right font-mono">
                            {val.toFixed(4)}
                          </span>
                        }
                      />
                    ))
                  )}
                </div>
              </Panel>
            </div>

            {/* Explainer info */}
            <Panel>
              <PanelHeader
                eyebrow="Method"
                title={`${explainer} explanation`}
                sub={`Model: ${modelLabel(selected)} · ${overall.length} features analysed`}
                right={<Pill tone="muted" size="sm">{selected}</Pill>}
              />
              <div className="px-5 py-4 grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <div className="text-[11px] text-ink-1 leading-relaxed mb-4">
                    <strong className="text-ink-0">How to read this chart:</strong> Each bar shows the mean absolute SHAP value for a feature — how much it shifts the model output.{" "}
                    {explainer === "TreeExplainer"
                      ? "TreeExplainer computes exact SHAP values using the tree structure directly (fast, exact)."
                      : "KernelExplainer approximates SHAP values by perturbing feature values (model-agnostic but slower)."}
                  </div>
                  <div className="rounded-lg bg-surface ring-1 ring-line-subtle p-3.5">
                    <div className="text-[10px] uppercase tracking-[.14em] text-ink-2 mb-2">In plain English</div>
                    <p className="text-[12px] text-ink-1 leading-relaxed">
                      The model uses{" "}
                      <span className="text-ink-0 font-medium">{overall[0]?.[0] ?? "—"}</span>{" "}
                      as its most important signal.{" "}
                      {overall[1] && <>
                        <span className="text-ink-0 font-medium">{overall[1][0]}</span> is second.{" "}
                      </>}
                      Features with high SHAP values push predictions strongly toward attack classes.
                    </p>
                  </div>
                </div>

                <div>
                  <div className="text-[10.5px] uppercase tracking-[.14em] text-ink-2 mb-2">Top features by SHAP rank</div>
                  <div className="space-y-1.5">
                    {overall.slice(0, 8).map(([feat, val], i) => (
                      <div key={feat} className="flex items-center gap-2.5 text-[11.5px]">
                        <span className="tabular-nums text-ink-3 w-5 text-right font-mono">{i + 1}</span>
                        <span className="flex-1 text-ink-1 truncate">{feat}</span>
                        <div className="h-0.5 w-12 rounded-full bg-surface-elevated overflow-hidden">
                          <div
                            className="h-full rounded-full"
                            style={{ width: `${(val / maxOverall) * 100}%`, background: color }}
                          />
                        </div>
                        <span className="tabular-nums text-ink-2 font-mono text-[10.5px] w-12 text-right">
                          {val.toFixed(4)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </Panel>

            {/* Beeswarm plots */}
            {classes.slice(0, 1).map((c) => {
              const slug =
                c.replace(/[^a-z0-9]/gi, "_").replace(/_+/g, "_").replace(/^_|_$/g, "") || "class";
              const png = shapFigureUrl(selected, `summary_${slug}.png`);
              return (
                <Panel key={c}>
                  <PanelHeader eyebrow="Beeswarm" title={`SHAP summary plot — ${c}`} />
                  <div className="p-4">
                    <img
                      src={png}
                      alt={`SHAP beeswarm ${c}`}
                      className="w-full rounded-lg"
                      onError={(e) => {
                        const el = e.target as HTMLImageElement;
                        if (el.parentElement) el.parentElement.style.display = "none";
                      }}
                    />
                  </div>
                </Panel>
              );
            })}
          </>
        )}
      </div>
    </AppShell>
  );
}
