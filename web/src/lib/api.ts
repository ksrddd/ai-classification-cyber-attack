const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// Model metrics/comparison never change between retrains — cache 5 min on server
const STATIC_OPTS: RequestInit = { next: { revalidate: 300 } };
// Live data (model list) — revalidate every 60s so a new train run is picked up
const LIVE_OPTS: RequestInit = { next: { revalidate: 60 } };
// No fetch cache — used by pages that opt into force-dynamic
const FORCE_OPTS: RequestInit = { cache: "no-store" };

async function get<T>(path: string, opts: RequestInit = STATIC_OPTS): Promise<T> {
  const res = await fetch(`${BASE}${path}`, opts);
  if (!res.ok) throw new Error(`API ${path} → ${res.status}`);
  return res.json();
}

// ── Shared response types ────────────────────────────────────────────────────

export type PerClassStats = {
  precision: number;
  recall: number;
  f1: number;
  support?: number;
};

/** Flat metrics dict returned by /api/models/:name/metrics */
export type ModelMetrics = {
  accuracy: number;
  f1_weighted: number;
  f1_macro: number;
  precision_weighted: number;
  precision_macro: number;
  recall_weighted: number;
  recall_macro: number;
  roc_auc: number;
  matthews_corrcoef: number;
  per_class?: Record<string, PerClassStats>;
  [key: string]: unknown; // tolerate extra keys the backend may add
};

// ── API functions ────────────────────────────────────────────────────────────

export async function getOverview() {
  return get<{
    mode: string;
    labels: string[];
    row_counts: Record<string, number>;
    label_distribution: Record<string, number>;
    n_features: number;
    shape: number[];
    missing_total: number;
    infinite_total: number;
    duplicate_row_count: number;
    classification_mode: string;
    label_classes: string[];
  }>("/api/overview");
}

export async function getModels() {
  return get<{ models: string[] }>("/api/models", LIVE_OPTS);
}

/** Pass signal to cancel the in-flight request when the model selector changes. */
export async function getModelMetrics(name: string, signal?: AbortSignal) {
  return get<ModelMetrics>(`/api/models/${name}/metrics`, {
    ...STATIC_OPTS,
    signal,
  });
}

export async function getModelReport(name: string, signal?: AbortSignal) {
  return get<Record<string, Record<string, number>>>(`/api/models/${name}/report`, {
    ...STATIC_OPTS,
    signal,
  });
}

export async function getCompare(opts?: RequestInit) {
  return get<Record<string, ModelMetrics>>("/api/compare", opts ?? STATIC_OPTS);
}

export async function getShap(name: string) {
  return get<{
    overall: [string, number][];
    per_class: Record<string, [string, number][]>;
  }>(`/api/shap/${name}`);
}

export function figureUrl(filename: string) {
  return `${BASE}/api/figures/${filename}`;
}

export function shapFigureUrl(model: string, filename: string) {
  return `${BASE}/api/shap/${encodeURIComponent(model)}/figures/${encodeURIComponent(filename)}`;
}

export async function predictCsv(model: string, file: File) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/api/predict?model=${encodeURIComponent(model)}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Prediction failed");
  }
  return res.json() as Promise<{
    n_rows: number;
    class_counts: Record<string, number>;
    preview: Record<string, unknown>[];
    columns: string[];
  }>;
}

export { FORCE_OPTS };
