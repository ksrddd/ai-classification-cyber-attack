const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`API ${path} → ${res.status}`);
  return res.json();
}

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
  return get<{ models: string[] }>("/api/models");
}

export async function getModelMetrics(name: string) {
  return get<Record<string, unknown>>(`/api/models/${name}/metrics`);
}

export async function getModelReport(name: string) {
  return get<Record<string, Record<string, number>>>(`/api/models/${name}/report`);
}

export async function getCompare() {
  return get<Record<string, Record<string, number>>>("/api/compare");
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
