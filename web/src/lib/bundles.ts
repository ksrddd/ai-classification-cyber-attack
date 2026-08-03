/**
 * Client for the bundle-aware API.
 *
 * The dashboard renders results bundles produced by two different pipelines
 * (CICIDS2017 and CSE-CIC-IDS2018) whose on-disk layouts have almost nothing
 * in common. The backend normalises them; this module types the result.
 *
 * The single rule that governs every helper here: a metric the bundle did
 * not record arrives as `null` and must be rendered as absent. Formatting it
 * as 0 would turn "the false-positive rate was never measured" into "the
 * false-positive rate is zero", which reads as the best possible score.
 */

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const STATIC_OPTS: RequestInit = { next: { revalidate: 300 } };
const LIVE_OPTS: RequestInit = { next: { revalidate: 60 } };

async function get<T>(path: string, opts: RequestInit = STATIC_OPTS): Promise<T> {
  const res = await fetch(`${BASE}${path}`, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail ?? `API ${path} → ${res.status}`);
  }
  return res.json();
}

// ── Types ────────────────────────────────────────────────────────────────────

/** Every field is nullable on purpose — see the module docstring. */
export type BundleMetrics = {
  accuracy: number | null;
  balanced_accuracy: number | null;
  f1_macro: number | null;
  f1_weighted: number | null;
  precision_macro: number | null;
  precision_weighted: number | null;
  recall_macro: number | null;
  recall_weighted: number | null;
  mcc: number | null;
  binary_fpr: number | null;
  binary_recall: number | null;
  false_alarms_fp: number | null;
  missed_attacks_fn: number | null;
  train_seconds: number | null;
  predict_seconds: number | null;
  throughput_flows_per_sec: number | null;
  model_size_mb: number | null;
  cv_f1_macro_mean: number | null;
  cv_f1_macro_std: number | null;
  label_shuffle_f1_macro: number | null;
  hp_tuned: boolean | null;
  accelerator: string | null;
  f1_macro_ci_low: number | null;
  f1_macro_ci_high: number | null;
};

export type BundleSummary = {
  id: string;
  dataset: string;
  layout: "cicids2017" | "ids2018";
  n_models: number;
  n_classes: number;
  split_protocol: string | null;
  n_train: number | null;
  n_test: number | null;
  n_features: number | null;
  class_weighting: string | null;
  hp_tuned: boolean | null;
};

export type BundleRun = {
  run_name: string | null;
  split_protocol: string | null;
  random_state: number | null;
  n_train: number | null;
  n_test: number | null;
  n_features: number | null;
  n_classes: number | null;
  class_names: string[];
  per_class_n_train: Record<string, number>;
  per_class_n_test: Record<string, number>;
  majority_baseline_acc: number | null;
  hp_tuned: boolean | null;
  class_weighting: string | null;
  accelerator: string | null;
  dropped_columns?: string[];
};

export type BundleDetail = {
  id: string;
  dataset: string;
  layout: "cicids2017" | "ids2018";
  run: BundleRun;
  classes: string[];
  models: Record<string, BundleMetrics>;
};

export type PerClassRow = {
  class: string;
  precision: number;
  recall: number;
  "f1-score": number;
  support: number;
};

// ── Calls ────────────────────────────────────────────────────────────────────

export async function listBundles() {
  return get<{ bundles: BundleSummary[]; default: string | null }>(
    "/api/bundles",
    LIVE_OPTS,
  );
}

export async function getBundle(bundle?: string) {
  const q = bundle ? `?bundle=${encodeURIComponent(bundle)}` : "";
  return get<BundleDetail>(`/api/bundle${q}`);
}

export async function getBundleReport(model: string, bundle?: string) {
  const params = new URLSearchParams({ model });
  if (bundle) params.set("bundle", bundle);
  return get<{ model: string; rows: PerClassRow[] }>(
    `/api/bundle/report?${params}`,
  );
}

export async function getBundleConfusion(model: string, bundle?: string) {
  const params = new URLSearchParams({ model });
  if (bundle) params.set("bundle", bundle);
  return get<{ model: string; labels: string[]; rows: number[][] }>(
    `/api/bundle/confusion?${params}`,
  );
}

// ── Absent-aware formatting ──────────────────────────────────────────────────

/** Sentinel the UI renders wherever a bundle recorded nothing. */
export const NIL = "—"; // em dash

export function isAbsent(value: unknown): value is null | undefined {
  return value === null || value === undefined || Number.isNaN(value);
}

/** Fixed-point score, or the absent sentinel. Never returns "0.0000" for null. */
export function score(value: number | null | undefined, digits = 4): string {
  return isAbsent(value) ? NIL : (value as number).toFixed(digits);
}

/** Thousands-separated integer, or the absent sentinel. */
export function count(value: number | null | undefined): string {
  return isAbsent(value) ? NIL : (value as number).toLocaleString("en-US");
}

/** Percentage with one decimal, or the absent sentinel. */
export function percent(value: number | null | undefined, digits = 3): string {
  return isAbsent(value) ? NIL : `${((value as number) * 100).toFixed(digits)}%`;
}

/** Human duration, or the absent sentinel. */
export function duration(seconds: number | null | undefined): string {
  if (isAbsent(seconds)) return NIL;
  const s = seconds as number;
  if (s < 90) return `${s.toFixed(1)}s`;
  const h = Math.floor(s / 3600);
  const m = Math.round((s % 3600) / 60);
  return h > 0 ? `${h}h ${String(m).padStart(2, "0")}m` : `${m}m`;
}

/** Which of the requested fields this bundle has no value for. */
export function absentFields(
  metrics: BundleMetrics,
  fields: (keyof BundleMetrics)[],
): (keyof BundleMetrics)[] {
  return fields.filter((f) => isAbsent(metrics[f]));
}

/**
 * Rank models by a metric, skipping those that never recorded it.
 *
 * Models missing the metric are returned separately rather than sorted to
 * the bottom, because "not measured" is not the same as "worst".
 */
export function rankBy(
  models: Record<string, BundleMetrics>,
  metric: keyof BundleMetrics,
  descending = true,
): { ranked: [string, BundleMetrics][]; unmeasured: string[] } {
  const entries = Object.entries(models);
  const measured = entries.filter(([, m]) => !isAbsent(m[metric]));
  const unmeasured = entries.filter(([, m]) => isAbsent(m[metric])).map(([n]) => n);
  measured.sort(([, a], [, b]) => {
    const av = a[metric] as number;
    const bv = b[metric] as number;
    return descending ? bv - av : av - bv;
  });
  return { ranked: measured, unmeasured };
}
