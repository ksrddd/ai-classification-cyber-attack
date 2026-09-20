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

/**
 * Which pipeline produced a bundle, and therefore what it can be asked for.
 *
 * `crossdataset` is not a training run: it has no per-model metric row and no
 * champion, so `models` arrives empty and the transfer payload lives in
 * `crossdataset` instead. Pages must branch on this before reading `models` —
 * see BundleGate, which renders the mismatch as a designed state rather than
 * letting a page draw an empty ranking table.
 */
export type BundleLayout =
  | "cicids2017"
  | "ids2018"
  | "crossdataset"
  | "audit";

export type BundleSummary = {
  id: string;
  dataset: string;
  layout: BundleLayout;
  n_models: number;
  n_classes: number;
  split_protocol: string | null;
  n_train: number | null;
  n_test: number | null;
  n_features: number | null;
  class_weighting: string | null;
  hp_tuned: boolean | null;
  /** Which question an audit asked. Null on every other layout. */
  audit_kind: AuditKind | null;
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

/** One paired comparison against the bundle's leading model. */
export type BundleSignificance = {
  delta_f1_macro: number | null;
  delta_ci_low: number | null;
  delta_ci_high: number | null;
  mcnemar_p: number | null;
  disagreeing_flows: number | null;
  leader_only_right: number | null;
  rival_only_right: number | null;
  /**
   * True only when the paired interval excludes zero AND McNemar rejects.
   * Null when the bundle ran no significance tests at all -- which is not the
   * same as "no difference found".
   */
  separable_from_leader: boolean | null;
};

/**
 * One (train corpus → test corpus) result for one model under one split mode.
 *
 * `self` marks the diagonal — trained and tested on the same corpus — which is
 * the ceiling the off-diagonal cells are measured against. Diagonal cells
 * carry no `recovered_pct`: it is 100% there by construction, and printing a
 * tautology in the same column as a measurement reads as the best score in
 * the table.
 */
export type TransferCell = {
  mode: string;
  model: string;
  train: string;
  test: string;
  self: boolean;
  f1_macro: number | null;
  /** Macro-F1 of always predicting the majority class on the test corpus. */
  floor: number | null;
  /** Macro-F1 of a model trained on the test corpus itself. */
  ceiling: number | null;
  gap: number | null;
  /** Percent of (ceiling − floor) the transfer recovered. Null on diagonals. */
  recovered_pct: number | null;
  /** Spread across the protocol's five seeds. */
  sd: number | null;
};

export type TransferPerClass = {
  mode: string;
  model: string;
  train: string;
  test: string;
  self: boolean;
  f1: Record<string, number | null>;
};

/** How much a random split inflates one corpus's ceiling over a temporal one. */
export type LeakageRow = {
  test: string;
  model: string;
  chronological: number | null;
  random: number | null;
  inflation: number | null;
};

export type SupportRow = {
  mode: string;
  test: string;
  per_class: Record<string, number | null>;
};

export type CrossDataset = {
  modes: string[];
  models: string[];
  classes: string[];
  /** Raw shape from corpora.json — row counts, cleaning tallies, class counts. */
  corpora: Record<string, Record<string, unknown>>;
  cells: TransferCell[];
  per_class: TransferPerClass[];
  leakage: LeakageRow[];
  support: SupportRow[];
};

/**
 * An audit bundle: evidence that a result can be believed, not a score.
 *
 * The four audits have no schema in common because they ask four different
 * questions, so `kind` discriminates and the union below is narrowed on it.
 * Nothing here ranks models — `models` arrives empty, as it does for the
 * cross-dataset layout, and for the same reason: reporting an empty ranking
 * would read as "these models scored nothing".
 */
export type AuditKind =
  | "adversarial_validation"
  | "feature_mapping"
  | "stacking_leakage"
  | "ablation";

/** One shared class under adversarial validation, with its ablation curve. */
export type AdversarialClass = {
  shared_class: string;
  n_per_side: number | null;
  n_seeds: number | null;
  auc: number | null;
  auc_sd: number | null;
  auc_per_seed: (number | null)[];
  /** Keyed by how many top-ranked features were removed before re-fitting. */
  ablation: Record<string, number | null>;
  top_features: { feature: string; importance: number | null; sd: number | null }[];
};

/**
 * The attempt to break an AUC of 1.0, per class.
 *
 * `model_ladder` runs from a single-threshold decision stump up to the
 * gradient-boosted ensemble; `null_ladder` is the identical procedure on one
 * corpus split into random halves, so a rung that separates nothing still has
 * to land at chance there.
 */
export type AdversarialDiagnosticClass = {
  shared_class: string;
  n_per_side: number | null;
  auc_mean: number | null;
  auc_sd: number | null;
  best_single_feature: string | null;
  best_single_feature_auc: number | null;
  best_single_feature_median: Record<string, number | null>;
  /** Columns whose 0.1st–99.9th percentile ranges do not overlap at all. */
  disjoint_columns: string[];
  rows_identical_across_corpora: number | null;
  rows_identical_across_split: number | null;
  near_duplicate_test_share: number | null;
  model_ladder: Record<string, number | null>;
  null_ladder: Record<string, Record<string, number | null>>;
};

/**
 * One boundary drawn through the data, and how well a model separates it.
 *
 * `kind` is what the AUC means. `cross` is the corpus boundary, `within` is a
 * capture boundary inside one corpus, and `null` is a random split — a
 * boundary known to mean nothing, which every procedure must fail to separate.
 */
export type ControlContrast = {
  auc: number | null;
  auc_sd: number | null;
  stump: number | null;
  kind: string;
  boundary: string;
  n_available_per_side: number | null;
};

/**
 * The within-dataset control: does a capture boundary inside one corpus
 * separate as well as the boundary between corpora?
 *
 * The random-halves null already showed the procedure cannot invent
 * separation. This asks the harder question, because a random split destroys
 * the file and time structure that the rival explanation lives in.
 */
/**
 * One contrast from the capture-day follow-up.
 *
 * `same tool, other day` holds the attack tool fixed and moves the capture
 * day; `other tool, same class` does the reverse. The pair exists because the
 * within-dataset control could not tell them apart on its own.
 */
export type CaptureDayContrast = {
  label: string;
  kind: string;
  boundary: string;
  n_per_side: number | null;
  auc: number | null;
  auc_sd: number | null;
  stump: number | null;
};

/**
 * One attack program measured on both sides, with the class-level contrast it
 * has to be read against.
 *
 * `same_tool` holds the program fixed and changes only the lab and the year.
 * `whole_class` is the ordinary cross-dataset contrast for the class that tool
 * belongs to, re-run at the same rows per side — without it the tool-matched
 * number would be compared against a figure from a different draw.
 */
export type ToolPair = {
  tool: string;
  shared_class: string;
  ids2017_label: string;
  ids2018_label: string;
  /** The two labels do not say the same word; the pairing rests on CIC's docs. */
  identity_from_documentation: boolean;
  n_per_side: number | null;
  same_tool: { auc: number | null; auc_sd: number | null; stump: number | null };
  whole_class: { auc: number | null; auc_sd: number | null; stump: number | null };
  delta: number | null;
};

export type WithinDatasetControl = {
  question: string | null;
  /** Null when the tool-matched control was never run. */
  tool_matched: {
    question: string | null;
    seeds: (number | null)[];
    tool_pairs: ToolPair[];
    different_tools_within_ids2017: {
      shared_class: string;
      tools: string[];
      n_per_side: number | null;
      auc: number | null;
      stump: number | null;
    }[];
  } | null;
  /** Null when the follow-up was never run. */
  capture_day: {
    question: string | null;
    corpus: string | null;
    min_per_side: number | null;
    same_tool_other_day: CaptureDayContrast[];
    other_tool_same_class: CaptureDayContrast[];
  } | null;
  verdict: string | null;
  reason: string | null;
  seeds: (number | null)[];
  min_per_side: number | null;
  separable_threshold: number | null;
  n_classes_judged: number | null;
  min_gap: number | null;
  max_within_auc: number | null;
  classes: {
    shared_class: string;
    /** Every contrast for a class runs at this many rows a side. */
    n_per_side: number | null;
    cross_auc: number | null;
    /** The highest within-dataset contrast, not the mean — the strongest rival. */
    within_auc_max: number | null;
    gap: number | null;
    contrasts: Record<string, ControlContrast>;
  }[];
};

export type AdversarialAudit = {
  kind: "adversarial_validation";
  seeds: (number | null)[];
  n_features: number | null;
  min_per_side: number | null;
  max_per_side: number | null;
  ablation_k: number[];
  classes: AdversarialClass[];
  ranking: {
    feature: string;
    mean: number | null;
    max: number | null;
    count: number | null;
  }[];
  /** Null when the run predates `scripts/diagnose_av.py` — not "passed". */
  diagnostics: {
    split: string | null;
    preprocessing: string | null;
    seeds: (number | null)[];
    max_per_side: number | null;
    near_dup_quantile: number | null;
    classes: AdversarialDiagnosticClass[];
  } | null;
  /** Null when `scripts/within_dataset_control.py` was never run. */
  control: WithinDatasetControl | null;
};

export type FeatureMappingAudit = {
  kind: "feature_mapping";
  n_rows: Record<string, number | null>;
  n_features: number | null;
  n_distinct_measurements: number | null;
  invariants: {
    invariant: string;
    verdict: string;
    rate_ids2017: number | null;
    rate_ids2018: number | null;
    columns: string[];
    note: string | null;
  }[];
  duplicate_groups: string[][];
  scale_audit: {
    feature: string;
    flag: string;
    reason: string;
    ids2017: Record<string, number | null>;
    ids2018: Record<string, number | null>;
  }[];
};

export type StackingLeakageAudit = {
  kind: "stacking_leakage";
  corpora: {
    dataset: string;
    label: string;
    run_utc: string | null;
    audit_rows: number | null;
    seed: number | null;
    passed: boolean;
    n_checks: number | null;
    n_failed: number | null;
    checks: {
      section: string;
      check: string;
      passed: boolean;
      detail: string;
    }[];
  }[];
};

export type AblationAudit = {
  kind: "ablation";
  summary: {
    k: number | null;
    mode: string;
    ceiling: number | null;
    transfer: number | null;
    gap: number | null;
    recovered_pct: number | null;
  }[];
  ks: number[];
  per_model: {
    model: string;
    /** Recovered percent, keyed by how many features were dropped. */
    recovered_pct: Record<string, number | null>;
    change: number | null;
  }[];
  /** The decision rule, recorded before any of these runs existed. */
  preregistration: string | null;
};

export type Audit =
  | AdversarialAudit
  | FeatureMappingAudit
  | StackingLeakageAudit
  | AblationAudit;

export type BundleDetail = {
  id: string;
  dataset: string;
  layout: BundleLayout;
  run: BundleRun;
  classes: string[];
  models: Record<string, BundleMetrics>;
  /** Empty when the bundle never ran paired tests. */
  significance: Record<string, BundleSignificance>;
  /** Present only on the `crossdataset` layout; null for training bundles. */
  crossdataset: CrossDataset | null;
  /** Present only on the `audit` layout; null everywhere else. */
  audit: Audit | null;
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
