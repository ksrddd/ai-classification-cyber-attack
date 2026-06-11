export const CLASS_COLORS: Record<string, string> = {
  BENIGN:      "#10B981",
  DoS:         "#EF4444",
  DDoS:        "#F43F5E",
  PortScan:    "#F59E0B",
  Bot:         "#A855F7",
  "Web Attack":"#EC4899",
  "Brute Force":"#F97316",
  Infiltration:"#6366F1",
  Heartbleed:  "#38BDF8",
  Other:       "#6C7488",
};

export const MODEL_COLORS: Record<string, string> = {
  random_forest:       "#10B981",
  xgboost:             "#3B82F6",
  lightgbm:            "#A855F7",
  catboost:            "#F59E0B",
  mlp:                 "#EC4899",
  logistic_regression: "#6C7488",
};

export const MODEL_LABELS: Record<string, string> = {
  random_forest:       "Random Forest",
  xgboost:             "XGBoost",
  lightgbm:            "LightGBM",
  catboost:            "CatBoost",
  mlp:                 "MLP Neural Net",
  logistic_regression: "Logistic Regression",
};

export function classColor(name: string): string {
  for (const [key, color] of Object.entries(CLASS_COLORS)) {
    if (name.toLowerCase().includes(key.toLowerCase()) || key.toLowerCase().includes(name.toLowerCase())) {
      return color;
    }
  }
  return CLASS_COLORS[name] ?? "#6C7488";
}

export function modelColor(name: string): string {
  return MODEL_COLORS[name] ?? "#22D3EE";
}

export function modelLabel(name: string): string {
  return MODEL_LABELS[name] ?? name.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

export function severityColor(sev: string): string {
  const map: Record<string, string> = {
    critical: "#F43F5E", danger: "#EF4444", warn: "#F59E0B", info: "#38BDF8", ok: "#10B981",
  };
  return map[sev] ?? map.info;
}
