/**
 * Renders a value the bundle never recorded.
 *
 * This exists so "absent" is impossible to confuse with a real measurement.
 * The two bundles record different subsets of the metric set — the 2018
 * pipeline has no cross-validation or label-shuffle control, the 2017 one has
 * no bootstrap confidence intervals — and a dash that says why is far more
 * honest than a zero that reads as a perfect score.
 */

import { clsx } from "clsx";
import { isAbsent } from "@/lib/bundles";

export function Nil({ reason }: { reason?: string }) {
  return (
    <span
      title={reason ?? "This bundle did not record a value for this field"}
      className="text-ink-3 font-mono cursor-help select-none"
    >
      —
    </span>
  );
}

/**
 * Show `children` when the value exists, otherwise the absent marker.
 * Keeps the null check next to the render instead of scattered through pages.
 */
export function Value({
  of,
  reason,
  className,
  children,
}: {
  of: unknown;
  reason?: string;
  className?: string;
  children: React.ReactNode;
}) {
  if (isAbsent(of)) return <Nil reason={reason} />;
  return <span className={clsx("font-mono tabular-nums", className)}>{children}</span>;
}
