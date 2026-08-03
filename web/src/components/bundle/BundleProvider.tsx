"use client";

/**
 * Holds the bundle the dashboard is currently showing.
 *
 * The selection has to survive navigation between pages, so it lives in the
 * URL (`?bundle=`) rather than in component state — that also makes a view of
 * a specific run shareable as a link, which matters when someone is arguing
 * about a number.
 */

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import type { BundleSummary } from "@/lib/bundles";
import { listBundles } from "@/lib/bundles";

type BundleContextValue = {
  bundles: BundleSummary[];
  active: BundleSummary | null;
  activeId: string | null;
  select: (id: string) => void;
  loading: boolean;
  error: string | null;
};

const BundleContext = createContext<BundleContextValue>({
  bundles: [],
  active: null,
  activeId: null,
  select: () => {},
  loading: true,
  error: null,
});

export function useBundle() {
  return useContext(BundleContext);
}

export function BundleProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const requested = params.get("bundle");

  const [bundles, setBundles] = useState<BundleSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listBundles()
      .then(({ bundles }) => {
        if (!cancelled) setBundles(bundles);
      })
      .catch((e: Error) => {
        // A dead API is a state the dashboard renders, not a crash: the
        // no-bundle screen explains what to run.
        if (!cancelled) setError(e.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Fall back to the first bundle when the URL names one that is not on disk,
  // which happens whenever a shared link outlives the run it pointed at.
  const active = useMemo(() => {
    if (!bundles.length) return null;
    return bundles.find((b) => b.id === requested) ?? bundles[0];
  }, [bundles, requested]);

  const select = useCallback(
    (id: string) => {
      const next = new URLSearchParams(params.toString());
      next.set("bundle", id);
      router.replace(`${pathname}?${next.toString()}`, { scroll: false });
    },
    [params, pathname, router],
  );

  const value = useMemo<BundleContextValue>(
    () => ({
      bundles,
      active,
      activeId: active?.id ?? null,
      select,
      loading,
      error,
    }),
    [bundles, active, select, loading, error],
  );

  return <BundleContext.Provider value={value}>{children}</BundleContext.Provider>;
}
