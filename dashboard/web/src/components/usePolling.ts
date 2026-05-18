"use client";

import { useEffect, useState } from "react";

export function usePolling<T>(url: string, intervalMs: number, initial: T): T {
  const [data, setData] = useState<T>(initial);
  useEffect(() => {
    let cancelled = false;
    async function tick() {
      try {
        const r = await fetch(url, { cache: "no-store" });
        if (!r.ok) return;
        const json = (await r.json()) as T;
        if (!cancelled) setData(json);
      } catch {
        // swallow — keep last good data
      }
    }
    tick();
    const t = setInterval(tick, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [url, intervalMs]);
  return data;
}
