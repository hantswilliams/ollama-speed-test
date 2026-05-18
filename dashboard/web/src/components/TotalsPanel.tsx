"use client";

import { usePolling } from "./usePolling";

type Bucket = {
  model: string;
  requests: number;
  output_tokens: number;
  prompt_tokens: number;
  avg_output_tps: number | null;
};

type StatsResponse = { today: Bucket[]; week: Bucket[]; all: Bucket[] };

function sum(b: Bucket[], key: "requests" | "output_tokens" | "prompt_tokens"): number {
  return b.reduce((acc, x) => acc + (x[key] ?? 0), 0);
}

function fmt(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "k";
  return n.toString();
}

function Kpi({ label, value, unit }: { label: string; value: string; unit?: string }) {
  return (
    <div className="kpi">
      <div className="label">{label}</div>
      <div className="value">
        {value}
        {unit && <span className="unit">{unit}</span>}
      </div>
    </div>
  );
}

export default function TotalsPanel() {
  const data = usePolling<StatsResponse>("/api/stats", 5000, { today: [], week: [], all: [] });
  const buckets: Array<["Today" | "7 days" | "All-time", Bucket[]]> = [
    ["Today", data.today],
    ["7 days", data.week],
    ["All-time", data.all],
  ];
  return (
    <div className="card">
      <h2>Cumulative usage</h2>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
        {buckets.map(([label, b]) => (
          <div
            key={label}
            style={{
              border: "1px solid #1f252c",
              borderRadius: 8,
              padding: 14,
              background: "#0f1317",
            }}
          >
            <div className="sub" style={{ marginBottom: 12, color: "#8a9099", fontSize: 12, textTransform: "uppercase", letterSpacing: "0.06em" }}>
              {label}
            </div>
            <div style={{ display: "flex", gap: 24, marginBottom: 14 }}>
              <Kpi label="Requests" value={fmt(sum(b, "requests"))} />
              <Kpi label="Output tokens" value={fmt(sum(b, "output_tokens"))} />
              <Kpi label="Prompt tokens" value={fmt(sum(b, "prompt_tokens"))} />
            </div>
            <div className="muted" style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>
              By model
            </div>
            {b.length === 0 ? (
              <div className="muted" style={{ fontSize: 12 }}>no data yet</div>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th style={{ textAlign: "right" }}>Reqs</th>
                    <th style={{ textAlign: "right" }}>Out tok</th>
                    <th style={{ textAlign: "right" }}>Avg tok/s</th>
                  </tr>
                </thead>
                <tbody>
                  {b.map((row) => (
                    <tr key={row.model}>
                      <td title={row.model} style={{ maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {row.model}
                      </td>
                      <td className="num">{row.requests}</td>
                      <td className="num">{fmt(row.output_tokens)}</td>
                      <td className="num">{row.avg_output_tps?.toFixed(1) ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
