"use client";

import { usePolling } from "./usePolling";

type Row = {
  id: number;
  timestamp: string;
  model: string;
  endpoint: string;
  streamed: number;
  prompt_tokens: number | null;
  output_tokens: number | null;
  output_tps: number | null;
  total_duration_ns: number | null;
  wall_time_sec: number | null;
};
type Resp = { rows: Row[] };

function fmtTime(iso: string): string {
  return new Date(iso).toLocaleTimeString();
}

function fmtDur(ns: number | null): string {
  if (!ns) return "—";
  return (ns / 1_000_000_000).toFixed(2) + "s";
}

export default function RequestsTable() {
  const { rows } = usePolling<Resp>("/api/requests?limit=50", 3000, { rows: [] });

  if (rows.length === 0) {
    return <div className="empty">no requests logged yet</div>;
  }

  return (
    <table>
      <thead>
        <tr>
          <th>Time</th>
          <th>Model</th>
          <th>Endpoint</th>
          <th></th>
          <th style={{ textAlign: "right" }}>Prompt tok</th>
          <th style={{ textAlign: "right" }}>Output tok</th>
          <th style={{ textAlign: "right" }}>Out tok/s</th>
          <th style={{ textAlign: "right" }}>Total</th>
          <th style={{ textAlign: "right" }}>Wall</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.id}>
            <td className="muted">{fmtTime(r.timestamp)}</td>
            <td title={r.model} style={{ maxWidth: 280, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {r.model}
            </td>
            <td><span className="pill">{r.endpoint}</span></td>
            <td>{r.streamed ? <span className="pill">stream</span> : null}</td>
            <td className="num">{r.prompt_tokens ?? "—"}</td>
            <td className="num">{r.output_tokens ?? "—"}</td>
            <td className="num">{r.output_tps?.toFixed(1) ?? "—"}</td>
            <td className="num">{fmtDur(r.total_duration_ns)}</td>
            <td className="num">{r.wall_time_sec?.toFixed(2) ?? "—"}s</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
