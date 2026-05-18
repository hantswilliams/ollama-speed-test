"use client";

import { usePolling } from "./usePolling";

type LoadedModel = {
  name: string;
  model: string;
  size?: number;
  size_vram?: number;
  expires_at?: string;
};
type Resp = { models: LoadedModel[]; error?: string };

function fmtBytes(n?: number): string {
  if (!n) return "—";
  if (n >= 1024 ** 3) return (n / 1024 ** 3).toFixed(1) + " GB";
  if (n >= 1024 ** 2) return (n / 1024 ** 2).toFixed(0) + " MB";
  return n + " B";
}

function untilExpiry(iso?: string): string {
  if (!iso) return "—";
  const ms = new Date(iso).getTime() - Date.now();
  if (ms <= 0) return "expired";
  const s = Math.floor(ms / 1000);
  if (s < 60) return s + "s";
  const m = Math.floor(s / 60);
  if (m < 60) return m + "m " + (s % 60) + "s";
  const h = Math.floor(m / 60);
  return h + "h " + (m % 60) + "m";
}

export default function LoadedModelsPanel() {
  const data = usePolling<Resp>("/api/ps", 3000, { models: [] });

  if (data.error) return <div className="empty">Ollama not reachable: {data.error}</div>;
  if (data.models.length === 0) return <div className="empty">no models resident</div>;

  return (
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th style={{ textAlign: "right" }}>VRAM</th>
          <th style={{ textAlign: "right" }}>Idle in</th>
        </tr>
      </thead>
      <tbody>
        {data.models.map((m) => (
          <tr key={m.name}>
            <td title={m.name} style={{ maxWidth: 220, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {m.name}
            </td>
            <td className="num">{fmtBytes(m.size_vram ?? m.size)}</td>
            <td className="num">{untilExpiry(m.expires_at)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
