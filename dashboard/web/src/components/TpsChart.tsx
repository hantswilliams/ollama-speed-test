"use client";

import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from "recharts";
import { usePolling } from "./usePolling";

type Point = { timestamp: string; model: string; output_tps: number | null };
type Resp = { series: Point[] };

const COLORS = ["#5aa9ff", "#ffb84d", "#7ee787", "#ff6b6b", "#c792ea", "#ffd43b"];

export default function TpsChart() {
  const { series } = usePolling<Resp>("/api/timeseries?limit=200", 3000, { series: [] });

  if (series.length === 0) {
    return <div className="empty">no requests logged yet — point a client at <code>localhost:11435</code></div>;
  }

  const models = Array.from(new Set(series.map((p) => p.model)));
  const data = series.map((p, idx) => {
    const point: Record<string, number | string> = {
      idx,
      label: new Date(p.timestamp).toLocaleTimeString(),
    };
    point[p.model] = p.output_tps ?? 0;
    return point;
  });

  return (
    <div style={{ width: "100%", height: 280 }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#1f252c" strokeDasharray="3 3" />
          <XAxis dataKey="label" stroke="#8a9099" fontSize={11} />
          <YAxis stroke="#8a9099" fontSize={11} width={40} />
          <Tooltip
            contentStyle={{ background: "#14181d", border: "1px solid #1f252c", borderRadius: 6 }}
            labelStyle={{ color: "#8a9099" }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {models.map((m, i) => (
            <Line
              key={m}
              type="monotone"
              dataKey={m}
              stroke={COLORS[i % COLORS.length]}
              strokeWidth={2}
              dot={{ r: 2 }}
              connectNulls
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
