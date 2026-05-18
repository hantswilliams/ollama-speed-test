import { NextResponse } from "next/server";
import { db, tableExists } from "@/lib/db";

export const dynamic = "force-dynamic";

type Bucket = {
  model: string;
  requests: number;
  output_tokens: number;
  prompt_tokens: number;
  avg_output_tps: number | null;
};

function bucket(sinceIso: string | null): Bucket[] {
  const where = sinceIso ? "WHERE timestamp >= ?" : "";
  const stmt = db().prepare(
    `SELECT model,
            COUNT(*) as requests,
            COALESCE(SUM(output_tokens), 0) as output_tokens,
            COALESCE(SUM(prompt_tokens), 0) as prompt_tokens,
            AVG(output_tps) as avg_output_tps
     FROM requests
     ${where}
     GROUP BY model
     ORDER BY output_tokens DESC`
  );
  return (sinceIso ? stmt.all(sinceIso) : stmt.all()) as Bucket[];
}

export async function GET() {
  if (!tableExists()) {
    return NextResponse.json({ today: [], week: [], all: [] });
  }
  const now = new Date();
  const startOfToday = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate())).toISOString();
  const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString();
  return NextResponse.json({
    today: bucket(startOfToday),
    week: bucket(weekAgo),
    all: bucket(null),
  });
}
