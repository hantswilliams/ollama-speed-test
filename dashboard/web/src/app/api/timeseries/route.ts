import { NextResponse } from "next/server";
import { db, tableExists } from "@/lib/db";

export const dynamic = "force-dynamic";

type Row = { timestamp: string; model: string; output_tps: number | null };

export async function GET(req: Request) {
  if (!tableExists()) return NextResponse.json({ series: [] });
  const { searchParams } = new URL(req.url);
  const limit = Math.min(parseInt(searchParams.get("limit") ?? "200", 10) || 200, 2000);
  const rows = db()
    .prepare(
      `SELECT timestamp, model, output_tps
       FROM requests
       WHERE output_tps IS NOT NULL
       ORDER BY id DESC
       LIMIT ?`
    )
    .all(limit) as Row[];
  return NextResponse.json({ series: rows.reverse() });
}
