import { NextResponse } from "next/server";
import { db, tableExists, type RequestRow } from "@/lib/db";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  if (!tableExists()) return NextResponse.json({ rows: [] });
  const { searchParams } = new URL(req.url);
  const limit = Math.min(parseInt(searchParams.get("limit") ?? "50", 10) || 50, 500);
  const rows = db()
    .prepare(
      `SELECT id, timestamp, model, endpoint, streamed,
              prompt_tokens, output_tokens, output_tps, prompt_tps,
              total_duration_ns, wall_time_sec
       FROM requests
       ORDER BY id DESC
       LIMIT ?`
    )
    .all(limit) as RequestRow[];
  return NextResponse.json({ rows });
}
