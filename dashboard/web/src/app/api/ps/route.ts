import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const r = await fetch("http://localhost:11434/api/ps", { cache: "no-store" });
    if (!r.ok) return NextResponse.json({ models: [], error: `ollama returned ${r.status}` });
    const data = await r.json();
    return NextResponse.json({ models: data.models ?? [] });
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return NextResponse.json({ models: [], error: msg });
  }
}
