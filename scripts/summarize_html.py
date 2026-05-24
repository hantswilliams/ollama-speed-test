"""Generate outputs/summary.html from all benchmark data in outputs/.

Sources, in order of precedence:
  1. outputs/benchmarks.db (canonical store; populated by current scripts)
  2. outputs/*_raw*.csv    (older runs that predate the DB)

Rows are deduped on (batch_id, model, prompt_name, run). Output is a single
self-contained HTML file — no JS, inline CSS, no external assets.

Usage:
    python3 scripts/summarize_html.py
"""

import html as htmlmod
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
DB_PATH = OUTPUT_DIR / "benchmarks.db"
HTML_PATH = OUTPUT_DIR / "summary.html"


def load_db() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM benchmark_runs", conn)


def load_csvs() -> pd.DataFrame:
    frames = []
    for csv in sorted(OUTPUT_DIR.glob("*_raw*.csv")):
        try:
            df = pd.read_csv(csv)
        except Exception as exc:
            print(f"  skipping {csv.name}: {exc}")
            continue
        if "backend" not in df.columns:
            df["backend"] = "llama-server" if csv.name.startswith("llama_") else "ollama"
        if "batch_id" not in df.columns:
            df["batch_id"] = csv.stem
        if "batch_started_at" not in df.columns:
            df["batch_started_at"] = datetime.fromtimestamp(
                csv.stat().st_mtime, tz=timezone.utc
            ).isoformat()
        df["_source_file"] = csv.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def combine() -> pd.DataFrame:
    db = load_db()
    csvs = load_csvs()
    combined = pd.concat([db, csvs], ignore_index=True, sort=False)
    if combined.empty:
        return combined
    key = [c for c in ("batch_id", "model", "prompt_name", "run") if c in combined.columns]
    if key:
        combined = combined.drop_duplicates(subset=key, keep="first")
    return combined


def fmt_table(df: pd.DataFrame, classes: str = "") -> str:
    return df.to_html(index=False, float_format="%.2f", border=0, classes=classes, na_rep="—")


def render(df: pd.DataFrame) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if df.empty:
        body = "<p>No benchmark data found in <code>outputs/</code>. Run <code>test_ollama.py</code> or <code>test_llama.py</code> first.</p>"
        return _wrap(body, now)

    n_runs = len(df)
    n_models = df["model"].nunique() if "model" in df else 0
    n_backends = df["backend"].nunique() if "backend" in df else 1
    n_devices = df["hostname"].nunique() if "hostname" in df else 0

    summary = (
        df.groupby(["backend", "model", "category"], dropna=False)
        .agg(
            runs=("output_tps", "count"),
            avg_tps=("output_tps", "mean"),
            min_tps=("output_tps", "min"),
            max_tps=("output_tps", "max"),
            avg_output_tokens=("output_tokens", "mean"),
            avg_wall_time_sec=("wall_time_sec", "mean"),
            last_run=("batch_started_at", "max"),
        )
        .reset_index()
        .sort_values("avg_tps", ascending=False, na_position="last")
    )

    detail_blocks = []
    for (backend, model), grp in df.groupby(["backend", "model"], dropna=False):
        per_prompt = (
            grp.groupby(["category", "prompt_name"], dropna=False)
            .agg(
                runs=("output_tps", "count"),
                avg_tps=("output_tps", "mean"),
                min_tps=("output_tps", "min"),
                max_tps=("output_tps", "max"),
                avg_wall_time_sec=("wall_time_sec", "mean"),
            )
            .reset_index()
            .sort_values(["category", "prompt_name"])
        )
        label = htmlmod.escape(f"{backend} / {model}")
        avg_tps = grp["output_tps"].mean()
        avg_tps_str = f"{avg_tps:.1f} tok/s avg" if pd.notna(avg_tps) else "no tps data"
        detail_blocks.append(
            f"<details><summary>{label} — {len(grp)} runs, {avg_tps_str}</summary>{fmt_table(per_prompt)}</details>"
        )

    device_block = ""
    if "hostname" in df.columns and df["hostname"].notna().any():
        device_cols = [c for c in ("hostname", "device_model", "chip", "memory_gb", "os_version") if c in df.columns]
        devices = (
            df[device_cols + ["output_tps"]]
            .groupby(device_cols, dropna=False)
            .agg(runs=("output_tps", "count"), avg_tps=("output_tps", "mean"))
            .reset_index()
            .sort_values("runs", ascending=False)
        )
        device_block = f"<h2>Devices</h2>{fmt_table(devices)}"

    body = f"""
<section class="hero">
  <div><strong>{n_runs}</strong>runs</div>
  <div><strong>{n_models}</strong>models</div>
  <div><strong>{n_backends}</strong>backends</div>
  <div><strong>{n_devices}</strong>devices</div>
</section>
<h2>Summary by model and category</h2>
<p class="hint">Sorted by avg output tokens/sec, descending — fastest models first.</p>
{fmt_table(summary, classes="summary")}
{device_block}
<h2>Per-prompt detail</h2>
<p class="hint">Click a model to expand its prompt-level breakdown.</p>
{''.join(detail_blocks)}
"""
    return _wrap(body, now)


def _wrap(body: str, now: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LLM Benchmark Summary</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 2rem; max-width: 1150px; color: #222; line-height: 1.45; }}
  h1 {{ margin-bottom: 0.2rem; }}
  h2 {{ margin-top: 2rem; border-bottom: 1px solid #eee; padding-bottom: 0.3rem; }}
  .subtitle {{ color: #666; margin-bottom: 1.2rem; font-size: 0.9rem; }}
  .hint {{ color: #777; font-size: 0.85rem; margin: 0.2rem 0 0.8rem; }}
  .hero {{ display: flex; gap: 1rem; margin: 1rem 0 1.5rem; flex-wrap: wrap; }}
  .hero div {{ background: #f4f5f7; padding: 0.6rem 1rem; border-radius: 6px;
              min-width: 90px; text-align: center; }}
  .hero strong {{ font-size: 1.5rem; display: block; color: #0b5fff; }}
  table {{ border-collapse: collapse; margin: 0.4rem 0 1rem;
           font-size: 0.88rem; width: 100%; }}
  th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #f7f7f9; font-weight: 600; }}
  table.summary tbody tr:hover {{ background: #fafbfc; }}
  details {{ margin: 0.4rem 0; border: 1px solid #eee; border-radius: 6px;
             padding: 0 0.8rem; }}
  summary {{ cursor: pointer; padding: 0.6rem 0; font-weight: 600;
             list-style-position: outside; }}
  code {{ background: #f4f5f7; padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>LLM Benchmark Summary</h1>
<p class="subtitle">Generated {now} · sources: <code>benchmarks.db</code> + <code>*_raw*.csv</code></p>
{body}
</body>
</html>
"""


def main() -> None:
    df = combine()
    HTML_PATH.write_text(render(df))
    rows = len(df) if not df.empty else 0
    print(f"Wrote {HTML_PATH}  ({rows} runs from {df['_source_file'].nunique() if '_source_file' in df else '?'} CSVs + DB)")


if __name__ == "__main__":
    main()
