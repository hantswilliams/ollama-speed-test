"""Generate outputs/summary.html from all benchmark data in outputs/.

Sources, in order of precedence:
  1. outputs/benchmarks.db (canonical store; populated by current scripts)
  2. outputs/*_raw*.csv    (older runs that predate the DB)

Per-model metadata (context window, params, file size, license) is fetched
once per unique model and cached in outputs/model_meta.json. Pass --no-fetch
to regenerate the HTML from cache only, with no network calls.

Rows are deduped on (batch_id, model, prompt_name, run). Output is a single
self-contained HTML file — no JS, inline CSS, no external assets.

Usage:
    python3 scripts/summarize_html.py            # fetch missing metadata
    python3 scripts/summarize_html.py --no-fetch # offline regenerate
"""

import argparse
import html as htmlmod
import json
import re
import shutil
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
DB_PATH = OUTPUT_DIR / "benchmarks.db"
HTML_PATH = OUTPUT_DIR / "summary.html"
META_CACHE_PATH = OUTPUT_DIR / "model_meta.json"

HF_API_BASE = "https://huggingface.co/api/models"
HF_REPO_BASE = "https://huggingface.co"
OLLAMA_LIBRARY_BASE = "https://ollama.com/library"

# Tried in order when a llama-server model string has no org/ prefix
HF_GUESS_ORGS = ("bartowski", "ggml-org", "lmstudio-community", "TheBloke", "unsloth")


# ----- data loading -------------------------------------------------------


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


# ----- metadata cache -----------------------------------------------------


def load_meta_cache() -> dict:
    if not META_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(META_CACHE_PATH.read_text())
    except json.JSONDecodeError:
        print(f"  warning: {META_CACHE_PATH} is corrupt, starting fresh")
        return {}


def save_meta_cache(cache: dict) -> None:
    META_CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def cache_key(backend: str, model: str) -> str:
    return f"{backend}::{model}"


# ----- metadata fetchers --------------------------------------------------


def parse_hf_model_string(model_str: str) -> tuple[str, str]:
    """Split 'org/repo:Quant' or 'repo:Quant' or 'repo' into (base, quant)."""
    if ":" in model_str:
        base, quant = model_str.rsplit(":", 1)
        return base, quant
    return model_str, ""


def resolve_hf_repo(base: str, timeout: float = 5.0) -> str | None:
    """If base already has 'org/', return it; otherwise probe common orgs."""
    if "/" in base:
        return base
    for org in HF_GUESS_ORGS:
        candidate = f"{org}/{base}"
        try:
            r = requests.get(f"{HF_API_BASE}/{candidate}", timeout=timeout)
            if r.status_code == 200:
                return candidate
        except requests.RequestException:
            continue
    return None


def humanize_params(total: int | None) -> str:
    if not total:
        return ""
    if total >= 1_000_000_000:
        return f"{total / 1_000_000_000:.1f}B"
    if total >= 1_000_000:
        return f"{total / 1_000_000:.1f}M"
    return str(total)


def humanize_gb(bytes_: int | None) -> str:
    if not bytes_:
        return ""
    return f"{bytes_ / (1024**3):.2f}"


def fetch_hf_meta(model_str: str, timeout: float = 10.0) -> dict:
    """Fetch metadata for a llama-server / HF GGUF model."""
    base, quant = parse_hf_model_string(model_str)
    repo = resolve_hf_repo(base, timeout=timeout)
    out = {
        "backend": "llama-server",
        "model": model_str,
        "url": "",
        "architecture": "",
        "params": "",
        "context": "",
        "size_gb": "",
        "license": "",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    if not repo:
        out["error"] = f"could not resolve HF repo for {base!r}"
        return out

    out["url"] = f"{HF_REPO_BASE}/{repo}"
    try:
        r = requests.get(f"{HF_API_BASE}/{repo}", timeout=timeout)
        r.raise_for_status()
        info = r.json()
    except requests.RequestException as exc:
        out["error"] = f"HF API error: {exc}"
        return out

    gguf = info.get("gguf") or {}
    out["architecture"] = gguf.get("architecture", "")
    out["context"] = gguf.get("context_length", "")
    out["params"] = humanize_params(gguf.get("total"))
    license_ = (info.get("cardData") or {}).get("license") or ""
    if not license_:
        for tag in info.get("tags", []):
            if isinstance(tag, str) and tag.startswith("license:"):
                license_ = tag.split(":", 1)[1]
                break
    out["license"] = license_

    # File size for the specific quant — needs the tree endpoint
    if quant:
        try:
            tr = requests.get(f"{HF_API_BASE}/{repo}/tree/main", timeout=timeout)
            tr.raise_for_status()
            quant_lc = quant.lower()
            for f in tr.json():
                if f.get("type") != "file":
                    continue
                path = f.get("path", "")
                if path.lower().endswith(".gguf") and quant_lc in path.lower():
                    size = (f.get("lfs") or {}).get("size") or f.get("size")
                    if size:
                        out["size_gb"] = humanize_gb(size)
                    break
        except requests.RequestException:
            pass

    return out


def fetch_ollama_meta(model_tag: str, timeout: float = 10.0) -> dict:
    """Parse `ollama show <model>` for metadata."""
    base_name = model_tag.split(":", 1)[0]
    out = {
        "backend": "ollama",
        "model": model_tag,
        "url": f"{OLLAMA_LIBRARY_BASE}/{base_name}",
        "architecture": "",
        "params": "",
        "context": "",
        "size_gb": "",  # ollama doesn't surface file size directly
        "license": "",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    if not shutil.which("ollama"):
        out["error"] = "ollama CLI not on PATH"
        return out
    try:
        proc = subprocess.run(
            ["ollama", "show", model_tag],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        out["error"] = "ollama show timed out"
        return out
    if proc.returncode != 0:
        out["error"] = (proc.stderr or "ollama show failed").strip()[:200]
        return out

    section = None
    license_lines = []
    for raw in proc.stdout.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        # Section headers are two-space indented and unindented relative to fields
        if not raw.startswith("    ") and not raw.startswith("\t"):
            section = stripped.lower()
            continue
        if section == "model":
            m = re.match(r"(\S[\S ]*?)\s{2,}(\S.*)", stripped)
            if not m:
                continue
            key, value = m.group(1).lower(), m.group(2).strip()
            if key == "architecture":
                out["architecture"] = value
            elif key == "parameters":
                out["params"] = value
            elif key == "context length":
                try:
                    out["context"] = int(value)
                except ValueError:
                    out["context"] = value
        elif section == "license":
            license_lines.append(stripped)

    if license_lines:
        out["license"] = license_lines[0]
    return out


# ----- enrichment ---------------------------------------------------------


STALE_DAYS = 30


def is_fresh(entry: dict) -> bool:
    ts = entry.get("fetched_at")
    if not ts:
        return False
    try:
        when = datetime.fromisoformat(ts)
    except ValueError:
        return False
    return (datetime.now(timezone.utc) - when).days < STALE_DAYS


def enrich_metadata(df: pd.DataFrame, no_fetch: bool) -> dict:
    cache = load_meta_cache()
    if df.empty:
        return cache
    pairs = df[["backend", "model"]].drop_duplicates().itertuples(index=False, name=None)
    fetched = 0
    for backend, model in pairs:
        if pd.isna(backend) or pd.isna(model):
            continue
        key = cache_key(backend, model)
        existing = cache.get(key)
        if existing and is_fresh(existing) and not existing.get("error"):
            continue
        if no_fetch:
            continue
        print(f"  fetching: {key}")
        if backend == "ollama":
            cache[key] = fetch_ollama_meta(model)
        elif backend == "llama-server":
            cache[key] = fetch_hf_meta(model)
        else:
            continue
        fetched += 1
    if fetched:
        save_meta_cache(cache)
        print(f"  cached metadata for {fetched} model(s)")
    return cache


# ----- rendering helpers --------------------------------------------------


def get_meta(cache: dict, backend: str, model: str) -> dict:
    return cache.get(cache_key(backend, model), {})


def model_link_html(backend: str, model: str, meta: dict) -> str:
    text = htmlmod.escape(str(model))
    url = meta.get("url")
    if url:
        return f'<a href="{htmlmod.escape(url)}" target="_blank" rel="noopener">{text}</a>'
    return text


def fmt_table(df: pd.DataFrame, classes: str = "", escape: bool = True) -> str:
    return df.to_html(
        index=False, float_format="%.2f", border=0, classes=classes,
        na_rep="—", escape=escape,
    )


def build_models_table(df: pd.DataFrame, cache: dict) -> pd.DataFrame:
    rows = []
    for (backend, model), grp in df.groupby(["backend", "model"], dropna=False):
        if pd.isna(backend) or pd.isna(model):
            continue
        meta = get_meta(cache, backend, model)
        rows.append({
            "backend": backend,
            "model": model_link_html(backend, model, meta),
            "architecture": meta.get("architecture") or "",
            "params": meta.get("params") or "",
            "context": meta.get("context") or "",
            "size_gb": meta.get("size_gb") or "",
            "license": meta.get("license") or "",
            "runs": len(grp),
            "avg_tps": grp["output_tps"].mean(),
        })
    return pd.DataFrame(rows).sort_values("avg_tps", ascending=False, na_position="last")


def linkify_model_column(df: pd.DataFrame, cache: dict) -> pd.DataFrame:
    df = df.copy()
    df["model"] = df.apply(
        lambda r: model_link_html(r["backend"], r["model"], get_meta(cache, r["backend"], r["model"])),
        axis=1,
    )
    return df


# ----- main rendering ----------------------------------------------------


def render(df: pd.DataFrame, cache: dict) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if df.empty:
        body = "<p>No benchmark data found in <code>outputs/</code>. Run <code>test_ollama.py</code> or <code>test_llama.py</code> first.</p>"
        return _wrap(body, now)

    n_runs = len(df)
    n_models = df["model"].nunique() if "model" in df else 0
    n_backends = df["backend"].nunique() if "backend" in df else 1
    n_devices = df["hostname"].nunique() if "hostname" in df else 0

    models_table = build_models_table(df, cache)

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
    summary_linked = linkify_model_column(summary, cache)

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
        meta = get_meta(cache, backend, model)
        link = model_link_html(backend, model, meta)
        avg_tps = grp["output_tps"].mean()
        avg_tps_str = f"{avg_tps:.1f} tok/s avg" if pd.notna(avg_tps) else "no tps data"
        ctx = meta.get("context") or ""
        ctx_str = f" · {ctx} ctx" if ctx else ""
        summary_html = f"<summary>{htmlmod.escape(backend)} / {link} — {len(grp)} runs, {avg_tps_str}{ctx_str}</summary>"
        detail_blocks.append(f"<details>{summary_html}{fmt_table(per_prompt)}</details>")

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

<h2>Models</h2>
<p class="hint">Click a model name to open its HuggingFace or Ollama page. Metadata cached in <code>outputs/model_meta.json</code>; refresh with <code>python scripts/summarize_html.py</code>.</p>
{fmt_table(models_table, escape=False)}

<h2>Summary by model and category</h2>
<p class="hint">Sorted by avg output tokens/sec, descending — fastest models first.</p>
{fmt_table(summary_linked, classes="summary", escape=False)}

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
  th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #eee;
            vertical-align: top; }}
  th {{ background: #f7f7f9; font-weight: 600; }}
  table.summary tbody tr:hover {{ background: #fafbfc; }}
  a {{ color: #0b5fff; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
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
    parser = argparse.ArgumentParser(
        description="Generate outputs/summary.html from benchmark data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip network calls; render from cached metadata only.",
    )
    args = parser.parse_args()

    df = combine()
    if not df.empty:
        print(f"Loaded {len(df)} rows ({df['model'].nunique()} unique models).")
    cache = enrich_metadata(df, no_fetch=args.no_fetch)
    HTML_PATH.write_text(render(df, cache))
    rows = len(df) if not df.empty else 0
    csv_count = df["_source_file"].nunique() if "_source_file" in df else "?"
    print(f"Wrote {HTML_PATH}  ({rows} runs from {csv_count} CSVs + DB)")


if __name__ == "__main__":
    main()
