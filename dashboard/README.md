# Ollama Dashboard

Live tracking of token usage and throughput across every Ollama call.

```
client ──> :11435 (proxy)  ──>  :11434 (ollama)
                │
                └──> SQLite (data/usage.db)
                          │
                          └──> :3030 (Next.js dashboard)
```

The proxy is a transparent forward proxy: it speaks the exact same Ollama API on port `11435`, forwards every request to the real Ollama on `11434`, and logs per-request stats (`eval_count`, `eval_duration`, model, timestamps, etc.) to `data/usage.db`. Anything you point at `11435` — OpenCode, the benchmark script, `curl`, OpenWebUI — gets logged.

## Setup

### 1. Proxy (Python)

```bash
cd dashboard/proxy
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
python main.py
```

> Don't have `uv`? Install with `brew install uv` (or see https://docs.astral.sh/uv/).

Proxy starts on `http://127.0.0.1:11435`.

### 2. Web dashboard (Next.js)

In a second terminal:

```bash
cd dashboard/web
npm install
npm run dev
```

Open `http://localhost:3030`.

> `better-sqlite3` is a native module. If `npm install` fails it usually means you're missing build tools (`xcode-select --install` on macOS).

## Pointing your tools at the proxy

The whole thing only collects data if your tools hit the proxy instead of Ollama directly.

- **OpenCode** — in `~/.config/opencode/opencode.json`, set `provider.ollama.options.baseURL` to `http://localhost:11435/v1` instead of `http://localhost:11434/v1`.
- **The benchmark script (`test.py`)** — change `OLLAMA_URL` to `http://localhost:11435/api/generate`.
- **Anything else** — replace `11434` with `11435` in the base URL.

Direct hits to `11434` bypass logging by design (so you can A/B if needed).

## What it shows

- **Cumulative usage** — totals for today / 7 days / all-time, broken down by model.
- **Output tok/s chart** — last 200 requests, one line per model.
- **Loaded models** — live `ollama ps` (which models are resident, VRAM, idle timer).
- **Recent requests** — last 50 requests with timestamps, models, token counts, throughput.

All panels poll every 3–5 seconds. No WebSockets, no SSE — just `fetch` on a timer.

## Schema

Single `requests` table in `data/usage.db`:

```sql
CREATE TABLE requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    model TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    streamed INTEGER NOT NULL,
    prompt_tokens INTEGER,
    output_tokens INTEGER,
    prompt_eval_duration_ns INTEGER,
    eval_duration_ns INTEGER,
    load_duration_ns INTEGER,
    total_duration_ns INTEGER,
    output_tps REAL,
    prompt_tps REAL,
    wall_time_sec REAL,
    client_ip TEXT
);
```

Open it directly with `sqlite3 dashboard/data/usage.db` for ad-hoc queries.

## Notes / caveats

- Only `/api/generate` and `/api/chat` are instrumented. Other endpoints (`/api/tags`, `/api/pull`, embeddings, etc.) pass through but aren't logged.
- The proxy parses streaming NDJSON responses to find the final `done: true` chunk — that's where Ollama puts the token counts. Streaming clients see no added latency; the parse happens alongside the byte-for-byte passthrough.
- The proxy doesn't authenticate. Don't expose `11435` to the network.
- SQLite is in WAL mode, so the dashboard can read while the proxy writes. Concurrent writers are not supported — run one proxy instance.
