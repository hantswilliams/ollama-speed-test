# Ollama / Llama-server Speed Test

Benchmark local LLM inference across two runtimes — **Ollama** and **llama.cpp's `llama-server`** — on the same prompts, same hardware, same output schema, so the numbers compare apples-to-apples.

## TL;DR

**Launch OpenCode against a local model — one command per option:**

```bash
# RECOMMENDED for serious agentic work: llama-server + Qwen3.6-35B-A3B-MTP
# (~35 tok/s on M5 Pro). Largest of the three (35B-A3B MoE), MTP heads
# built into the GGUF accelerate generation, and tool calling via MCP is
# confirmed working in opencode (tested end-to-end with brain MCP).
python launch_opencode_local.py --backend llama-server \
    --hf ggml-org/Qwen3.6-35B-A3B-MTP-GGUF \
    --spec-type draft-mtp --spec-draft-n-max 3

# Fastest in-chat code path: llama-server + DeepSeek-Coder-V2-Lite Q4_K_M
# (~120 tok/s on M5 Pro). Pure code generation / explanation. Tool calling
# is weak/unreliable — pick this when you mostly want fast inline answers
# and don't need opencode to actually run commands or edit files.
python launch_opencode_local.py --backend llama-server \
    --hf bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF:Q4_K_M

# Ollama path: qwen3-coder 30B-A3B (~50 tok/s, supports tool / function
# calling natively. Easier model management — `ollama pull` once and it's
# tracked, no manual GGUF wrangling.)
python launch_opencode_local.py --model qwen3-coder:30b-a3b-q4_K_M

# or if somewhere else after activating the virtual env:
python /Users/hants/Development/Python/ollama-llama-speed-test/launch_opencode_local.py --model qwen3-coder:30b-a3b-q4_K_M
```

**Choosing between them:**

| Pick | When | Throughput | Tool calls |
|---|---|---|---|
| **Qwen3.6-35B-A3B-MTP** (llama-server) | **Default for agentic opencode work** — strongest reasoning, MCP-verified | ~35 tok/s | Yes (verified with MCP) |
| **DeepSeek-Coder-V2-Lite** (llama-server) | Fast in-chat code answers, no agentic actions | ~120 tok/s | Unreliable |
| **qwen3-coder:30b-a3b** (Ollama) | Agentic opencode with easy model management (no GGUF wrangling) | ~50 tok/s | Yes (native) |

> Tool-calling note: "MCP-verified" means we've end-to-end-tested it with a real MCP server in opencode (brain MCP, `brain_wiki_stats` call returned correct data). "Native" means the model family officially supports function calling but we haven't run a specific MCP test against this exact build. "Unreliable" means the model emits malformed tool calls or ignores tools in agentic flows.

The script auto-starts the inference server if it isn't already running, writes the right `~/.config/opencode/opencode.json`, and opens a Terminal.app window running `opencode` in your current directory.

**When you're done, shut the backends down:**

```bash
python launch_opencode_local.py --stop                    # kill llama-server
python launch_opencode_local.py --stop --include-ollama   # also stop `ollama serve`
```

**Switching models or RAM feels bloated?** Standalone reset tool — runs from anywhere, no project deps:

```bash
python services/llm_cleanup/cleanup.py                 # kill llama-server + evict Ollama models, show delta
python services/llm_cleanup/cleanup.py --include-ollama --purge   # full reset incl. macOS file cache
python services/llm_cleanup/cleanup.py --dry-run       # preview without changes
```

**Benchmark + view results:**

```bash
python test_llama.py --hf bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF:Q4_K_M
python test_ollama.py --models "qwen3-coder:30b-a3b-q4_K_M"
python scripts/summarize_html.py && open outputs/summary.html
```

## Repository Structure

- `test_ollama.py` — benchmarks Ollama models (talks to `localhost:11434` or the dashboard proxy on `:11435`)
- `test_llama.py` — benchmarks a running llama-server instance (default `localhost:8080`)
- `outputs/` — CSV results per run, plus `benchmarks.db` SQLite store that accumulates every run across both backends and across devices
- `scripts/` — analysis + reporting helpers
  - `scripts/visualize.py` — text analysis of a CSV
  - `scripts/enhanced_visualize.py` — fuller report including device info
  - `scripts/create_charts.py` — saves matplotlib charts; auto-facets by host when multiple devices are in the data
  - `scripts/summarize_html.py` — generates `outputs/summary.html` from the DB + all CSVs in `outputs/`
- `dashboard/` — Next.js + FastAPI proxy that captures Ollama traffic live (see [dashboard/README.md](dashboard/README.md))
- `services/privacy_filter/` — OpenAI Privacy Filter local install + tests (separate concern, see [services/privacy_filter/README.md](services/privacy_filter/README.md))
- `services/llm_cleanup/` — standalone "free up local RAM" tool: kills llama-server, evicts Ollama-loaded models, reports the delta. Run any time, no project deps (see [services/llm_cleanup/README.md](services/llm_cleanup/README.md))
- `launch_opencode_local.py` — launch OpenCode against a local model (Ollama by default; `--backend llama-server --hf <repo>` for llama.cpp)
- `launch_ollama_service.py` — expose Ollama on the LAN (see [LAN_SHARING.md](LAN_SHARING.md))

## How to Run

### Ollama benchmarks

```bash
# Default models
python3 test_ollama.py

# Specific models
python3 test_ollama.py --models "qwen3-coder:30b-a3b-q4_K_M" "gemma4:e4b"

# Tagged output files
python3 test_ollama.py --suffix "mytest"
```

### llama-server benchmarks

Two modes — pick whichever fits.

**Self-managed (recommended): the script launches `llama-server` and shuts it down when done.** Cleanup runs even on Ctrl+C or errors. Requires `llama-server` on your `PATH` (or pass `--llama-bin /full/path/to/llama-server`).

```bash
# Dense
python3 test_llama.py --hf ggml-org/Qwen3.6-27B-MTP-GGUF \
    --spec-type draft-mtp --spec-draft-n-max 2

# MoE
python3 test_llama.py --hf ggml-org/Qwen3.6-35B-A3B-MTP-GGUF \
    --spec-type draft-mtp --spec-draft-n-max 3
```

Each invocation writes its own CSVs (auto-suffixed from the loaded model name) and appends to `outputs/benchmarks.db` with a unique `batch_id`. The server's own logs go to `outputs/llama_server_<batch>.log` so you can debug startup failures.

**Externally managed: you run `llama-server` yourself in another terminal.**

```bash
# Terminal 1
llama-server -hf ggml-org/Qwen3.6-27B-MTP-GGUF \
    --spec-type draft-mtp --spec-draft-n-max 2

# Terminal 2 — auto-detects the loaded model via /v1/models
python3 test_llama.py
```

Other useful flags (apply to either mode):

```bash
# Custom label and filename suffix
python3 test_llama.py --hf … --model-label "qwen3.6-35b-A3B-MTP" --suffix "mtp-spec3"

# Custom URL / port (also tells the launched server to bind there)
python3 test_llama.py --hf … --url http://localhost:8081

# Forward arbitrary extra args to llama-server (repeatable, one token per use)
python3 test_llama.py --hf … --llama-extra --ctx-size --llama-extra 8192

# Longer wait for first-time HF model downloads (default 600s)
python3 test_llama.py --hf … --startup-timeout 1800
```

Running `python3 test_llama.py` with neither `--hf` nor a server already listening will print a one-line hint instead of a stack trace.

## Output

Three formats per run:

1. **Raw CSV** — every prompt × every run (15 rows per model, 3 runs × 5 prompts)
2. **Summary CSV** — aggregated by model + category
3. **`outputs/benchmarks.db`** — single SQLite store, append-only, both backends write here with a `backend` column. This is the right artifact for cross-run, cross-device, cross-backend comparison.

All outputs include hardware columns (hostname, device model, chip, CPU count, memory, OS) so you can compare across machines once you copy the DB around.

## Visualization

```bash
pip install matplotlib seaborn
python3 scripts/create_charts.py outputs/ollama_benchmark_summary.csv
```

When a CSV contains multiple hosts (e.g. you merged data from two devices), `create_charts.py` automatically switches the hue dimension to hostname so the comparison is visible at a glance.

### One-shot HTML summary

For a quick browser-friendly overview that pulls from **everything** in `outputs/` — the SQLite DB plus older CSVs — run:

```bash
python3 scripts/summarize_html.py             # fetch any missing model metadata, then render
python3 scripts/summarize_html.py --no-fetch  # offline regenerate from cache only
```

Single self-contained HTML file (no JS, no external assets):

- **Models reference table** — every unique model with clickable link (HuggingFace for llama-server rows, Ollama library for Ollama rows), architecture, parameter count, context window, file size (GB), and license.
- **Summary by backend × model × category** — sorted by avg tok/s descending. Model names link out to their source pages.
- **Devices** — per-host run counts and avg tok/s, useful when you merge DBs from multiple machines.
- **Per-prompt detail** — collapsible `<details>` per (backend, model) with per-prompt timings and context size in the header.

Per-model metadata (context, params, size, license) is fetched once per model — from the HuggingFace API for llama-server models, from `ollama show` locally for Ollama models — and cached in `outputs/model_meta.json`. Re-runs only fetch new or stale (>30 days) entries. Pass `--no-fetch` for fully-offline regeneration.

## Live dashboard

A FastAPI proxy + Next.js dashboard logs Ollama requests as they happen, with per-IP and per-model breakdowns. See [dashboard/README.md](dashboard/README.md).

By default `launch_opencode_local.py` (Ollama backend) connects opencode directly to Ollama, so the proxy doesn't see those requests. Pass `--via-proxy` to route through it for dashboard logging:

```bash
# Terminal 1 — start the proxy first (it's required when --via-proxy is on)
cd dashboard/proxy && python main.py

# Terminal 2
python launch_opencode_local.py --model qwen3-coder:30b-a3b-q4_K_M --via-proxy
```

Not currently wired up for llama-server traffic.
