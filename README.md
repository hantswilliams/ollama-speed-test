# Ollama / Llama-server Speed Test

Benchmark local LLM inference across two runtimes — **Ollama** and **llama.cpp's `llama-server`** — on the same prompts, same hardware, same output schema, so the numbers compare apples-to-apples.

## Repository Structure

- `test_ollama.py` — benchmarks Ollama models (talks to `localhost:11434` or the dashboard proxy on `:11435`)
- `test_llama.py` — benchmarks a running llama-server instance (default `localhost:8080`)
- `outputs/` — CSV results per run, plus `benchmarks.db` SQLite store that accumulates every run across both backends and across devices
- `scripts/` — visualization helpers
  - `scripts/visualize.py` — text analysis of a CSV
  - `scripts/enhanced_visualize.py` — fuller report including device info
  - `scripts/create_charts.py` — saves matplotlib charts; auto-facets by host when multiple devices are in the data
- `dashboard/` — Next.js + FastAPI proxy that captures Ollama traffic live (see [dashboard/README.md](dashboard/README.md))
- `services/privacy_filter/` — OpenAI Privacy Filter local install + tests (separate concern, see [services/privacy_filter/README.md](services/privacy_filter/README.md))
- `launch_opencode_local.py` — launch OpenCode against a local Ollama model
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

Start `llama-server` separately first. Examples:

```bash
# Dense
llama-server -hf ggml-org/Qwen3.6-27B-MTP-GGUF \
    --spec-type draft-mtp --spec-draft-n-max 2

# MoE
llama-server -hf ggml-org/Qwen3.6-35B-A3B-MTP-GGUF \
    --spec-type draft-mtp --spec-draft-n-max 3
```

Then in another terminal:

```bash
# Auto-detects the loaded model via /v1/models
python3 test_llama.py

# Tagged output + custom label
python3 test_llama.py --model-label "qwen3.6-35b-A3B-MTP" --suffix "mtp-spec3"

# Custom URL or port
python3 test_llama.py --url http://localhost:8081
```

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

## Live dashboard

A FastAPI proxy + Next.js dashboard logs every Ollama request as it happens, with per-IP and per-model breakdowns. See [dashboard/README.md](dashboard/README.md). Not currently wired up for llama-server traffic.
